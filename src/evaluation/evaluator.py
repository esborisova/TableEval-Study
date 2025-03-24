import logging
from typing import List, Dict, Optional, Union
from tqdm.auto import tqdm
from models import LanguageModel
from tasks import TaskManager
from metrics import Metrics
from utils import load_samples, generate_prompt, generate_output_folder, dump_files
from transformers import AutoProcessor
import random
import torch
import gc
import numpy
import math

TableResults = Dict[str, str]

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(
        self,
        model: LanguageModel,
        tasks: Optional[List[Union[str, dict, object]]] = None,
        num_fewshot: Optional[Union[int, str]] = 0,
        batch_size: Optional[Union[int, str]] = None,
        random_seed: int = 0,
        log_samples: bool = False,
        log_logits: bool = False,
        use_chat_template: bool = False,
        current_datetime: str = "",
        output_path: str = "",
        save_columns: List = [],
        task_path: str = "./",
        data_base_path: None | str = None,
    ) -> None:
        """
        Instantiate and evaluate a model on a list of tasks.
        :param model: LM object, see models.py
        :param model_args: Optional[str, dict] String or dict arguments for each model class, see LM.create_from_arg_string and LM.create_from_arg_object. Ignored if `model` argument is a LM object.
        :param tasks: list[Union[str, dict, Task]] List of task names.
        :param num_fewshot: int Number of examples in few-shot context
        :param batch_size: int or str, optional Batch size for model
        :param random_seed: int Random seed for python's random module. If set to None, the seed will not be set.
        :param task_path: str Path to task YAML files.
        :param data_base_path: None | str Base path for datasets.
        """
        self.register = TaskManager(task_path=task_path)
        self.tasks_list = self.register.get_task(tasks)
        self.num_fewshot = int(num_fewshot)
        self.batch_size = int(batch_size)
        self.log_samples = log_samples
        self.model = model
        self.use_chat_template = use_chat_template
        self.current_datetime = current_datetime
        self.output_path = output_path
        self.log_logits = log_logits
        self.save_columns = save_columns
        self.data_base_path = data_base_path

        random.seed(random_seed)
        numpy.random.seed(random_seed)
        torch.manual_seed(random_seed)

    def simple_eval(self, task_name="", limit_samples: int = 0) -> Dict:

        if task_name:
            self.tasks_list = self.register.get_task(task_name)

        for task in self.tasks_list:
            task_results = []

            save_columns, num_fewshot, logits_folder = self.init_task(task)

            samples, few_shot_samples = self.load_all_samples(task, num_fewshot)

            if limit_samples > 0:
                logger.info(f"Limit samples to {limit_samples}")

                samples = samples.select(range(limit_samples))

            logger.info(f"Samples: {samples}")

            inputs = generate_prompt(
                samples, few_shot_samples, num_fewshot, task, self.use_chat_template
            )
            # run all samples
            for i in tqdm(range(0, len(inputs), self.batch_size)):
                outputs, logits = self.model(
                    inputs[i : i + self.batch_size],
                )

                task_results.extend(
                    self.log_results(
                        outputs,
                        logits,
                        samples,
                        inputs,
                        i,
                        task,
                        logits_folder,
                        save_columns,
                    )
                )
            self.reset(delete_model=True)
            results = self.get_scores(task_results, task)
            if "perplexity" in task["metric_list"]:
                results[task["task_name"]]["scores"]["perplexity"] = (
                    self.calc_perplexity(samples, inputs, task)
                )

        return results

    def get_scores(self, task_results, task):
        """calculate the scores for all metrics for the given task"""
        results = {}
        # calculation of the scores
        scores = {}
        metric_calc = Metrics(model_id=self.model.get_model_info())
        # load all results
        metric_calc.add(
            prediction=[x["prediction"] for x in task_results],
            reference=[x["reference"] for x in task_results],
        )
        # calculate the scores for the task and each metric
        for metric in task["metric_list"]:
            if "perplexity" == metric:
                continue
            metric_calc.metric_type(metric)
            scores[metric] = metric_calc.compute()

        # save results
        results[task["task_name"]] = {
            "scores": scores,
            "sample_logs": task_results if self.log_samples else None,
        }
        return results

    def calc_perplexity(self, samples, inputs, task):
        pp_model = self.model.load_model()
        pp_model.eval()
        loss = 0
        for input_text, sample in zip(inputs, samples):
            # Handle multimodal input (text + image)
            if "multi_modal_data" in task:
                inputs_processed = self.model.generate_inputs(
                    [input_text],
                )["input_ids"].cpu()
                labels_processed = self.model.processor.tokenizer(
                    [sample[task["doc_to_target"]]], return_tensors="pt"
                )["input_ids"]
            else:
                inputs_processed = self.model.generate_inputs(
                    input_text,
                )["input_ids"]
                labels_processed = self.model.generate_inputs(
                    sample[task["doc_to_target"]],
                )["input_ids"]
            concat_input = torch.cat(
                (inputs_processed, labels_processed),
                dim=1,
            ).to(self.model.device)
            with torch.no_grad():
                outputs = pp_model(
                    input_ids=concat_input,
                    labels=concat_input,
                )
            loss += outputs.loss.cpu()
        avg_loss = loss / len(samples)
        return math.exp(avg_loss)

    def init_task(self, task):
        # get save_columns
        if "save_columns" in task:
            save_columns = task["save_columns"].split(",")
        else:
            save_columns = self.save_columns

        # get safe files for score, logits and results
        _, _, logits_folder = generate_output_folder(
            self.output_path,
            model_name=self.model.get_model_info(),
            task_name=task["task_name"],
            current_datetime=self.current_datetime,
            log_logits=self.log_logits,
        )

        # get number of few-shot examples
        if "num_fewshot" in task.keys():
            num_fewshot = max(int(task["num_fewshot"]), self.num_fewshot)
        else:
            num_fewshot = self.num_fewshot

        return save_columns, num_fewshot, logits_folder

    def reset(self, delete_model:bool = False):
        gc.collect()  # Collect garbage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Free unused cached memory
            torch.cuda.ipc_collect()  # Collect internal shared memory (for multiprocessing)
        if delete_model:
            del self.model.model

        # Optional: Reset PyTorch’s CUDNN state
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.enabled = False

    def log_results(
        self,
        outputs,
        logits,
        samples,
        inputs,
        current_index,
        task,
        logits_folder,
        save_columns,
    ):

        # generate tuple for each output and sample
        if self.log_logits:
            dump_files(
                logits_folder, [logits_step.cpu() for logits_step in logits], "logits"
            )
        if self.log_samples:
            return [
                {
                    "prediction": outputs[j],
                    "reference": samples[current_index + j][task["doc_to_target"]],
                    "input": (
                        inputs[current_index + j]
                        if (
                            isinstance(inputs[current_index + j], str)
                            and not self.use_chat_template
                        )
                        else inputs[current_index + j][-1]
                    ),
                    "example": (
                        {x: samples[current_index + j][x] for x in save_columns}
                        if save_columns
                        else samples[current_index + j]
                    ),
                }
                for j in range(0, len(outputs))
            ]
        else:
            return [
                {
                    "prediction": outputs[j],
                    "reference": samples[current_index + j][task["doc_to_target"]],
                }
                for j in range(0, len(outputs))
            ]

    def load_all_samples(self, task, num_fewshot):
        # load the evaluation and few_shot samples
        if task["test_split"]:
            samples = load_samples(task["path"], task["test_split"], base_path=self.data_base_path)
            if num_fewshot > 0:
                if task["validation_split"]:
                    few_shot_samples = load_samples(
                        task["path"], task["validation_split"], base_path=self.data_base_path
                    )
                elif task["train_split"]:
                    few_shot_samples = load_samples(task["path"], task["train_split"], base_path=self.data_base_path)
                else:
                    few_shot_samples = None
            else:
                few_shot_samples = None
        else:
            samples = load_samples(task["path"], task["validation_split"], base_path=self.data_base_path)
            if num_fewshot > 0:
                if task["train_split"]:
                    few_shot_samples = load_samples(task["path"], task["train_split"], base_path=self.data_base_path)
                else:
                    few_shot_samples = None
            else:
                few_shot_samples = None
        # generate the input prompts
        return samples, few_shot_samples
