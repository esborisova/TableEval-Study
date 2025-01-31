from typing import List, Dict, Optional, Union
from tqdm.auto import tqdm
from models import LanguageModel
from tasks import TaskManager
from metrics import Metrics
from utils import load_samples, generate_prompt
import random
import torch
import numpy

TableResults = Dict[str, str]


class Evaluator:
    def __init__(
        self,
        model: LanguageModel,
        tasks: Optional[List[Union[str, dict, object]]] = None,
        num_fewshot: Optional[Union[int, str]] = 0,
        batch_size: Optional[Union[int, str]] = None,
        random_seed: int = 0,
        log_samples: bool = False,
        use_chat_template: bool = False,
    ) -> None:
        """
        Instantiate and evaluate a model on a list of tasks.
        :param model: LM object, see models.py
        :param model_args: Optional[str, dict] String or dict arguments for each model class, see LM.create_from_arg_string and LM.create_from_arg_object. Ignored if `model` argument is a LM object.
        :param tasks: list[Union[str, dict, Task]] List of task names.
        :param num_fewshot: int Number of examples in few-shot context
        :param batch_size: int or str, optional Batch size for model
        :param random_seed: int Random seed for python's random module. If set to None, the seed will not be set.
        """
        self.register = TaskManager()
        self.tasks_list = self.register.get_task(tasks)
        self.num_fewshot = int(num_fewshot)
        self.batch_size = int(batch_size)
        self.log_samples = log_samples
        self.model = model
        self.use_chat_template = use_chat_template
        random.seed(random_seed)
        numpy.random.seed(random_seed)
        torch.manual_seed(random_seed)

    def simple_eval(self) -> Dict:
        results = {}
        for task in self.tasks_list:

            if "num_fewshot" in task.keys():
                num_fewshot = max(int(task["num_fewshot"]), self.num_fewshot)
            else:
                num_fewshot = self.num_fewshot
            task_results = []
            # load the evaluation and few_shot samples
            if task["test_split"]:
                samples = load_samples(task["path"], task["test_split"])
                if num_fewshot > 0:
                    if task["validation_split"]:
                        few_shot_samples = load_samples(
                            task["path"], task["validation_split"]
                        )
                    elif task["train_split"]:
                        few_shot_samples = load_samples(
                            task["path"], task["train_split"]
                        )
                    else:
                        few_shot_samples = None
                else:
                    few_shot_samples = None
            else:
                samples = load_samples(task["path"], task["validation_split"])
                if num_fewshot > 0:
                    if task["train_split"]:
                        few_shot_samples = load_samples(
                            task["path"], task["train_split"]
                        )
                    else:
                        few_shot_samples = None
                else:
                    few_shot_samples = None
            # generate the input prompts
            inputs = generate_prompt(
                samples, few_shot_samples, num_fewshot, task, self.use_chat_template
            )

            # run all samples
            for i in tqdm(range(0, len(inputs), self.batch_size)):
                outputs, logits = self.model(inputs[i : i + self.batch_size])
                # generate tuple for each output and sample
                task_results.extend(
                    [
                        {
                            "prediction": outputs[j],
                            "reference": samples[i + j][task["doc_to_target"]],
                            "input": (
                                inputs[i + j]
                                if (
                                    isinstance(inputs[i + j], str)
                                    and not self.use_chat_template
                                )
                                else inputs[i + j][-1]
                            ),
                            "example": samples[i + j],
                            "logits": [
                                logits_step[j].cpu().tolist() for logits_step in logits
                            ],
                        }
                        for j in range(0, len(outputs))
                    ]
                )

            # calculation of the scores
            scores = {}
            metric_calc = Metrics()
            # load all results
            metric_calc.add(
                prediction=[x["prediction"] for x in task_results],
                reference=[x["example"][task["doc_to_target"]] for x in task_results],
            )
            # calculate the scores for the task and each metric
            for metric in task["metric_list"]:
                metric_calc.metric_type(metric)
                scores[metric] = metric_calc.compute()

            # save results
            if self.log_samples:
                results[task["task_name"]] = {
                    "scores": scores,
                    "results": task_results,
                }
            else:
                results[task["task_name"]] = {
                    "scores": scores,
                }
        return results

    def reset(self):
        # TODO: clean up
        pass
