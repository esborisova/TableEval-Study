from typing import List, Optional
from datasets.packaged_modules import text
from jinja2 import Template
from datasets import Dataset, load_dataset, load_from_disk
import os
import json
import random
import copy


def load_samples(path: str, split: str) -> Dataset:
    """Load the dataset from a HF source. either from a local source or from
    the Hub."""
    if os.path.exists(path):
        dataset = load_from_disk(path)
        dataset = dataset[split]

    else:
        dataset = load_dataset(path, split=f"{split}")
    return dataset


def generate_output_folder(
    output_path, model_name, task_name, current_datetime, log_logits: bool = False
):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_generator, model_id = model_name.split("/")

    if not os.path.exists(f"{dir_path}/{output_path}"):
        os.makedirs(f"{dir_path}/{output_path}")

    if not os.path.exists(f"{dir_path}/{output_path}/{model_generator}"):
        os.makedirs(f"{dir_path}/{output_path}/{model_generator}")

    results_name = f"{dir_path}/{output_path}/{model_generator}/results_{task_name}_{model_id}_{current_datetime}.json"
    scores_name = f"{dir_path}/{output_path}/{model_generator}/scores_{task_name}_{model_id}_{current_datetime}.json"
    if log_logits:
        logits_name = f"{dir_path}/{output_path}/{model_generator}/logits_{task_name}_{model_id}_{current_datetime}.json"
    else:
        logits_name = None

    return scores_name, results_name, logits_name


def dump_files(output_path, result, space):
    if space == "scores":
        with open(
            output_path,
            "a+",
        ) as f:
            json.dump(result[space], f, indent=4)
    elif space == "logits":
        with open(
            output_path,
            "a+",
        ) as f:
            json.dump([x[space] for x in result], f, indent=4)
    elif space == "results":
        with open(
            output_path,
            "a+",
        ) as f:
            json.dump(
                [
                    {
                        "prediction": x["prediction"],
                        "reference": x["reference"],
                        "input": x["input"],
                        "example": x["example"],
                    }
                    for x in result
                ],
                f,
                indent=4,
            )


def save_results(results, scores_path, logits_path):

    for result in results.values():
        dump_files(scores_path, result, "scores")
        if "results" in result.keys():
            if logits_path:
                dump_files(scores_path, result, "logits")

            dump_files(scores_path, result, "results")


def generate_prompt(
    samples, few_shot_samples, num_fewshot, task, prompt_template: bool = False
):
    """There are two options to generate the prompt. Either as a single string
    or using the chat template structure of a list with meta data. For more
    information please check https://huggingface.co/docs/transformers/chat_templating"""
    if task.get("ignore_columns"):
        samples = drop_nones(samples, task["ignore_columns"].split(","))

    if prompt_template:
        return generate_template_prompt(
            samples,
            few_shot_samples,
            num_fewshot,
            task,
        )
    else:
        return generate_string_prompt(
            samples,
            few_shot_samples,
            num_fewshot,
            task,
        )


def drop_nones(samples, column_names):
    return [s for s in samples if all(s.get(c) is not None for c in column_names)]


def generate_template_prompt(samples, few_shot_samples, num_fewshot, task):
    """generate the prompts from the template of the yampl file but not in the
    template structure but as a single input string"""

    init_message = []
    if "instruction" in task:
        if task["instruction"]:
            init_message.append({"role": "system", "content": task["instruction"]})

    if num_fewshot != 0:
        if isinstance(few_shot_samples, list):
            few_shot_examples = random.sample(few_shot_samples, num_fewshot)
        else:
            few_shot_examples = few_shot_samples.shuffle().select(range(num_fewshot))

        few_shot_text_samples = sample_to_text(
            task["doc_to_text"],
            few_shot_examples,
        )

        for few_shot_example, sample in zip(few_shot_text_samples, few_shot_examples):
            if not task.get("multi_modal_data"):
                init_message = text_to_template(
                    init_message, few_shot_example, sample, task["doc_to_target"]
                )
            else:
                init_message = mm_to_template(
                    init_message, few_shot_example, sample, task["doc_to_target"]
                )
    samples_with_input_text = sample_to_text(task["doc_to_text"], samples)
    outputs = []

    for input in samples_with_input_text:
        message = copy.deepcopy(init_message)
        if not task.get("multi_modal_data"):
            outputs.append(text_to_template(message, input))
        else:
            outputs.append([input[0], mm_to_template(message, input[1])])
    return outputs


def mm_to_template(
    template: list, text_sample: str, sample=None, doc_to_target: str = ""
):
    template.append(
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": text_sample}],
        }
    )
    if doc_to_target:
        template.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample[doc_to_target]}],
            }
        )
    return template


def text_to_template(
    template: list, text_sample: str, sample=None, doc_to_target: str = ""
):
    template.append({"role": "user", "content": text_sample})
    if doc_to_target:
        template.append({"role": "assistant", "content": sample[doc_to_target]})
    return template


def generate_string_prompt(samples, few_shot_samples, num_fewshot, task):
    """generate the prompts from the template of the yampl file but not in the
    template structure but as a single input string"""
    samples_with_input_text = sample_to_text(task["doc_to_text"], samples)

    if num_fewshot != 0:
        text_samples = sample_to_text(
            task["doc_to_text"], few_shot_samples, task["doc_to_target"]
        )
    else:
        text_samples = []

    if "instruction" in task:
        few_shot_prompt = task["instruction"]
    else:
        few_shot_prompt = ""
    if not task.get("multi_modal_data"):
        # for text parsing
        return text_to_prompt(
            samples_with_input_text, text_samples, few_shot_prompt, num_fewshot
        )
    else:
        # for image parsing split images samples from text samples
        textual_prompts = text_to_prompt(
            [s[1] for s in samples_with_input_text],
            text_samples,
            few_shot_prompt,
            num_fewshot,
        )
        # join image and text samples
        for sample, full_prompt in zip(samples_with_input_text, textual_prompts):
            sample[1] = full_prompt
        return samples_with_input_text


def text_to_prompt(sample_to_prompt, few_shot_to_prompt, few_shot_prompt, num_fewshot):
    # sample the few shot examples
    if num_fewshot != 0:
        few_shot_examples = random.sample(few_shot_to_prompt, num_fewshot)
        # generate the few shot example and instruction prompt
        few_shot_prompt += "\n".join(few_shot_examples) + "\n"
    else:
        few_shot_prompt += ""
    # return list of prompts
    return [few_shot_prompt + i for i in sample_to_prompt]


def sample_to_text(template, samples: List, target: str = "") -> List[str]:
    if isinstance(template, str):
        return text_gen(template, samples, target)
    else:
        return template(samples)


def text_gen(template, samples: List, target: str = "") -> List[str]:
    if target:
        t = Template(template + "{{%s}}" % target)
    else:
        t = Template(template)
    prompts = []
    for sample in samples:
        # TODO: first.
        prompts.append(t.render(sample))
    return prompts
