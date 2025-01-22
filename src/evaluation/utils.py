from typing import List
from datetime import datetime
from jinja2 import Template
from datasets import Dataset, load_dataset, load_from_disk
import os
import json
import random


def load_samples(path: str, split: str) -> Dataset:
    """Load the dataset from a HF source. either from a local source or from
    the Hub."""
    if os.path.exists(path):
        dataset = load_from_disk(path)
        dataset = dataset[split]

    else:
        dataset = load_dataset(path, split=f"{split}")
    return dataset


def save_results(output_path, results, model_name, log_logits: bool = False):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_generator, model_id = model_name.split("/")
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

    if not os.path.exists(f"{dir_path}/{output_path}"):
        os.makedirs(f"{dir_path}/{output_path}")

    if not os.path.exists(f"{dir_path}/{output_path}/{model_generator}"):
        os.makedirs(f"{dir_path}/{output_path}/{model_generator}")

    for task_name, result in results.items():
        with open(
            f"{dir_path}/{output_path}/{model_generator}/scores_{task_name}_{model_id}_{current_datetime}.json",
            "w+",
        ) as f:
            json.dump(
                result["scores"],
                f,
            )  # indent=4
        if "results" in result.keys():
            if log_logits:
                with open(
                    f"{dir_path}/{output_path}/{model_generator}/logits_{task_name}_{model_id}_{current_datetime}.json",
                    "w+",
                ) as f:
                    json.dump(
                        [x["logits"] for x in result["results"]],
                        f,
                    )  # indent=4

            with open(
                f"{dir_path}/{output_path}/{model_generator}/results_{task_name}_{model_id}_{current_datetime}.json",
                "w+",
            ) as f:
                json.dump(
                    [
                        {
                            "prediction": x["prediction"],
                            "reference": x["reference"],
                            "input": x["input"],
                            "example": x["example"],
                        }
                        for x in result["results"]
                    ],
                    f,
                )  # indent=4


def generate_prompt(
    samples, few_shot_samples, num_fewshot, task, prompt_template: bool = False
):
    """There are two options to generate the prompt. Either as a single string
    or using the chat template structure of a list with meta data. For more
    information please check https://huggingface.co/docs/transformers/chat_templating"""
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


def generate_template_prompt(samples, few_shot_samples, num_fewshot, task):
    """generate the prompts from the template of the yampl file but not in the
    template structure but as a single input string"""
    pass


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
    if isinstance(task["doc_to_text"], str):
        # for text parsing
        return text_to_prompt(
            samples_with_input_text, text_samples, few_shot_prompt, num_fewshot
        )
    else:
        # for image parsing
        textual_prompts = text_to_prompt(
            [s[1] for s in samples_with_input_text],
            text_samples,
            few_shot_prompt,
            num_fewshot,
        )
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
        return multi_modal_gen(template, samples, target)


def multi_modal_gen(function, samples: List, target: str = "") -> List[List[str]]:
    # The return value should be a list of a list of an image and a prompt
    return function(samples)


def text_gen(template, samples: List, target: str = "") -> List[str]:
    if target:
        t = Template(template + "{{%s}}" % target)
    else:
        t = Template(template)
    prompts = []
    for sample in samples:
        prompts.append(t.render(sample))
    return prompts
