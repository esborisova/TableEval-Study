from typing import List
from datetime import datetime
from jinja2 import Template
from datasets import Dataset, load_dataset, load_from_disk
import os
import json


def prompt_gen(template, samples: List, target: str = "") -> List[str]:
    if isinstance(template, str):
        return text_gen(template, samples, target)
    else:
        return multi_modal_gen(template, samples, target)


def multi_modal_gen(function, samples: List, target: str = "") -> List[[str, str]]:
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


def load_samples(path: str, split: str) -> Dataset:
    if os.path.exists(path):
        dataset = load_from_disk(path)
        dataset = dataset[split]

    else:
        dataset = load_dataset(path, split=f"{split}")
    return dataset


def save_results(output_path, results, model_name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists(f"{dir_path}/{output_path}"):
        os.makedirs(f"{dir_path}/{output_path}")
    # TODO: change genereic output name
    with open(
        f"{dir_path}/{output_path}/{model_name}_{current_datetime}.json", "w"
    ) as f:
        json.dump(
            results,
            f,
        )  # indent=4
