from typing import List
from jinja2 import Template
from datasets import Dataset, load_dataset, load_from_disk
import os
import json


def prompt_gen(template: str, samples: List, target: str = "") -> List[str]:
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


def save_results(output_path, results):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(f"{dir_path}/{output_path}"):
            os.makedirs(f"{dir_path}/{output_path}")
    # TODO: change genereic output name
    with open( f"{dir_path}/{output_path}/result.json", "w") as f:
        json.dump(
            results,
            f,
        )  # indent=4
