from typing import List
from jinja2 import Template
from datasets import Dataset, load_dataset, load_from_disk, load_metric
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
    else:
        dataset = load_dataset(path, split=split)
    # breakpoint()
    return dataset


def calc_metric(metric: str, results: List, target):
    m = load_metric(metric)
    for result in results:
        m.add(prediction=result[0], reference=result[1][target])
    return m.compute()


def save_results(output_path, results):
    with open(output_path / "result.json", "w") as f:
        json.dump(
            results,
            f,
        )  # indent=4
