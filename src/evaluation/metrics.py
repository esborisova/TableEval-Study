import math
import random
import re
import string
from collections.abc import Iterable
from typing import List, Dict, Union

import numpy as np
import sacrebleu


METRIC_REGISTRY = {}


def register(name: str):
    def decorate(fn):
        assert (
            name not in METRIC_REGISTRY
        ), f"'{name}' conflicts with existing name in register!"

        METRIC_REGISTRY[name] = fn
        return fn

    return decorate


class Metrics:
    def __init__(self, metric_name: str = "") -> None:
        self.predictions: List[str] = []
        self.references: List[str] = []
        if metric_name:
            self.function = METRIC_REGISTRY[metric_name]
        else:
            self.function = None

    def add(self, prediction: Union[List, str], reference: Union[List, str]):
        if type(reference) is not type(prediction):
            print(
                f"WARNING: The type of reference ({type(reference)}) is different from the type of the prediction ({type(prediction)})"
            )
        if isinstance(reference, str):
            self.references.append(reference)
        else:
            if len(reference) != len(prediction):
                print(
                    f"WARNING: references {len(reference)} and predictions ({len(prediction)}) do not have the same length"
                )
            self.references.extend(reference)
        if isinstance(prediction, str):
            self.predictions.append(prediction)
        else:
            self.predictions.extend(prediction)

    def compute(self) -> Dict:
        return self.function(self.predictions, self.references)

    def reset(self):
        self.predictions = []
        self.references = []

    def metric_type(self, new_metric: str):
        self.function = METRIC_REGISTRY[new_metric]


@register("bleu")
def bleu(predictions, references):
    """The Bilingual Evaluation Understudy Score, or BLEU for short, is a metric
    for evaluating a generated sentence to a reference sentence. It counts matching
    n-grams in the candidate translation to n-grams in the reference text, where
    1-gram or unigram would be each token and a bigram comparison would be each
    word pair. The comparison is made regardless of word order
    Source: https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
    Paper: https://www.aclweb.org/anthology/P02-1040/

    Higher is better
    """
    return sacrebleu.corpus_bleu(predictions, references).score


# TODO: METEOR

# TODO: MoverS

# TODO: BLEURT

# TODO: BertS

# TODO: Accuracy

# TODO: Perplexity

# TODO: F1

# TODO: BLEU-1, BLEU-2, BLEU-3, BLEU-45, ROUGE-, ROUGE-4L (F measure), R-1, R-2, R-4, and R-L
