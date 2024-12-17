from typing import List, Dict, Union

import sacrebleu
import nltk
from bleurt import score
from evaluate import load
from moverscore_v2 import get_idf_dict, word_mover_score

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
    def __init__(self, metric_name: str = "", model_id: str = "") -> None:
        self.predictions: List[str] = []
        self.references: List[str] = []
        self.model_id = model_id
        self.metric_name = metric_name

        if metric_name:
            if "bleu" in metric_name:
                metric_name = "bleu"
            if "rouge" in metric_name:
                metric_name = "rouge"
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
        if "rouge" or "bleu" in self.metric_name:
            return self.function(self.predictions, self.references, self.metric_name)
        if "perplexity" in self.metric_name:
            return self.function(self.predictions, self.model_id)
        return self.function(self.predictions, self.references)

    def reset(self):
        self.predictions = []
        self.references = []

    def metric_type(self, new_metric: str):
        self.function = METRIC_REGISTRY[new_metric]


@register("meteor")
def meteor(predictions, references):
    """Return the mean of the meteor_score for each prediction and reference pair."""
    m_score = []
    for prediction, reference in zip(predictions, references):
        score = nltk.translate.meteor_score.meteor_score(reference, prediction)
        m_score.append(score)
    return m_score


@register("moverS")
def moverS(predictions, references):
    """Use the original version with BERTMNLI to reproduce the results.
    from moverscore import get_idf_dict, word_mover_score
    Recommend to use this version (DistilBERT) for evaluation, if the speed is your concern.
    """

    idf_dict_hyp = get_idf_dict(predictions)  # idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = get_idf_dict(references)  # idf_dict_ref = defaultdict(lambda: 1.)

    score = word_mover_score(
        references,
        predictions,
        idf_dict_ref,
        idf_dict_hyp,
        stop_words=[],
        n_gram=1,
        remove_subwords=True,
    )
    return score


@register("bleurt")
def bleurt(predictions, references):
    checkpoint = "bleurt/test_checkpoint"

    scorer = score.BleurtScorer(checkpoint)
    scores = scorer.score(references=references, candidates=predictions)
    assert isinstance(scores, list) and len(scores) == 1
    return scores


@register("bertS")
def bertS(predictions, references):
    bertscore = load("bertscore")
    results = bertscore.compute(
        predictions=predictions, references=references, lang="en"
    )
    return results


@register("accuracy")
def accuracy(predictions, references):
    if len(predictions) != len(references):
        raise ValueError("The length of predictions and references must be the same.")

    # Count the number of correct predictions
    correct_count = sum(pred == ref for pred, ref in zip(predictions, references))

    # Calculate accuracy
    accuracy = correct_count / len(references)

    return accuracy


@register("f1")
def f1(predictions, references):
    # Check if both lists have the same length
    if len(predictions) != len(references):
        raise ValueError("The length of predictions and references must be the same.")

    # Initialize counts
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives

    # Iterate through predictions and references
    for pred, ref in zip(predictions, references):
        if pred == 1 and ref == 1:
            tp += 1  # True Positive
        elif pred == 1 and ref == 0:
            fp += 1  # False Positive
        elif pred == 0 and ref == 1:
            fn += 1  # False Negative

    # Calculate Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Calculate F1 Score
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score


@register("perplexity")
def perplexity(predictions, model_id):
    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(predictions=predictions, model_id=model_id)
    return results


@register("rouge")
def rouge(predictions, references, r_type: str = ""):
    pass


@register("bleu")
def bleu(predictions, references, b_type: str = ""):
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


# TODO: BLEU-1, BLEU-2, BLEU-3, BLEU-45, ROUGE-, ROUGE-4L (F measure), R-1, R-2, R-4, and R-L
