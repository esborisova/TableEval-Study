from typing import List, Dict, Union
from evaluate import load
import re

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
        if ("rouge" or "bleu") in self.metric_name:
            return self.function(self.predictions, self.references, self.metric_name)
        if "perplexity" in self.metric_name:
            return self.function(self.predictions, self.model_id)
        return self.function(self.predictions, self.references)

    def reset(self):
        self.predictions = []
        self.references = []

    def metric_type(self, new_metric: str):
        self.metric_name = new_metric
        if new_metric:
            if "bleu" in new_metric:
                new_metric = "bleu"
            if "rouge" in new_metric:
                new_metric = "rouge"
            self.function = METRIC_REGISTRY[new_metric]
        else:
            self.function = None


@register("meteor")
def meteor(predictions, references):
    """Return the mean of the meteor_score for each prediction and reference pair."""
    import nltk
    m_score = []
    nltk.download("wordnet")
    for prediction, reference in zip(predictions, references):
        score = nltk.translate.meteor_score.single_meteor_score(
            reference.split(), prediction.split()
        )
        m_score.append(score)
    return {"meteor": sum(m_score) / len(m_score)}


@register("moverS")
def moverS(predictions, references):
    """Use the original version with BERTMNLI to reproduce the results.
    from moverscore import get_idf_dict, word_mover_score
    Recommend to use this version (DistilBERT) for evaluation, if the speed is your concern.
    """
    from moverscore_v2 import get_idf_dict, word_mover_score

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
    from bleurt import score
    
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

    # Iterate through predictions and references
    result_intersection = references.intersection(predictions)
    intersec = len(result_intersection)

    # Calculate Precision and Recall
    precision = intersec / len(references)
    recall = intersec / len(predictions)

    # Calculate F1 Score
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return {"f1": f1_score, "precision": precision, "recall": recall}


@register("perplexity")
def perplexity(predictions, model_id):
    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(predictions=predictions, model_id=model_id)
    return results


@register("rouge")
def rouge(predictions, references, r_type: str = ""):
    """there are multiple rouge scores (rougeS, rougeL, rougeN, rouge1, rouge2,
    rouge3, rouge4) you can evaluate on any if you write the type in the yaml
    file. More information read on https://thepythoncode.com/article/calculate-rouge-score-in-python
    """
    from rouge_score import rouge_scorer
    
    precision = []
    recall = []
    f1 = []
    scorer = rouge_scorer.RougeScorer([r_type], use_stemmer=True)
    for ref, pred in zip(references, predictions):
        score = scorer.score(ref, pred)
        precision.append(score[r_type].precision)
        recall.append(score[r_type].recall)
        f1.append(score[r_type].fmeasure)
    return {
        "f1": sum(f1) / len(f1),
        "precision": sum(precision) / len(precision),
        "recall": sum(recall) / len(recall),
    }


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
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    if re.match(".\\d+", b_type):
        weights = [0, 0, 0, 0, 0]
        index = int(b_type[-1])
        weights[index] = 1
        weights = tuple(weights)

    else:
        # default bleu is 4-gram
        weights = (0, 0, 0, 1)
        
    smoothing_function = SmoothingFunction().method1
    
    bleu_score = []
    for pred, ref in zip(predictions, references):
        bleu_score.append(sentence_bleu(ref.split(), pred.split(), weights=weights, smoothing_function=smoothing_function))
    return sum(bleu_score) / len(bleu_score)
