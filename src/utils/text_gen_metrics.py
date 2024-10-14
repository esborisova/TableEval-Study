import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import pandas as pd

nltk.download("wordnet")
nltk.download("omw-1.4")
from typing import List


def tokenize_latex(latex_list: List[str]) -> List[str]:
    return [latex.replace("\n", " ").split() for latex in latex_list]


def calculate_bleu_scores(
    tokenized_gold: List[List[str]], tokenized_results: List[List[str]], smoothing=None
) -> List[float]:
    bleu_scores = []
    for gold, generated in zip(tokenized_gold, tokenized_results):
        score = (
            sentence_bleu([gold], generated, smoothing_function=smoothing)
            if smoothing
            else sentence_bleu([gold], generated)
        )
        bleu_scores.append(score)
    return bleu_scores


def calculate_meteor_scores(
    tokenized_gold: List[List[str]], tokenized_results: List[List[str]]
) -> List[float]:
    meteor_scores = []
    for gold, generated in zip(tokenized_gold, tokenized_results):
        score = meteor_score([gold], generated)
        meteor_scores.append(score)
    return meteor_scores


def compute_rouge_scores(
    reference_list: List[str], hypothesis_list: List[str], scorer
) -> List[float]:
    scores_list = []
    for reference, hypothesis in zip(reference_list, hypothesis_list):
        scores = scorer.score(reference, hypothesis)
        scores_list.append(
            {
                "rouge1": scores["rouge1"].fmeasure,
                "rouge2": scores["rouge2"].fmeasure,
                "rougeL": scores["rougeL"].fmeasure,
            }
        )
    return scores_list


def create_rouge_df(
    image_ids: List[str], scores_list: List[float], prefix: str
) -> pd.DataFrame:
    df = pd.DataFrame(scores_list)
    return pd.DataFrame(
        {
            "image_id": image_ids,
            f"{prefix}_rouge1": df["rouge1"],
            f"{prefix}_rouge2": df["rouge2"],
            f"{prefix}_rougeL": df["rougeL"],
        }
    )


def compute_average_score(scores_df: pd.DataFrame):
    average = scores_df.loc[scores_df["average"].idxmax()]
    return average
