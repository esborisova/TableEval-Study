"""Script for computing BLEU, METEOR, and Rouge scores between gold latex tables in SciGen and Gemini generated."""
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction
from rouge_score import rouge_scorer
from ..utils.other import read_json
from ..utils.text_gen_metrics import (
    tokenize_latex,
    calculate_bleu_scores,
    compute_rouge_scores,
    create_rouge_df,
    calculate_meteor_scores,
)


def main():
    file_path = "../../data/SciGen/scigen_latex_gold_data.json"
    data = read_json(file_path)
    smoothing = SmoothingFunction().method1

    df = pd.DataFrame.from_dict(data, orient="index").reset_index()
    df.rename(columns={"index": "instance_key"}, inplace=True)
    df = df.dropna(
        subset=[
            "gemini_latex_from_img",
            "gemini_latex_from_dict",
            "gemini_latex_from_both",
        ]
    )
    df = df.reset_index(drop=True)

    image_ids = df["image_id"].tolist()
    latex_gold = df["table_latex"].tolist()
    df_columns = [
        "gemini_latex_from_img",
        "gemini_latex_from_dict",
        "gemini_latex_from_both",
    ]

    tokenized_results = {}
    for column in df_columns:
        tokenized_results[column] = tokenize_latex(df[column].tolist())

    tokenized_gold = tokenize_latex(latex_gold)

    bleu_scores = {
        column: calculate_bleu_scores(
            tokenized_gold, tokenized_results[column], smoothing
        )
        for column in df_columns
    }

    bleu_scores_df = pd.DataFrame(
        {
            "image_id": image_ids,
            "text_bleu_scores": bleu_scores["gemini_latex_from_dict"],
            "image_bleu_scores": bleu_scores["gemini_latex_from_img"],
            "image_text_bleu_scores": bleu_scores["gemini_latex_from_both"],
        }
    )

    bleu_average = bleu_scores_df[
        ["text_bleu_scores", "image_bleu_scores", "image_text_bleu_scores"]
    ].mean()
    print("Average blue scores:", bleu_average)
    bleu_scores_df.to_csv("../../data/formats_generation/latex_bleu_scores.csv")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    image_based = df["gemini_latex_from_img"].tolist()
    text_based = df["gemini_latex_from_dict"].tolist()
    image_text_based = df["gemini_latex_from_both"].tolist()

    image_based_scores = compute_rouge_scores(latex_gold, image_based, scorer)
    text_based_scores = compute_rouge_scores(latex_gold, text_based, scorer)
    image_text_based_scores = compute_rouge_scores(latex_gold, image_text_based, scorer)

    df_image_based = create_rouge_df(image_ids, image_based_scores, "img_based")
    df_text_based = create_rouge_df(image_ids, text_based_scores, "text_based")
    df_image_text_based = create_rouge_df(
        image_ids, image_text_based_scores, "img_text_based"
    )

    df_text_based = df_text_based.drop(columns=["image_id"])
    df_image_text_based = df_image_text_based.drop(columns=["image_id"])

    rouge_scores_df = pd.concat(
        [df_image_based, df_text_based, df_image_text_based], axis=1
    )
    rouge_average = rouge_scores_df[
        [
            "img_based_rouge1",
            "img_based_rouge2",
            "img_based_rougeL",
            "text_based_rouge1",
            "text_based_rouge2",
            "text_based_rougeL",
            "img_text_based_rouge1",
            "img_text_based_rouge2",
            "img_text_based_rougeL",
        ]
    ].mean()
    print("Average rouge scores:", rouge_average)
    rouge_scores_df.to_csv("../../data/formats_generation/latex_rouge_scores.csv")

    meteor_scores = {
        column: calculate_meteor_scores(tokenized_gold, tokenized_results[column])
        for column in df_columns
    }

    meteor_scores_df = pd.DataFrame(
        {
            "image_id": image_ids,
            "text_meteor_scores": meteor_scores["gemini_latex_from_dict"],
            "image_meteor_scores": meteor_scores["gemini_latex_from_img"],
            "image_text_meteor_scores": meteor_scores["gemini_latex_from_both"],
        }
    )

    meteor_average = meteor_scores_df[
        ["text_meteor_scores", "image_meteor_scores", "image_text_meteor_scores"]
    ].mean()
    print("Average meteor scores:", meteor_average)
    meteor_scores_df.to_csv("../../data/formats_generation/latex_meteor_scores.csv")


if __name__ == "__main__":
    main()
