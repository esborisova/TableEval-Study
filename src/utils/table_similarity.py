import ast
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from other import read_json
import numpy as np


def clean_headers(row) -> list:
    return (
        [item for item in ast.literal_eval(row) if item != "[EMPTY]"]
        if pd.notna(row)
        else []
    )


def clean_content(row) -> list:
    return (
        [
            item
            for sublist in ast.literal_eval(row)
            for item in sublist
            if item != "[EMPTY]"
        ]
        if pd.notna(row)
        else []
    )


def join_table_content(row) -> str:
    header_str = " ".join(row["clean_headers"]) if row["clean_headers"] else ""
    content_str = " ".join(row["clean_rows"]) if row["clean_rows"] else ""
    subheader_str = " ".join(row["clean_subheaders"]) if row["clean_subheaders"] else ""

    table_title = row["table_title"] if pd.notna(row["table_title"]) else ""
    table_caption = row["table_caption"] if pd.notna(row["table_caption"]) else ""
    table_footnote = row["table_footnote"] if pd.notna(row["table_footnote"]) else ""

    return f"{table_title} {table_caption} {header_str} {subheader_str} {content_str} {table_footnote}".lower()


def compute_cosine_similarity(vectorizer, text1: str, text2: str) -> float:
    vectors = vectorizer.transform([text1, text2]).toarray()
    return cosine_similarity(vectors)[0][1]


def get_json_content(jsons_rootdir: str, image_name: str) -> str:
    json_file_path = os.path.join(
        jsons_rootdir, image_name.replace(".jpg", "_words.json")
    )
    json_data = read_json(json_file_path)
    return " ".join([entry["text"].lower() for entry in json_data])


def calculate_similarity_for_row(row, jsons_rootdir: str, vectorizer):
    """Calculate similarity score for a given row in pmc subset with the corresponding JSON content."""
    if row["dataset"] == "PubTab1M":
        json_content = get_json_content(jsons_rootdir, row["image_name"])
        return compute_cosine_similarity(
            vectorizer, row["merged_content"], json_content
        )
    return None


def prepare_tables(
    df: pd.DataFrame,
    header: str,
    subheader: str,
    content: str,
    merged_content: bool = False,
) -> pd.DataFrame:
    df["clean_headers"] = df[header].apply(clean_headers)
    df["clean_subheaders"] = df[subheader].apply(clean_content)
    df["clean_rows"] = df[content].apply(clean_content)

    if merged_content is True:
        df["merged_content"] = df.apply(join_table_content, axis=1)
    return df


def find_best_table(candidates: pd.DataFrame, json_content: str, vectorizer) -> tuple:
    """Find the best matching table based on similarity score."""
    highest_similarity_score = 0
    best_table = None

    for _, row in candidates.iterrows():
        candidate_content = join_table_content(row)
        score = compute_cosine_similarity(vectorizer, json_content, candidate_content)

        if score > highest_similarity_score:
            highest_similarity_score = score
            best_table = row

    return best_table, highest_similarity_score


def substitute_table(
    df: pd.DataFrame,
    low_similarity_indices: list,
    pmc_all_tables: pd.DataFrame,
    jsons_rootdir: str,
    vectorizer,
    columns_to_update: list,
) -> pd.DataFrame:
    for indx in low_similarity_indices:
        paper_id = df["id"].iloc[indx]
        json_content = get_json_content(jsons_rootdir, df["image_name"].iloc[indx])

        candidate_tables = pmc_all_tables[pmc_all_tables["id"] == paper_id]
        best_table, highest_similarity_score = find_best_table(
            candidate_tables, json_content, vectorizer
        )

        if best_table is not None:
            for col in columns_to_update:
                if "similarity" in col:
                    df.at[indx, col] = round(highest_similarity_score, 2)
                else:
                    df.at[indx, col] = best_table[col]
    return df


def preprocess_table(df):
    return " ".join(df.astype(str).fillna("").values.flatten())


def compute_tables_similarity(gold_table, extracted_tables):
    gold_string = preprocess_table(gold_table)
    extracted_strings = [preprocess_table(table) for table in extracted_tables]

    vectorizer = CountVectorizer().fit([gold_string] + extracted_strings)
    vectors = vectorizer.transform([gold_string] + extracted_strings)

    similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    best_match_index = np.argmax(similarities)
    return extracted_tables[best_match_index], similarities[best_match_index]
