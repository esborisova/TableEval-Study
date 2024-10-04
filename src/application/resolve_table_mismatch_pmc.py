import json
import pandas as pd
import ast
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_json(file_path: str) -> list:
    with open(file_path) as f:
        return json.load(f)


def join_table_content(row) -> str:
    header_str = " ".join(row["clean_headers"]) if row["clean_headers"] else ""
    content_str = " ".join(row["clean_content"]) if row["clean_content"] else ""

    table_title = row["table_title"] if pd.notna(row["table_title"]) else ""
    table_caption = row["table_caption"] if pd.notna(row["table_caption"]) else ""
    table_footnote = row["table_footnote"] if pd.notna(row["table_footnote"]) else ""

    return f"{table_title} {table_caption} {header_str} {content_str} {table_footnote}"


def compute_cosine_similarity(vectorizer, text1: str, text2: str) -> float:
    vectors = vectorizer.transform([text1, text2]).toarray()
    return cosine_similarity(vectors)[0][1]


def calculate_similarity_for_row(row, jsons_rootdir: str, vectorizer):
    """Calculate similarity score for a given row with the corresponding JSON content."""
    if row["dataset"] == "PubTab1M":
        json_file_path = os.path.join(
            jsons_rootdir, row["image_name"].replace(".jpg", "_words.json")
        )
        json_data = read_json(json_file_path)
        json_content = " ".join([entry["text"].lower() for entry in json_data])

        return compute_cosine_similarity(
            vectorizer, row["merged_content"], json_content
        )
    return None


def main():
    comtqa_with_tables = pd.read_csv("../comtqa_df.csv")
    jsons_rootdir = "/Users/ekbo01/Downloads/PubTables-1M-Structure_Table_Words/"

    vectorizer = CountVectorizer()

    comtqa_with_tables["clean_headers"] = comtqa_with_tables["table_header"].apply(
        lambda x: [item for item in ast.literal_eval(x) if item != "[EMPTY]"]
        if pd.notna(x)
        else []
    )

    comtqa_with_tables["clean_content"] = comtqa_with_tables["table_content"].apply(
        lambda x: [
            item
            for sublist in ast.literal_eval(x)
            for item in sublist
            if item != "[EMPTY]"
        ]
        if pd.notna(x)
        else []
    )

    comtqa_with_tables["merged_content"] = comtqa_with_tables.apply(
        lambda row: join_table_content(row) if row["dataset"] == "PubTab1M" else "",
        axis=1,
    )

    comtqa_with_tables["merged_content"] = comtqa_with_tables[
        "merged_content"
    ].str.lower()

    merged_content_list = comtqa_with_tables[
        comtqa_with_tables["dataset"] == "PubTab1M"
    ]["merged_content"].tolist()
    vectorizer.fit(merged_content_list)

    comtqa_with_tables["similarity_score"] = comtqa_with_tables.apply(
        lambda row: calculate_similarity_for_row(row, jsons_rootdir, vectorizer), axis=1
    )

    comtqa_with_tables["similarity_score"] = pd.to_numeric(
        comtqa_with_tables["similarity_score"], errors="coerce"
    )
    comtqa_with_tables["similarity_score"] = comtqa_with_tables[
        "similarity_score"
    ].round(2)
    pubmed_subset = comtqa_with_tables[comtqa_with_tables["dataset"] == "PubTab1M"]
    low_similarity_indices = pubmed_subset[
        pubmed_subset["similarity_score"] < 0.90
    ].index
    low_similarity_indices_list = low_similarity_indices.tolist()


if __name__ == "__main__":
    main()
