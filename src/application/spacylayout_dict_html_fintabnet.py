"""Pipeline for matching the tables exctracted using spaCy layout from fintabnet pdfs with the gold tables and 
obtaing headers, rows, and html"""
import pandas as pd
import numpy as np
import pickle
from datasets import load_from_disk
from ..utils.other import create_dataset_object, save_dataset_object
from ..utils.table_similarity import compute_tables_similarity


def main():
    ds = load_from_disk(f"../../data/ComTQA_data/comtqa_updated_2024-11-27")
    fintab_subset = ds.filter(lambda x: x["dataset"] == "FinTabNet")
    fintab_df = fintab_subset["train"].to_pandas()

    with open("../../data/ComTQA_data/other/spacylayout_output.pkl", "rb") as f:
        data = pickle.load(f)

    matched_ids = set()
    matched_tables = []

    for _, row in fintab_df.iterrows():
        for tab_dict in data:
            if (
                row["table_id"] == tab_dict["table_id"]
                and tab_dict["table_id"] not in matched_ids
            ):
                matched_ids.add(tab_dict["table_id"])
                extracted_tables = [
                    value for key, value in tab_dict.items() if "table_df" in key
                ]

                if len(extracted_tables) > 0:
                    gold_table = pd.DataFrame(
                        row["table_rows"].tolist(),
                        columns=row["table_headers"].tolist(),
                    )
                    most_similar_table, similarity_score = compute_tables_similarity(
                        gold_table, extracted_tables
                    )

                    table_headers = most_similar_table.columns.values.tolist()
                    table_headers = [
                        str(item)
                        if not isinstance(item, str)
                        else item
                        if not isinstance(item, float) or not np.isnan(item)
                        else ""
                        for item in table_headers
                    ]

                    table_rows = most_similar_table.values.tolist()
                    table_html = most_similar_table.to_html(index=False)

                else:
                    table_headers = None
                    table_rows = None
                    most_similar_table = None
                    similarity_score = None
                    table_html = None

                table_entry = {
                    "table_id": tab_dict["table_id"],
                    "matched_table_spacylayout": most_similar_table,
                    "table_headers_spacylayout": table_headers,
                    "table_rows_spacylayout": table_rows,
                    "table_html_spacylayout": table_html,
                    "similarity_score_spacylayout": similarity_score,
                }

                matched_tables.append(table_entry)

    matched_tables_df = pd.DataFrame.from_dict(matched_tables)
    matched_tables_df.to_pickle(
        "../../data/ComTQA_data/other/matched_tables_spacylayout.pkl"
    )
    comtqa_df = ds["train"].to_pandas()

    merged_df = pd.merge(
        comtqa_df,
        matched_tables_df[
            [
                "table_id",
                "table_headers_spacylayout",
                "table_rows_spacylayout",
                "table_html_spacylayout",
                "similarity_score_spacylayout",
            ]
        ],
        on="table_id",
        how="left",
    )

    dataset_dict = create_dataset_object(merged_df)
    save_dataset_object(dataset_dict, "../../data/ComTQA_data/comtqa_updated")


if __name__ == "__main__":
    main()
