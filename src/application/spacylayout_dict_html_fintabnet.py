"""Pipeline for matching the tables exctracted using spaCy layout from fintabnet pdfs with the gold tables and 
obtaing headers, rows, and html."""

import pandas as pd
import numpy as np
import pickle
from datasets import load_from_disk
from ..utils.other import create_and_save_dataset
from ..utils.table_similarity import compute_tables_similarity
from ..utils.html_latex_convertion import fix_auto_generated_headers
from ..utils.xml_html_convertion import change_table_class, prettify_html, validate_html


def main():
    ds = load_from_disk(f"../../data/ComTQA_data/comtqa_updated_2024-11-27")
    fintab_subset = ds.filter(lambda x: x["dataset"] == "FinTabNet")
    fintab_df = fintab_subset["train"].to_pandas()

    with open("../../data/ComTQA_data/other/spacylayout_output.pkl", "rb") as f:
        data = pickle.load(f)

    data["matched_table_spacylayout"] = data["matched_table_spacylayout"].apply(
        lambda x: fix_auto_generated_headers(x) if x is not None else x
    )

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
                        (
                            str(item)
                            if not isinstance(item, str)
                            else (
                                item
                                if not isinstance(item, float) or not np.isnan(item)
                                else ""
                            )
                        )
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

    # assign none to tables which either incorrectly extracted or no cell values were extracted
    tables_to_exclude = [
        "TMO_2011_page_104_38704.png",
        "TMO_2011_page_104_38705.png",
        "ZBRA_2004_page_63_456.png",
    ]
    columns_to_update = [
        "table_headers_spacylayout",
        "table_rows_spacylayout",
        "table_html_spacylayout",
    ]
    merged_df.loc[
        merged_df["image_name"].isin(tables_to_exclude), columns_to_update
    ] = None

    # remove format indicator
    merged_df["table_html_spacylayout"] = merged_df["table_html_spacylayout"].apply(
        lambda x: change_table_class(x) if x is not None else x
    )

    # prettify html
    merged_df["table_html_spacylayout"] = merged_df["table_html_spacylayout"].apply(
        lambda x: prettify_html(x) if x is not None else x
    )

    # validate html
    validated_html = validate_html(
        merged_df, "table_html_spacylayout", "table_id", "fin_val_html_spacylayout"
    )

    create_and_save_dataset(merged_df, "train", "../../data/ComTQA_data/comtqa_updated")


if __name__ == "__main__":
    main()
