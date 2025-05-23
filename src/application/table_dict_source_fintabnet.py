"""Script for collecting table headers and rows in fintabnet subset 
based on html generated from source annotations."""
import numpy as np
import pandas as pd
from datasets import load_from_disk
from ..utils.other import create_and_save_dataset


def main():
    ds = load_from_disk(f"../../data/ComTQA_data/comtqa_updated_2024-11-26")
    comtqa_df = ds["train"].to_pandas()
    fintab_subset = comtqa_df[comtqa_df["dataset"] == "FinTabNet"]
    fintab_df = fintab_subset["train"].to_pandas()

    parsed_tables = []

    for _, row in fintab_df.iterrows():
        html_df = pd.read_html(row["table_html"], header=[0])
        html_df = html_df[0]
        html_df.columns = [
            col if not col.startswith("Unnamed:") else "" for col in html_df.columns
        ]
        headers = html_df.columns.values
        rows = html_df.values
        rows = [
            [
                str(item)
                if not isinstance(item, str)
                else item
                if not isinstance(item, float) or not np.isnan(item)
                else ""
                for item in sublist
            ]
            for sublist in rows
        ]
        parsed_tables.append(
            {
                "image_name": row["image_name"],
                "table_headers": headers,
                "table_rows": rows,
                "table_id": row["table_id"],
                "question": row["question"],
                "answer": row["answer"],
            }
        )
        parsed_tables_df = pd.DataFrame(parsed_tables)
        merged_df = pd.merge(
            comtqa_df,
            parsed_tables_df,
            on=["image_name", "table_id", "question", "answer"],
            how="left",
            suffixes=("_df2", "_df1"),
        )
        merged_df["table_headers"] = merged_df["table_headers_df1"].combine_first(
            merged_df["table_headers_df2"]
        )
        merged_df["table_rows"] = merged_df["table_rows_df1"].combine_first(
            merged_df["table_rows_df2"]
        )
        merged_df = merged_df.drop(
            columns=[
                "table_headers_df1",
                "table_headers_df2",
                "table_rows_df2",
                "table_rows_df1",
            ]
        )
        create_and_save_dataset(merged_df, "train", "../../data/ComTQA_data/comtqa_updated")

if __name__ == "__main__":
    main()
