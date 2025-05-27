"""Pipeline for adding PMC tables to ComTQA dataset based on tables titles."""
from datasets import load_dataset
import pandas as pd
import re
from datetime import datetime
from ..utils.other import create_and_save_dataset


def main():
    comtqa = load_dataset("ByteDance/ComTQA")
    tables_df = pd.read_csv(
        "../../data/ComTQA_data/pubmed/utils/pubmed_tables_updated_2014-12-01.csv"
    )

    tables_df["cleaned_table_title"] = tables_df["table_title"].apply(
        lambda x: x.lower().replace(" ", "")
    )

    comtqa = comtqa["train"].to_pandas()
    comtqa["cleaned_table_title"] = ""
    comtqa["id"] = ""

    for index, _ in comtqa.iterrows():
        if comtqa["dataset"][index] == "PubTab1M":
            id = comtqa["image_name"][index].split("_")[0]
            comtqa["id"][index] = id
            match = re.search(r"table_(\d+)", comtqa["image_name"][index])
            if match:
                cleaned_table_title = re.sub(
                    r"(\d+)",
                    lambda num: str(int(num.group(0)) + 1),
                    match.group(0).replace("_", ""),
                )
                comtqa["cleaned_table_title"][index] = cleaned_table_title
        else:
            comtqa["cleaned_table_title"][index] = None
            comtqa["id"][index] = None

    merged_df = pd.merge(comtqa, tables_df, how="left")
    date = datetime.now().strftime("%Y-%m-%d")
    merged_df.to_csv(
        f"../../data/ComTQA_data/pubmed/utils/comtqa_df_updated_{date}.csv", index=False
    )
    create_and_save_dataset(merged_df, "train", "../../data/ComTQA_data/comtqa_updated")


if __name__ == "__main__":
    main()
