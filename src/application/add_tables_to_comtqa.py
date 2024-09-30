from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd
import re


def main():
    comtqa = load_dataset("ByteDance/ComTQA")
    tables_df = pd.read_csv("../../data/pubmed/pubmed_tables.csv")

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
    merged_df.to_csv("../../data/pubmed/comtqa_df.csv", index=False)

    df_reset = merged_df.reset_index(drop=True)
    hf_dataset = Dataset.from_pandas(df_reset)
    hf_dataset_dict = DatasetDict({"train": hf_dataset})
    hf_dataset_dict.save_to_disk("../../data/comtqa")


if __name__ == "__main__":
    main()
