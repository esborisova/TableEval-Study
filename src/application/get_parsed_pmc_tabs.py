from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from datetime import datetime
from ..utils.parse_xml_table import parsed_tab_xml_df


def main():
    ds = load_from_disk("../../../data/ComTQA_data/comtqa_updated_2024-11-20")
    comtqa_df = ds["train"].to_pandas()
    comtqa_df[["table_headers", "table_subheaders", "table_rows"]] = comtqa_df.apply(
        parsed_tab_xml_df,
        axis=1,
        condition_col="dataset",
        condition_value="PubTab1M",
        xml_col="table_xml",
    )
    dataset = Dataset.from_pandas(comtqa_df)
    dataset_dict = DatasetDict({"train": dataset})
    date = datetime.now().strftime("%Y-%m-%d")
    dataset_dict.save_to_disk(f"../../../data/ComTQA_data/comtqa_updated_{date}")


if __name__ == "__main__":
    main()
