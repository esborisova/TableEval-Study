"""Pipeline for convering tables HTML in fintabnet subset of comtqa into XML"""
from ..utils.xml_html_convertion import html_to_xml_table
from ..utils.other import create_dataset_object, save_dataset_object


def main():
    comtqa = load_from_disk(f"../../data/ComTQA_data/comtqa_updated_2024-12-03")
    comtqa_df = comtqa["train"].to_pandas()
    fintabnet_df = comtqa_df[comtqa_df["dataset"] == "FinTabNet"]

    fintabnet_html_source = fintabnet_df["table_html"].tolist()
    fintabnet_xml_source = [html_to_xml_table(html) for html in fintabnet_html_source]
    fintabnet_df["table_xml"] = fintabnet_xml_source

    fintabnet_html_spacy = fintabnet_df["table_html_spacylayout"].tolist()
    fintabnet_xml_spacy = [
        html_to_xml_table(html) if html is not None else html
        for html in fintabnet_html_spacy
    ]
    fintabnet_df["table_xml_spacylayout"] = fintabnet_xml_spacy

    merged_df = comtqa_df.merge(
        fintabnet_df[
            [
                "table_id",
                "table_xml",
                "table_xml_spacylayout",
                "image_name",
                "question",
                "answer",
            ]
        ],
        on=["table_id", "image_name", "question", "answer"],
        how="left",
        suffixes=("", "_fintab"),
    )

    merged_df["table_xml"] = merged_df["table_xml"].combine_first(
        merged_df["table_xml_fintab"]
    )
    merged_df = merged_df.drop(columns=["table_xml_fintab"])

    dataset_dict = create_dataset_object(merged_df)
    save_dataset_object(dataset_dict, "../../data/ComTQA_data/comtqa_updated")


if __name__ == "__main__":
    main()
