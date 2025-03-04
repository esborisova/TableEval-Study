"""Pipeline for convering tables HTML in numericNLG and 
fintabnet subset of comtqa into XML"""

import ast
from datasets import load_from_disk
from ..utils.xml_html_convertion import html_to_xml_table, prettify_xml
from ..utils.other import create_and_save_dataset


def main():
    comtqa = load_from_disk(f"../../data/ComTQA_data/comtqa_updated_2024-12-03")
    numericnlg = load_from_disk("../../data/numericNLG/numericnlg_updated_2024-11-28")

    comtqa_df = comtqa["train"].to_pandas()
    numericnlg_df = numericnlg["test"].to_pandas()
    fintabnet_df = comtqa_df[comtqa_df["dataset"] == "FinTabNet"]

    columns_to_apply = ["row_headers", "column_headers", "contents"]
    for column in columns_to_apply:
        numericnlg_df[column] = numericnlg_df[column].apply(ast.literal_eval)

    numericnlg_xml = [
        html_to_xml_table(
            row.table_html_clean, row.table_id, row.table_name, row.caption
        )
        for row in numericnlg_df.itertuples(index=False)
    ]
    numericnlg_xml = [prettify_xml(xml) for xml in numericnlg_xml]
    numericnlg_df["table_xml"] = numericnlg_xml

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

    create_and_save_dataset(merged_df, "train", "../../data/ComTQA_data/comtqa_updated")
    create_and_save_dataset(
        numericnlg_df, "test", "../../data/numericNLG/numericnlg_updated"
    )


if __name__ == "__main__":
    main()
