"""Converting PMC tables XML into HTML format."""

from datasets import load_from_disk
import re
from ..utils.xml_html_convertion import (
    pmc_tables_to_html,
    validate_html,
    remove_html_indicator,
    remove_empty_tags,
    prettify_html,
)
from ..utils.other import create_and_save_dataset


def main():
    comtqa = load_from_disk("../../data/ComTQA_data/comtqa_updated_2024-12-01")
    comtqa_df = comtqa["train"].to_pandas()

    pmc_subset = comtqa_df[comtqa_df["dataset"] == "PubTab1M"]
    pmc_xml = pmc_subset["table_xml"].tolist()
    pmc_xml_no_meta = pmc_subset["table_xml_no_meta"].tolist()

    pmc_html = [pmc_tables_to_html(table) for table in pmc_xml]
    pmc_html_no_meta = [pmc_tables_to_html(table) for table in pmc_xml_no_meta]

    # remove second table (by-product from the original xml)
    cleaned_html = [
        re.sub(r'<table\s+frame="[^"]*"\s+rules="[^"]*">', "", html)
        for html in pmc_html
    ]
    cleaned_html = [re.sub(r"\n\s*\n", "\n", html) for html in cleaned_html]
    cleaned_html_no_meta = [
        re.sub(r'<table\s+frame="[^"]*"\s+rules="[^"]*">', "", html)
        for html in pmc_html_no_meta
    ]
    cleaned_html_no_meta = [
        re.sub(r"\n\s*\n", "\n", html) for html in cleaned_html_no_meta
    ]
    pmc_subset["table_html"] = cleaned_html
    pmc_subset["table_html_no_meta"] = cleaned_html_no_meta

    validated_html = validate_html(pmc_subset, "table_html",  "id", "pmc_html_val")
    validated_html_no_meta = validate_html(pmc_subset, "table_html_no_meta", "id", "pmc_html_no_meta_val")

    validated_html = remove_html_indicator(validated_html)
    validated_html_no_meta = remove_html_indicator(validated_html_no_meta)

    validated_html = remove_empty_tags(validated_html)
    validated_html_no_meta = remove_empty_tags(validated_html_no_meta)

    pretty_html = [prettify_html(html) for html in validated_html]
    pretty_html_no_meta = [prettify_html(html) for html in validated_html_no_meta]

    pmc_subset["table_html"] = pretty_html
    pmc_subset["table_html_no_meta"] = pretty_html_no_meta

    merged_df = comtqa_df.merge(
        pmc_subset[
            [
                "id",
                "table_html",
                "table_html_no_meta",
                "table_title",
                "image_name",
                "question",
                "answer",
            ]
        ],
        on=["id", "image_name", "table_title", "question", "answer"],
        how="left",
        suffixes=("", "_pmc"),
    )
    merged_df["table_html"] = merged_df["table_html"].combine_first(
        merged_df["table_html_pmc"]
    )
    merged_df = merged_df.drop(columns=["table_html_pmc"])

    create_and_save_dataset(merged_df, "test", "../../data/ComTQA_data/comtqa_updated")


if __name__ == "__main__":
    main()
