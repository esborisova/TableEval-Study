"""Pipeline for converting tables HTML into LaTeX in numericNLG, Logic2Text, LogicNLG, and PMC subset of ComTQA"""

from datasets import load_from_disk
from ..utils.other import read_json, create_and_save_dataset
from ..utils.html_latex_convertion import convert_html_to_latex_tables


def main():
    comtqa = load_from_disk("../../data/ComTQA_data/comtqa_updated_2024-12-06")
    comtqa = comtqa["train"].to_pandas()
    pmc_subset = comtqa[comtqa["dataset"] == "PubTab1M"]
    fin_subset = comtqa[comtqa["dataset"] == "FinTabNet"]

    numericnlg = load_from_disk("../../data/numericNLG/numericnlg_test_2024_12_17.hf")
    numericnlg = numericnlg.to_pandas()
    numericnlg = numericnlg.rename(columns={"table_latex": "table_latex_gemini"})
    numericnlg = numericnlg.drop(columns=["table_latex_source"])

    logic2text = load_from_disk(
        "../../data/Logic2Text/logic2text_filtered_updated_2025-01-28"
    )
    logic2text = logic2text["test"].to_pandas()

    logicnlg = load_from_disk(
        "../../data/LogicNLG/logicnlg_filtered_updated_2025-01-28"
    )
    logicnlg = logicnlg["test"].to_pandas()

    replacements_path = "../utils/latex_symbols.json"
    replacements = read_json(replacements_path)

    html_replacements = replacements["html_escape_characters"]
    tex_replacements = replacements["other_escape_characters"]
    all_replacements = html_replacements.copy()
    all_replacements.update(tex_replacements)

    tables_latex_pmc = convert_html_to_latex_tables(
        pmc_subset,
        html_column="table_html",
        html_replacements=html_replacements,
        tex_replacements=tex_replacements,
        all_replacements=all_replacements,
        title_column="table_title",
        caption_column="table_caption",
        footnote_column="table_footnote",
    )
    pmc_subset["table_latex"] = tables_latex_pmc
    merged_df = comtqa.merge(
        pmc_subset[["id", "image_name", "table_latex"]],
        on=["id", "image_name"],
        how="left",
    )

    tables_latex_fin_source = convert_html_to_latex_tables(
        fin_subset,
        html_column="table_html",
        html_replacements=html_replacements,
        tex_replacements=tex_replacements,
        all_replacements=all_replacements,
    )
    fin_subset["table_latex"] = tables_latex_fin_source

    tables_latex_fin_spacy = convert_html_to_latex_tables(
        fin_subset,
        html_column="table_html_spacylayout",
        html_replacements=html_replacements,
        tex_replacements=tex_replacements,
        all_replacements=all_replacements,
    )
    fin_subset["table_latex_spacylayout"] = tables_latex_fin_spacy

    merged_df = comtqa.merge(
        fin_subset[["id", "image_name", "table_latex", "table_latex_spacylayout"]],
        on=["id", "image_name"],
        how="left",
    )

    create_and_save_dataset(merged_df, "train", "../../data/ComTQA_data/comtqa_updated")

    tex_replacements_copy = (
        tex_replacements.copy()
    )  # remove parallel symbol sinse it is used
    # in numericnlg html to merge the headers
    tex_replacements.pop("||", None)

    all_replacements_copy = all_replacements.copy()
    all_replacements_copy.pop("||", None)

    tables_latex_numericnlg = convert_html_to_latex_tables(
        numericnlg,
        html_column="table_html_clean",
        html_replacements=html_replacements,
        tex_replacements=tex_replacements_copy,
        all_replacements=all_replacements_copy,
        title_column="table_name",
        caption_column="caption",
    )
    numericnlg["table_latex"] = tables_latex_numericnlg
    create_and_save_dataset(
        numericnlg, "test", "../../data/numericNLG/numericnlg_updated"
    )

    tables_latex_logic2text = convert_html_to_latex_tables(
        logic2text,
        "table_html",
        html_replacements,
        tex_replacements,
        all_replacements,
        caption_column="title",
    )
    logic2text["table_latex"] = tables_latex_logic2text
    create_and_save_dataset(
        logic2text, "test", "../../data/Logic2Text/logic2text_updated"
    )

    tables_latex_logicnlg = convert_html_to_latex_tables(
        logicnlg,
        "table_html",
        html_replacements,
        tex_replacements,
        all_replacements,
        caption_column="title",
    )
    logicnlg["table_latex"] = tables_latex_logicnlg
    create_and_save_dataset(logicnlg, "test", "../../data/LogicNLG/logicnlg_updated")


if __name__ == "__main__":
    main()
