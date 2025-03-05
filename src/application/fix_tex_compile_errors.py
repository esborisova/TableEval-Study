"""Script for fixing remaining latex compile errors in numericNLG, Logic2Text, LogicNLG, and ComTQA PMC subset instances"""

import pandas as pd
import re
from datasets import load_from_disk
from ..utils.other import create_and_save_dataset


def replace_latex_symbols_df(df, column_name, condition, old_sybm, new_symb):
    df.loc[condition, column_name] = df.loc[condition, column_name].apply(
        lambda x: x.replace(old_sybm, new_symb) if isinstance(x, str) else x
    )


def main():
    comtqa = load_from_disk("../../data/ComTQA_data/comtqa_updated_2024-12-25")
    comtqa_df = comtqa["train"].to_pandas()
    comtqa_condition = (comtqa_df["dataset"] == "PubTab1M") & (
        comtqa_df["id"] == "PMC1079919"
    )
    replace_latex_symbols_df(comtqa_df, "table_latex", comtqa_condition, "$", r"\$")
    create_and_save_dataset(comtqa_df, "train", "../../data/ComTQA_data/comtqa_updated")

    numericnlg = load_from_disk("../../data/numericNLG/numericnlg_updated_2024-12-25")
    numericnlg_df = numericnlg["test"].to_pandas()
    numericnlg_condition = numericnlg_df["table_id_paper"].isin(
        ["P18-1021table_5", "D18-1176table_1"]
    )
    replace_latex_symbols_df(
        numericnlg_df, "table_latex", numericnlg_condition, r"\}", r"$\pm$"
    )
    create_and_save_dataset(
        numericnlg_df, "test", "../../data/numericNLG/numericnlg_updated"
    )

    logic2text = load_from_disk("../../data/Logic2Text/logic2text_updated_2025-02-23")
    logic2text_df = logic2text["test"].to_pandas()
    logic2text_condition = logic2text_df["filename"].isin(
        [
            "2-17290159-1.html.csv",
            "2-11622562-3.html.csv",
            "2-16514242-7.html.csv",
            "2-18369370-4.html.csv",
        ]
    )
    replace_latex_symbols_df(
        logic2text_df, "table_latex", logic2text_condition, "$", r"\$"
    )
    create_and_save_dataset(
        logic2text_df, "test", "../../data/Logic2Text/logic2text_updated"
    )

    logicnlg = load_from_disk("../../data/LogicNLG/logicnlg_updated_2025-02-23")
    logicnlg_df = logicnlg["test"].to_pandas()
    logicnlg_condition = logicnlg_df["filename"].isin(
        [
            "2-14583258-3.html.csv",
            "2-11603116-4.html.csv",
            "2-14640450-3.html.csv",
            "2-12586672-1.html.csv",
            "2-14611590-3.html.csv",
            "2-14611590-4.html.csv",
            "2-15346009-4.html.csv",
            "2-18048776-7.html.csv",
        ]
    )
    replace_latex_symbols_df(logicnlg_df, "table_latex", logicnlg_condition, "$", r"\$")
    create_and_save_dataset(logicnlg_df, "test", "../../data/LogicNLG/logicnlg_updated")

    fintabnet = load_from_disk("../../data/ComTQA_data/comtqa_fin_updated_2025-03-04")
    fintabnet = fintabnet["test"].to_pandas()
    log_files = {
        "spacylayout": "../../data/ComTQA_data/fintabnet/latex_files_spacylayout/compilied_files/logs/error_log.txt",
        "source": "../../data/ComTQA_data/fintabnet/latex_files/compilied_files/logs/error_log.txt",
    }

    files_to_fix = {
        key: [
            file.split(".")[0]
            for file in re.findall(r"File: .*?/([^/]+\.tex)", open(path).read())
        ]
        for key, path in log_files.items()
    }

    files_to_fix_spacy = files_to_fix["spacylayout"]
    files_to_fix_source = files_to_fix["latex"]

    column_id_map = {
        "table_latex": files_to_fix_source,
        "table_latex_spacylayout": files_to_fix_spacy,
    }

    for col, id_list in column_id_map.items():
        fintabnet[col] = fintabnet.apply(
            lambda row: (
                re.sub(r"(?<!\\)\$", r"\\$", row[col])
                if isinstance(row[col], str) and row["table_id"] in id_list
                else row[col]
            ),
            axis=1,
        )

    fintabnet["table_latex_spacylayout"] = fintabnet.apply(
        lambda row: (
            row["table_latex_spacylayout"].replace(r"\cdot", r"$\cdot$")
            if row["table_id"] == "90513"
            else row["table_latex_spacylayout"]
        ),
        axis=1,
    )

    fintabnet["table_latex_spacylayout"] = fintabnet.apply(
        lambda row: (
            row["table_latex_spacylayout"].replace(r"�", r"…")
            if row["table_id"] == "69285"
            else row["table_latex_spacylayout"]
        ),
        axis=1,
    )

    create_and_save_dataset(
        fintabnet, "test", "../../data/ComTQA_data/comtqa_fin_updated"
    )


if __name__ == "__main__":
    main()
