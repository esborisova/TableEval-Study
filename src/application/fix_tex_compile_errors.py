"""Script for fixing remaining the latex compile errors in numericNLG and ComTQA PMC subset instances"""

import pandas as pd
from datasets import load_from_disk
from ..utils.other import create_and_save_dataset


def replace_latex_symbols_df(df, column_name, condition, old_sybm, new_symb):
    df.loc[condition, column_name] = df.loc[condition, column_name].apply(
        lambda x: x.replace(old_sybm, new_symb) if isinstance(x, str) else x
    )


def main():
    comtqa = load_from_disk("../../../data/ComTQA_data/comtqa_updated_2024-12-25")
    comtqa_df = comtqa["train"].to_pandas()
    comtqa_condition = (comtqa_df["dataset"] == "PubTab1M") & (
        comtqa_df["id"] == "PMC1079919"
    )
    replace_latex_symbols_df(comtqa_df, "table_latex", comtqa_condition, "$", r"\$")
    create_and_save_dataset(
        comtqa_df, "train", "../../../data/ComTQA_data/comtqa_updated"
    )

    numericnlg = load_from_disk(
        "../../../data/numericNLG/numericnlg_updated_2024-12-25"
    )
    numericnlg_df = numericnlg["test"].to_pandas()
    numericnlg_condition = numericnlg_df["table_id_paper"].isin(
        ["P18-1021table_5", "D18-1176table_1"]
    )
    replace_latex_symbols_df(
        numericnlg_df, "table_latex", numericnlg_condition, r"\}", r"$\pm$"
    )
    create_and_save_dataset(
        numericnlg_df, "test", "../../../data/numericNLG/numericnlg_updated"
    )


if __name__ == "__main__":
    main()
