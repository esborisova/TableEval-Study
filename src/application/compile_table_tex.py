"""Pipeline for creating tex files with tables and compiling them for numericNLG PMC subset in ComTQA"""

from datasets import load_from_disk
from ..utils.html_latex_convertion import (
    process_and_save_table_tex,
    compile_tex_files_in_dir,
)


def main():
    root_dir_pmc = "../../data/ComTQA_data/pubmed/latex_files/"
    root_dir_numericnlg = "../../data/numericNLG/latex_files/"

    output_dir_pmc = "../../data/ComTQA_data/pubmed/latex_files/compilied_files/"
    output_dir_numericnlg = "../../data/numericNLG/latex_files/compilied_files/"

    error_log_file_pmc = (
        "../../data/ComTQA_data/pubmed/latex_files/compilied_files/logs/error_log.txt"
    )
    error_log_file_numericnlg = (
        "../../data/numericNLG/latex_files/compilied_files/logs/error_log.txt"
    )

    comtqa = load_from_disk("../../data/ComTQA_data/comtqa_updated_2024-12-25")
    comtqa = comtqa["train"].to_pandas()
    pmc_subset = comtqa[comtqa["dataset"] == "PubTab1M"]

    numericnlg = load_from_disk("../../data/numericNLG/numericnlg_updated_2024-12-25")
    numericnlg = numericnlg["test"].to_pandas()

    process_and_save_table_tex(pmc_subset, "id", "table_latex", root_dir_pmc)
    process_and_save_table_tex(
        numericnlg, "table_id_paper", "table_latex", root_dir_numericnlg
    )

    compile_tex_files_in_dir(
        tex_dir=root_dir_pmc,
        output_dir=output_dir_pmc,
        error_log_file=error_log_file_pmc,
    )

    compile_tex_files_in_dir(
        tex_dir=root_dir_numericnlg,
        output_dir=output_dir_numericnlg,
        error_log_file=error_log_file_numericnlg,
    )


if __name__ == "__main__":
    main()
