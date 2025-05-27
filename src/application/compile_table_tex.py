"""Pipeline for creating tex files with tables and compiling them for numericNLG, 
Logic2Text, LogicNLG, and PMC subset in ComTQA"""

from datasets import load_from_disk
from ..utils.html_latex_convertion import (
    process_and_save_table_tex,
    compile_tex_files_in_dir,
)


def main():

    datasets = {
        "pmc": {
            "root_dir": "../../data/ComTQA_data/pubmed/latex_files/",
            "output_dir": "../../data/ComTQA_data/pubmed/latex_files/compilied_files/",
            "error_log_file": "../../data/ComTQA_data/pubmed/latex_files/compilied_files/logs/error_log.txt",
            "dataset_path": "../../data/ComTQA_data/comtqa_updated_2024-12-25",
            "split": "train",
            "filter_col": "dataset",
            "filter_val": "PubTab1M",
            "id_col": "id",
            "latex_col": "table_latex",
        },
        "fintabnet_source": {
            "root_dir": "../../data/ComTQA_data/fintabnet/latex_files/",
            "output_dir": "../../data/ComTQA_data/fintabnet/latex_files/compilied_files/",
            "error_log_file": "../../data/ComTQA_data/fintabnet/latex_files/compilied_files/logs/error_log.txt",
            "dataset_path": "../../data/ComTQA_data/comtqa_updated_2024-12-25",
            "split": "train",
            "filter_col": "dataset",
            "filter_val": "FinTabNet",
            "id_col": "table_id",
            "latex_col": "table_latex",
        },
        "fintabnet_spacylayout": {
            "root_dir": "../../data/ComTQA_data/fintabnet/latex_files_spacylayout/",
            "output_dir": "../../data/ComTQA_data/fintabnet/latex_files_spacylayout/compilied_files/",
            "error_log_file": "../../data/ComTQA_data/fintabnet/latex_files_spacylayout/compilied_files/logs/error_log.txt",
            "dataset_path": "../../data/ComTQA_data/comtqa_updated_2024-12-25",
            "split": "train",
            "filter_col": "dataset",
            "filter_val": "FinTabNet",
            "id_col": "table_id",
            "latex_col": "table_latex_spacylayout",
        },
        "numericnlg": {
            "root_dir": "../../data/numericNLG/latex_files/",
            "output_dir": "../../data/numericNLG/latex_files/compilied_files/",
            "error_log_file": "../../data/numericNLG/latex_files/compilied_files/logs/error_log.txt",
            "dataset_path": "../../data/numericNLG/numericnlg_updated_2024-12-25",
            "split": "test",
            "id_col": "table_id_paper",
            "latex_col": "table_latex",
        },
        "logic2text": {
            "root_dir": "../../data/Logic2Text/latex_files/",
            "output_dir": "../../data/Logic2Text/latex_files/compiled_files/",
            "error_log_file": "../../data/Logic2Text/latex_files/compiled_files/logs/error_log.txt",
            "dataset_path": "../../data/Logic2Text/logic2text_filtered_updated_2025-02-23",
            "split": "test",
            "id_col": "table_id",
            "latex_col": "table_latex",
        },
        "logicnlg": {
            "root_dir": "../../data/LogicNLG/latex_files/",
            "output_dir": "../../data/LogicNLG/latex_files/compiled_files/",
            "error_log_file": "../../data/LogicNLG/latex_files/compiled_files/logs/error_log.txt",
            "dataset_path": "../../data/LogicNLG/logicnlg_updated_2025-02-23",
            "split": "test",
            "id_col": "table_id",
            "latex_col": "table_latex",
        },
    }

    for _, config in datasets.items():
        dataset = load_from_disk(config["dataset_path"])
        df = dataset[config["split"].lower()].to_pandas()

        if "filter_col" in config and "filter_val" in config:
            df = df[df[config["filter_col"]] == config["filter_val"]]

        process_and_save_table_tex(
            df, config["id_col"], config["latex_col"], config["root_dir"]
        )

        compile_tex_files_in_dir(
            tex_dir=config["root_dir"],
            output_dir=config["output_dir"],
            error_log_file=config["error_log_file"],
        )


if __name__ == "__main__":
    main()
