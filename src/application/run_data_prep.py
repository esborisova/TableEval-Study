import datasets
from datasets import load_dataset
import os
from ..utils.prepare_data import DataPrep, load_scigen_dataset


def main():
    file_names = ["numericnlg", "scigen"]
    scigen_rootdir = "../../data/SciGen/original_test_sets/"
    save_directory_numericnlg = "../../data/test/numericNLG/"

    for file_name in file_names:
        if "numericnlg" in file_name:
            data = load_dataset("kasnerz/numericnlg", split="test")
            data_pre_instance = DataPrep(data, save_directory_numericnlg)
            data_pre_instance.process_numericnlg()

        scigen_file_names = os.listdir(scigen_rootdir)
        for name in scigen_file_names:
            file_path = os.path.join(scigen_rootdir, name)
            data_tuple = load_scigen_dataset(file_path)
            save_directory_scigen = f"../../data/SciGen/{name}/pdfs/"
            data_pre_instance = DataPrep(data_tuple, save_directory_scigen)


if __name__ == "__main__":
    main()
