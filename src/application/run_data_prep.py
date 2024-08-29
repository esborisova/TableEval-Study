import datasets
from datasets import load_dataset
from ..utils.prepare_data import DataPrep, load_scigen_dataset


def main():
    file_names = ["numericnlg", "scigen"]
    scigen_rootdir = "../../data/SciGen/test_set/"
    save_directory_numericnlg = "../../data/numericNLG/pdfs/"
    save_directory_scigen = "../../data/SciGen/pdfs/"

    for file_name in file_names:
        if "numericnlg" in file_name:
            data = load_dataset("kasnerz/numericnlg", split="test")
            data_pre_instance = DataPrep(data, save_directory_numericnlg)
            data_pre_instance.process_numericnlg()
        data_list = load_scigen_dataset(scigen_rootdir)
        data_pre_instance = DataPrep(data_list, save_directory_scigen)
        data_pre_instance.process_scigen()


if __name__ == "__main__":
    main()
