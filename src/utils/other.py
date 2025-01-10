import json
import tarfile
import os
import pandas as pd
from datasets import Dataset, DatasetDict
from datetime import datetime


def read_json(file_path: str) -> list:
    with open(file_path) as f:
        return json.load(f)


def read_html(file: str):
    with open(os.path.join(file), "r") as f:
        content = f.read()
    return content


def create_dir(root_dir: str):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)


def copy_files(tar_dir: str, output_dir: str, image_files):
    with tarfile.open(tar_dir, "r:gz") as tar:
        os.makedirs(output_dir, exist_ok=True)
        tar.extractall(path=output_dir, members=image_files)


def find_file(root_folder: str, paper_id: str, file_format: str) -> List[str]:
    format = f".{file_format.lower()}"
    paper_folder = os.path.join(root_folder, paper_id)

    found_files = []
    if os.path.exists(paper_folder) and os.path.isdir(paper_folder):
        for root, _, files in os.walk(paper_folder):
            for file in files:
                if file.lower().endswith(format):
                    found_files.append(file)
    return found_files


def create_dataset_object(df: pd.DataFrame, split) -> Dataset:
    dataset = Dataset.from_pandas(df.reset_index(drop=True))
    dataset_dict = DatasetDict({split: dataset})
    return dataset_dict


def save_dataset_object(dataset_dict: dict, save_dir: str) -> None:
    date = datetime.now().strftime("%Y-%m-%d")
    dataset_dict.save_to_disk(f"{save_dir}_{date}")


def create_and_save_dataset(
    dataframe: pd.DataFrame, split: str, save_path: str
) -> None:
    dataset_dict = create_dataset_object(dataframe, split)
    save_dataset_object(dataset_dict, save_path)


def display_table(df: pd.DataFrame) -> str:
    return f"Table with columns: {', '.join(map(str, df.columns.tolist()))}"


def save_table_to_file(file_path: str, table: str):
    if not os.path.exists(file_path):
        try:
            with open(file_path, "w", encoding="utf-8") as tex_file:
                tex_file.write(table)
            print(f"Saved {file_path} successfully.")
        except Exception as e:
            print(f"Failed to save {file_path}: {e}")
    else:
        print(f"File {file_path} already exists. Skipping.")
