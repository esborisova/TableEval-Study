import json
import tarfile
import os
import pandas as pd
from datasets import Dataset, DatasetDict
from datetime import datetime

def read_json(file_path: str) -> list:
    with open(file_path) as f:
        return json.load(f)
    
def copy_files(tar_dir, output_dir, image_files):
    with tarfile.open(tar_dir, "r:gz") as tar:
        os.makedirs(output_dir, exist_ok=True)
        tar.extractall(path=output_dir, members=image_files)

def create_dataset_object(df: pd.DataFrame) -> Dataset:
    dataset = Dataset.from_pandas(df.reset_index(drop=True))
    dataset_dict = DatasetDict({"train": dataset})
    return dataset_dict

def save_dataset_object(dataset_dict, save_dir: str) -> None:
    date = datetime.now().strftime("%Y-%m-%d")
    dataset_dict.save_to_disk(f"{save_dir}_{date}")
def display_table(df: pd.DataFrame) -> str:
    return f"Table with columns: {', '.join(map(str, df.columns.tolist()))}"