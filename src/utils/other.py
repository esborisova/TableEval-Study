import json
import tarfile
import os


def read_json(file_path: str) -> list:
    with open(file_path) as f:
        return json.load(f)
    

def copy_files(tar_dir, output_dir, image_files):
    with tarfile.open(tar_dir, "r:gz") as tar:
        os.makedirs(output_dir, exist_ok=True)
        tar.extractall(path=output_dir, members=image_files)
