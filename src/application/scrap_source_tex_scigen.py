"Script for collecting and unzipping files with latex source code for SciGen data"
import os
import requests
import logging
import tarfile
from datasets import load_from_disk
from ..utils.other import create_dir


logging.basicConfig(
    filename="download_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def fetch_arxiv_source(arxiv_url: str):
    try:
        response = requests.get(arxiv_url)
        response.raise_for_status()
        return response
    except Exception as e:
        logging.error(f"Error downloading from {arxiv_url}: {e}")
        return None


def save_to_file(zip_path: str, content):
    try:
        with open(zip_path, "wb") as f:
            f.write(content)
        print(f"Downloaded successfully to {zip_path}")
        return zip_path
    except Exception as e:
        logging.error(f"Error saving to {zip_path}: {e}")
        return None


def download_arxiv_source(arxiv_id: str, download_path: str):
    arxiv_url = f"https://arxiv.org/e-print/{arxiv_id}"
    print(f"Downloading from {arxiv_url}")
    create_dir(download_path)
    response = fetch_arxiv_source(arxiv_url)
    if not response:
        return None
    zip_path = os.path.join(download_path, f"{arxiv_id}.tar.gz")
    return save_to_file(zip_path, response.content)


def extract_all_tar_gz_in_dir(directory: str, save_path: str):

    for filename in os.listdir(directory):
        if filename.endswith(".tar.gz"):
            file_path = os.path.join(directory, filename)
            with tarfile.open(file_path, "r:gz") as tar:
                extract_dir = os.path.join(save_path, filename.replace(".tar.gz", ""))
                create_dir(extract_dir)
                tar.extractall(path=extract_dir)
                print(f"Extracted {filename} to {extract_dir}")


def main():
    data_paths = [
        "../../data/SciGen/test-CL/test_CL_all_meta_all_formats_2024_11_25.hf",
        "../../data/SciGen/test-Other/test_Other_all_meta_all_formats_2024_11_25.hf",
    ]

    dowload_paths = [
        "../../data/SciGen/test-Other/latex_source_code/",
        "../../data/SciGen/test-Other/latex_source_code/",
    ]

    for data_path, download_path in zip(data_paths, dowload_paths):
        scigen_data = load_from_disk(data_path)
        scigen_data = scigen_data.to_pandas()
        arxiv_ids = list(
            set(scigen_data.loc[scigen_data["venue"] == "arxiv", "paper_id"])
        )

        for paper_id in arxiv_ids:
            download_arxiv_source(paper_id, download_path)

    unzip_save_paths = [
        "../../../data/SciGen/test-CL/latex_source_code/unzipped/",
        "../../../data/SciGen/test-Other/latex_source_code/unzipped/",
    ]

    for download_path, save_path in zip(dowload_paths, unzip_save_paths):
        try:
            extract_all_tar_gz_in_dir(download_path, save_path)
        except Exception as e:
            print(f"Error extracting tar: {e} for dir: {download_path}")


if __name__ == "__main__":
    main()
