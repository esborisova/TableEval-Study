import json
import os
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import logging
from typing import Tuple, Optional


class DataPrep:
    def __init__(
        self, data: Tuple[str, dict], save_directory: str, log_level=logging.INFO
    ):
        self.data = data
        self.save_directory = save_directory
        os.makedirs(save_directory, exist_ok=True)

        logging.basicConfig(
            filename=os.path.join(save_directory, "data_prep.log"),
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def download_pdf(self, pdf_url: str) -> bool:
        try:
            response = requests.get(pdf_url)
            response.raise_for_status()
            pdf_filename = os.path.join(self.save_directory, pdf_url.split("/")[-1])
            with open(pdf_filename, "wb") as f:
                f.write(response.content)
            return True
        except Exception as e:
            self.logger.error(f"Failed to download {pdf_url}: {e}")
            return False

    def download_pdf_by_title_arxiv(self, title: str) -> Tuple[bool, Optional[str]]:
        query = f'ti:"{title}"'
        url = f"http://export.arxiv.org/api/query?search_query={query}&start=0&max_results=5"

        try:
            response = requests.get(url)
            response.raise_for_status()
            root = ET.fromstring(response.content)

            entries = root.findall("{http://www.w3.org/2005/Atom}entry")
            if entries:
                for entry in entries:
                    paper_title = (
                        entry.find("{http://www.w3.org/2005/Atom}title")
                        .text.strip()
                        .lower()
                    )
                    if title.strip().lower() in paper_title:
                        paper_id = entry.find("{http://www.w3.org/2005/Atom}id").text
                        pdf_url = paper_id.replace("abs", "pdf") + ".pdf"
                        arxiv_id = pdf_url.split("/")[-1].replace(".pdf", "")
                        self.download_pdf(pdf_url)
                        return True, arxiv_id
        except Exception as e:
            self.logger.error(
                f"Failed to search or download from arXiv for title {title}: {e}"
            )
        return False, None

    def download_pdf_by_title_acl(self, title: str) -> Tuple[bool, Optional[str]]:
        try:
            df = pd.read_pickle(
                "../../data/acl-publication-info.74k.v3.full-sections-partial-topic-labels.pkl"
            )
            for _, row in df.iterrows():
                if title.lower() in row["title"].lower():
                    acl_id = row["acl_id"]
                    source_url = f"http://aclanthology.org/{acl_id}.pdf"
                    self.download_pdf(source_url)
                    return True, acl_id
        except Exception as e:
            self.logger.error(
                f"Failed to search or download from ACL Anthology for title {title}: {e}"
            )
        return False, None

    def log_download_failure(self, paper_id: str):
        error_message = f"Failed to download paper with ID {paper_id}"
        self.logger.error(error_message)
        with open(os.path.join(self.save_directory, "papers_not_found.txt"), "a") as f:
            f.write(f"{error_message}\n")

    def process_numericnlg(self):
        unique_paper_ids = list(set([item["paper_id"] for item in self.data]))
        dowloaded_paper = False
        for id in unique_paper_ids:
            source_url = f"http://aclanthology.org/{id}.pdf"
            dowloaded_paper = self.download_pdf(source_url)
            if not dowloaded_paper:
                self.log_download_failure(id)

    def save_updated_data(self, filename: str, data: dict):
        """Save the updated data back to the original JSON file."""
        save_dir = "/".join(self.save_directory.split("/")[:-1])
        file_path = os.path.join(save_dir, filename)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    def process_scigen(
        self,
    ):
        paper_ids = [self.data[1][key]["paper_id"] for key in self.data[1].keys()]
        unique_paper_ids = list(set(paper_ids))
        downloaded_paper_ids = set()
        id_to_substitute = {}

        paper_found = False

        for id in unique_paper_ids:
            source_url = f"https://arxiv.org/pdf/{id}.pdf"
            paper_found = self.download_pdf(source_url)
            if not paper_found:
                for key in self.data[1].keys():
                    if (self.data[1][key]["paper_id"] == id) and (
                        id not in downloaded_paper_ids
                    ):
                        title = self.data[1][key]["paper"]
                        paper_found, arxiv_id = self.download_pdf_by_title_arxiv(title)
                        if paper_found:
                            self.data[1][key]["paper_id"] = arxiv_id
                            downloaded_paper_ids.add(id)
                            downloaded_paper_ids.add(arxiv_id)
                            id_to_substitute[id] = arxiv_id
                        else:
                            paper_found, acl_id = self.download_pdf_by_title_acl(title)
                            if paper_found:
                                self.data[1][key]["paper_id"] = acl_id
                                downloaded_paper_ids.add(id)
                                downloaded_paper_ids.add(acl_id)
                                id_to_substitute[id] = acl_id
                            else:
                                self.log_download_failure(id)

        for key in self.data[1].keys():
            current_id = self.data[1][key]["paper_id"]
            if current_id in id_to_substitute:
                self.data[1][key]["paper_id"] = id_to_substitute[current_id]

        updated_json_filename = "updated_" + self.data[0].split("/")[-1]
        self.save_updated_data(updated_json_filename, self.data[1])


def load_scigen_dataset(file_path: str, return_tuple = False) -> Tuple[str, dict]:
    """Load datasets from the directory and return as a list of (filename, data) tuples."""
    with open(file_path) as f:
        data = json.load(f)
    if not return_tuple:
        return data
    else:
        return (file_path, data)
