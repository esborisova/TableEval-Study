import json
import os
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import logging


class DataPrep:
    def __init__(self, data, save_directory, log_level=logging.INFO):
        self.data = data
        self.save_directory = save_directory
        os.makedirs(save_directory, exist_ok=True)

        logging.basicConfig(
            filename=os.path.join(save_directory, "data_prep.log"),
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def download_pdf(self, pdf_url):
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

    def download_pdf_by_title_arxiv(self, title):
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
                        self.download_pdf(pdf_url)
                        return True
        except Exception as e:
            self.logger.error(
                f"Failed to search or download from arXiv for title {title}: {e}"
            )
        return False

    def download_pdf_by_title_acl(self, title):
        try:
            df = pd.read_pickle(
                "../../data/acl-publication-info.74k.v3.full-sections-partial-topic-labels.pkl"
            )
            for _, row in df.iterrows():
                if title.lower() in row["title"].lower():
                    acl_id = row["acl_id"]
                    source_url = f"http://aclanthology.org/{acl_id}.pdf"
                    self.download_pdf(source_url)
                    return True
        except Exception as e:
            self.logger.error(
                f"Failed to search or download from ACL Anthology for title {title}: {e}"
            )
        return False

    def log_download_failure(self, paper_id):
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

    def process_scigen(
        self,
    ):
        unique_paper_ids = set()

        for instance in self.data:
            paper_ids = {instance[key]["paper_id"] for key in instance.keys()}
            unique_paper_ids.update(paper_ids)

        paper_ids_list = list(unique_paper_ids)

        paper_found = False
        for id in paper_ids_list:
            source_url = f"https://arxiv.org/pdf/{id}.pdf"
            paper_found = self.download_pdf(source_url)
            if not paper_found:
                for instance in self.data:
                    for key in instance.keys():
                        if (
                            instance[key]["paper_id"] == id
                        ):  # need to fix dowload of the same file
                            title = instance[key]["paper"]
                            paper_found = self.download_pdf_by_title_arxiv(title)
                            if not paper_found:
                                paper_found = self.download_pdf_by_title_acl(title)
                                if not paper_found:
                                    self.log_download_failure(id)


def load_scigen_dataset(scigen_rootdir):
    data_list = []
    for file in os.listdir(scigen_rootdir):
        file_path = os.path.join(scigen_rootdir, file)
        with open(file_path) as f:
            data = json.load(f)
        data_list.append(data)
    return data_list
