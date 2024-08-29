import json
import os
import requests
import xml.etree.ElementTree as ET


class DataPrep:
    def __init__(self, data, save_directory):
        self.data = data
        self.save_directory = save_directory

    def download_pdf(self, pdf_url):
        response = requests.get(pdf_url)
        response.raise_for_status()
        pdf_filename = os.path.join(self.save_directory, pdf_url.split("/")[-1])
        with open(pdf_filename, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {pdf_filename}")

    def download_pdf_by_title_arxiv(self, title):
        query = f'ti:"{title}"'
        url = f"http://export.arxiv.org/api/query?search_query={query}&start=0&max_results=5"
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
        return False

    def process_numericnlg(self):
        unique_paper_ids = list(set([item["paper_id"] for item in self.data]))
        os.makedirs(self.save_directory, exist_ok=True)

        for id in unique_paper_ids[:3]:
            source_url = f"http://aclanthology.org/{id}.pdf"
            try:
                self.download_pdf(source_url)
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                with open(f"{self.save_directory}/papers_not_found.txt", "a") as f:
                    f.write(
                        f"Error: {error_type}, error message: {error_message} for paper id {id}\n"
                    )

    def process_scigen(
        self,
    ):
        unique_paper_ids = set()
        downloaded_ids = set()

        for instance in self.data:
            paper_ids = {instance[key]["paper_id"] for key in instance.keys()}
            unique_paper_ids.update(paper_ids)

        paper_ids_list = list(unique_paper_ids)
        os.makedirs(self.save_directory, exist_ok=True)
        for id in paper_ids_list[:3]:
            source_url = f"https://arxiv.org/pdf/{id}.pdf"
            try:
                self.download_pdf(source_url)
                print("Downloading paper based on paper id", {id})
            except Exception as e:
                print(f"Failed to download paper by ID {id}, trying by title...")
                try:
                    paper_found = False
                    for key in instance.keys():
                        if instance[key]["paper_id"] == id:
                            paper_found = self.download_pdf_by_title_arxiv(
                                instance[key]["paper"]
                            )
                            if not paper_found:
                                paper_found = """"""  # need to include search through other resources
                            if paper_found:
                                print(
                                    f'Downloaded paper based on title: {instance[key]["paper"]}'
                                )
                                downloaded_ids.add(id)
                            break
                    if not paper_found:
                        with open(
                            f"{self.save_directory}/papers_not_found.txt", "a"
                        ) as f:
                            f.write(f"No valid paper found for ID or title: {id}\n")

                except Exception as e:
                    error_type = type(e).__name__
                    error_message = str(e)
                    with open(f"{self.save_directory}/papers_not_found.txt", "a") as f:
                        f.write(
                            f"Error: {error_type}, error message: {error_message} for paper id {id}\n"
                        )


def load_scigen_dataset(scigen_rootdir):
    data_list = []
    for file in os.listdir(scigen_rootdir):
        file_path = os.path.join(scigen_rootdir, file)
        with open(file_path) as f:
            data = json.load(f)
        data_list.append(data)
    return data_list
