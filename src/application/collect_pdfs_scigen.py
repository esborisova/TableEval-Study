import requests
import json
import os
import xml.etree.ElementTree as ET


def download_pdf(pdf_url, save_directory):
    response = requests.get(pdf_url)
    response.raise_for_status()
    pdf_filename = os.path.join(save_directory, pdf_url.split("/")[-1])
    with open(pdf_filename, "wb") as f:
        f.write(response.content)
    print(f"Downloaded {pdf_filename}")


def download_pdf_by_title_arxiv(title, save_directory):
    query = f'ti:"{title}"'
    url = (
        f"http://export.arxiv.org/api/query?search_query={query}&start=0&max_results=5"
    )
    response = requests.get(url)
    response.raise_for_status()
    root = ET.fromstring(response.content)

    entries = root.findall("{http://www.w3.org/2005/Atom}entry")
    if entries:
        for entry in entries:
            paper_title = (
                entry.find("{http://www.w3.org/2005/Atom}title").text.strip().lower()
            )
            if title.strip().lower() in paper_title:
                paper_id = entry.find("{http://www.w3.org/2005/Atom}id").text
                pdf_url = paper_id.replace("abs", "pdf") + ".pdf"
                download_pdf(pdf_url, save_directory)
                return True
    return False


def main():
    rootdir = "../../data/SciGen/test_set/"
    unique_paper_ids = set()
    downloaded_ids = set()

    for file in os.listdir(rootdir):
        file_path = os.path.join(rootdir, file)

        with open(file_path) as f:
            data = json.load(f)

        paper_ids = {data[key]["paper_id"] for key in data.keys()}
        unique_paper_ids.update(paper_ids)

    paper_ids_list = list(unique_paper_ids)
   # paper_ids_list = [id for id in paper_ids_list if len(id) < 2]
    #print(paper_ids_list)
    save_directory = "../../data/SciGen/test_pdfs/"
    os.makedirs(save_directory, exist_ok=True)

    for id in paper_ids_list:
        source_url = f"https://arxiv.org/pdf/{id}.pdf"
        try:
            download_pdf(source_url, save_directory)
            print("Downloading paper based on paper id", {id})

        except Exception as e:
            print(f"Failed to download paper by ID {id}, trying by title...")
            try:
                paper_found = False
                for key in data.keys():
                    if data[key]["paper_id"] == id:
                        paper_found = download_pdf_by_title_arxiv(
                            data[key]["paper"], save_directory
                        )
                        if not paper_found:
                            paper_found = (
                                """"""  # need to include search through other resources
                            )
                        if paper_found:
                            print(
                                f'Downloaded paper based on title: {data[key]["paper"]}'
                            )
                            downloaded_ids.add(id)
                        break
                if not paper_found:
                    with open("not_valid_ids_scigen.txt", "a") as f:
                        f.write(f"No valid paper found for ID or title: {id}\n")

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                with open("not_valid_ids_scigen.txt", "a") as f:
                    f.write(
                        f"Error: {error_type}, error message: {error_message} for paper id {id}\n"
                    )


if __name__ == "__main__":
    main()
