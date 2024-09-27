from Bio import Entrez
from datasets import load_dataset
import os

def get_full_text_xml(pmcid):
    try:
        handle = Entrez.efetch(db="pmc", id=pmcid, rettype="xml", retmode="xml")
        xml_data = handle.read()
        handle.close()
        return xml_data
    except Exception as e:
        return f"Error retrieving full-text XML: {e}"


def save_xml_to_file(xml_content, filename):
    if isinstance(xml_content, bytes):
        xml_content = xml_content.decode("utf-8")
    with open(filename, "w", encoding="utf-8") as file:
        file.write(xml_content)


def collect_xml(pmc_ids, save_dir):
    for id in pmc_ids:
        xml_content = get_full_text_xml(id)
        file_path = os.path.join(save_dir, f"{id}.xml")
        save_xml_to_file(xml_content, file_path)


def main():
    Entrez.api_key = ""
    Entrez.email = ""
    save_dir = "../../data/pubmed/"

    ds = load_dataset("ByteDance/ComTQA")
    pubtab1m_data = ds.filter(lambda x: x["dataset"] == "PubTab1M")
    pmc_ids = [name.split("_")[0] for name in pubtab1m_data["train"]["image_name"]]
    print(len(pmc_ids))
    pmc_ids = list(set(pmc_ids))
    print(len(pmc_ids))
    collect_xml(pmc_ids, save_dir)


if __name__ == "__main__":
    main()
