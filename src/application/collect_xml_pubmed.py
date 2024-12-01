"""Pipeline for collecting PMC papers XML via API and extracting tables with their meta from those."""
from datasets import load_dataset
from Bio import Entrez
from datetime import datetime
from ..utils.process_xml_pmc import ProcessTableXML


def main():
    Entrez.api_key = ""
    Entrez.email = ""
    save_dir = "../../data/ComTQA_data/pubmed/xml/"

    ds = load_dataset("ByteDance/ComTQA")
    pubtab1m_data = ds.filter(lambda x: x["dataset"] == "PubTab1M")
    pmc_ids = [name.split("_")[0] for name in pubtab1m_data["train"]["image_name"]]
    pmc_ids = list(set(pmc_ids))
    processor = ProcessTableXML(pmc_ids, save_dir)
    processor.collect_xml()
    df_tables = processor.process_xml_files()
    date = datetime.now().strftime("%Y-%m-%d")
    df_tables.to_csv(
        f"../../data/ComTQA_data/pubmed/utils/pubmed_tables_updated_{date}.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
