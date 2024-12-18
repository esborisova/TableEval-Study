import json
import requests_cache

from datasets import load_dataset
from datetime import datetime
from logicnlg import fetch_html, extract_matched_table_html_and_similarity


def add_metadata(example):
    example["table_id"] = example["url"].split("/")[-1]
    example["title"] = example["topic"]
    example["table"] = example["table_header"] + example["table_cont"]
    return example

def map_example(example):
    # If html_content is None, skip by returning None fields
    if example.get("html_content") is None:
        return {
            "matched_table_html": None,
            "matched_table_similarity": None,
            "original_test_dict": None,
            "snapshot_timestamp": example.get("snapshot_timestamp")
        }

    best_table_html, best_similarity = extract_matched_table_html_and_similarity(
        example.get("html_content"),
        example.get("title", ""),
        example.get("table", [])
    )

    # Convert table_header from string to list
    table_header_str = example.get("table_header")
    table_header = eval(table_header_str) if table_header_str else []

    # Convert table_cont from string to list of lists
    table_cont_str = example.get("table_cont")
    table_cont = eval(table_cont_str) if table_cont_str else []

    # Create a list of dicts, each dict corresponding to a row in table_cont
    dict_repr = []
    if table_header and table_cont:
        for row in table_cont:
            row_dict = dict(zip(table_header, row))
            dict_repr.append(row_dict)
    else:
        dict_repr = None
    # Convert the dict to a JSON string
    dict_str = json.dumps(dict_repr) if dict_repr is not None else None

    return {
        "matched_table_html": best_table_html,
        "matched_table_similarity": best_similarity,
        "dict_str": dict_str,
        "snapshot_timestamp": example.get("snapshot_timestamp")
    }


if __name__ == "__main__":
    dataset = load_dataset("kasnerz/logic2text", split="test")

    WIKI_SCRAPE_DATE = "2019-03-23"
    scrape_time = datetime.strptime(WIKI_SCRAPE_DATE, "%Y-%m-%d")

    page_cache = {}
    requests_cache.install_cache('wayback_cache', expire_after=86400)  # 1-day expiration

    dataset = dataset.map(add_metadata)

    dataset = dataset.select(range(3))

    dataset = dataset.map(fetch_html, fn_kwargs={"wiki_url_column_name": "wiki"})

    dataset = dataset.map(map_example)

    print()
    dataset.save_to_disk("../../data/logic2text")
