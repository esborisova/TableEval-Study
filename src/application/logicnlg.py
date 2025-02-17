import json
import requests
import requests_cache

from datetime import datetime
from datasets import Dataset, load_dataset

from utils.logic_datasets_utils import safe_requests_get, split_table_column, fetch_html, \
    extract_matched_table_html_and_similarity


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

    # 5) Add dict representation from test_lm_data
    table_id = example["table_id"]
    original_test_dict = test_lm_data.get(table_id, None)

    # Convert the dict to a JSON string
    original_test_dict_str = json.dumps(original_test_dict) if original_test_dict is not None else None

    return {
        "matched_table_html": best_table_html,
        "matched_table_similarity": best_similarity,
        "original_test_dict": original_test_dict_str,
        "snapshot_timestamp": example.get("snapshot_timestamp")
    }


if __name__ == "__main__":
    page_cache = {}
    WIKI_SCRAPE_DATE = "2019-03-23"
    scrape_time = datetime.strptime(WIKI_SCRAPE_DATE, "%Y-%m-%d")

    data = load_dataset("kasnerz/logicnlg", split="test")

    # Load the original test_lm.json file to get the dict representation
    test_lm_url = "https://raw.githubusercontent.com/wenhuchen/LogicNLG/master/data/test_lm.json"
    test_lm_response = requests.get(test_lm_url)
    test_lm_data = test_lm_response.json()  # A dict keyed by table_id with a list of items

    # Fetch the JSON data
    mapping_url = "https://raw.githubusercontent.com/wenhuchen/LogicNLG/refs/heads/master/data/table_to_page.json"
    response = safe_requests_get(mapping_url)
    if response and response.status_code == 200:
        mapping = response.json()
    else:
        raise ValueError("No mapping has been downloaded.")

    # Ensure data is loaded as a list of dictionaries (converting from Hugging Face Dataset if needed)
    data = data.to_dict()  # Converts Hugging Face dataset to a Python dictionary format

    # Iterate over each entry in `data` and add mapping information if available
    for i, entry in enumerate(data["table_id"]):
        table_id = entry
        if table_id in mapping:
            # Add the mapping values to the current entry in `data`
            data["wiki_title"] = data.get("wiki_title", []) + [mapping[table_id][0]]
            data["wiki_url"] = data.get("wiki_url", []) + [mapping[table_id][1]]
        else:
            # If no mapping, add placeholders or empty values
            data["wiki_title"] = data.get("wiki_title", []) + [None]
            data["wiki_url"] = data.get("wiki_url", []) + [None]

    dataset = Dataset.from_dict(data)

    dataset = dataset.select(range(3))  # FIXME: Remove after debugging!
    dataset = dataset.map(split_table_column)

    requests_cache.install_cache('wayback_cache', expire_after=86400)  # 1-day expiration
    dataset = dataset.map(fetch_html, fn_kwargs={"cache": page_cache, "scrape_time": scrape_time})
    dataset = dataset.map(map_example)

    print()
    dataset.save_to_disk("../../data/logicnlg")
