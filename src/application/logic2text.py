import requests_cache

from datasets import load_dataset
from datetime import datetime
from logicnlg import fetch_html, map_example


dataset = load_dataset("kasnerz/logic2text", split="test")

WIKI_SCRAPE_DATE = "2019-03-23"
scrape_time = datetime.strptime(WIKI_SCRAPE_DATE, "%Y-%m-%d")

page_cache = {}
requests_cache.install_cache('wayback_cache', expire_after=86400)  # 1-day expiration

def add_metadata(example):
    example["table_id"] = example["url"].split("/")[-1]
    example["title"] = example["topic"]
    example["table"] = example["table_header"] + example["table_cont"]
    return example

dataset = dataset.map(add_metadata)

dataset = dataset.select(range(3))

dataset = dataset.map(fetch_html, fn_kwargs={"wiki_url_column_name": "wiki"})

dataset = dataset.map(map_example)

print()
dataset.save_to_disk("../../data/logic2text")


