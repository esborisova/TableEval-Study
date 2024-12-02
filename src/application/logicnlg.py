import requests

from datasets import Dataset, load_dataset
from datetime import datetime


data = load_dataset("kasnerz/logicnlg", split="train")
WIKI_SCRAPE_DATE = "2019-03-23"

# URL of the JSON file
mapping_url = "https://raw.githubusercontent.com/wenhuchen/LogicNLG/refs/heads/master/data/table_to_page.json"

# Fetch the JSON data
response = requests.get(mapping_url)

# Check if the request was successful
if response.status_code == 200:
    mapping = response.json()  # Parse JSON data
    #print(mapping)  # Print or process the JSON data
else:
    print(f"Failed to fetch data: {response.status_code}")

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

def split_table_column(example):
    try:
        table_data = eval(example['table'])  # Safely evaluate the string representation of the list
        if isinstance(table_data, list) and len(table_data) > 0 and isinstance(table_data[0], list):
            example['table_column_names'] = table_data[0]
            example['table_content_values'] = table_data[1:]
        else:
            example['table_column_names'] = None
            example['table_content_values'] = None
    except (SyntaxError, TypeError, IndexError):
        example['table_column_names'] = None
        example['table_content_values'] = None
    return example

dataset = dataset.select(range(3))  # FIXME: Remove after debugging!
dataset = dataset.map(split_table_column)

def get_archived_page_html(url, date):
    """
    Get the HTML content of a URL from the Wayback Machine as it was on a specific date.

    Args:
    - url: The URL of the page.
    - date: A datetime object representing the desired date.

    Returns:
    - The HTML content of the page as a string, or None if not found.
    """
    timestamp = date.strftime('%Y%m%d%H%M%S')
    wayback_url = f"http://archive.org/wayback/available?url={url}&timestamp={timestamp}"
    response = requests.get(wayback_url)
    if response.status_code != 200:
        print(f"Error accessing Wayback Machine for {url}: {response.status_code}")
        return None

    data = response.json()
    if 'archived_snapshots' in data and 'closest' in data['archived_snapshots']:
        snapshot_url = data['archived_snapshots']['closest']['url']
        snapshot_response = requests.get(snapshot_url)
        if snapshot_response.status_code == 200:
            return snapshot_response.text
    else:
        print(f"No archived version found for {url} at {timestamp}")
    return None

# Convert the date string to a datetime object
scrape_time = datetime.strptime(WIKI_SCRAPE_DATE, "%Y-%m-%d")
dataset = dataset.map(lambda example: {
    "html_content": get_archived_page_html(example["wiki_url"], scrape_time) if example["wiki_url"] else None})
print()
dataset.save_to_disk("../../data/logicnlg")
