import requests

from datasets import Dataset, load_dataset

data = load_dataset("kasnerz/logicnlg", split="train")

# URL of the JSON file
mapping_url = "https://raw.githubusercontent.com/wenhuchen/LogicNLG/refs/heads/master/data/table_to_page.json"

# Fetch the JSON data
response = requests.get(mapping_url)

# Check if the request was successful
if response.status_code == 200:
    mapping = response.json()  # Parse JSON data
    print(mapping)  # Print or process the JSON data
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

