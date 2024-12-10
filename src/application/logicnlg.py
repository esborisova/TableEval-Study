import requests
import requests_cache
import time
import unicodedata

from bs4 import BeautifulSoup
from datasets import Dataset, load_dataset
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


data = load_dataset("kasnerz/logicnlg", split="test")
WIKI_SCRAPE_DATE = "2019-03-23"

# URL of the JSON file
mapping_url = "https://raw.githubusercontent.com/wenhuchen/LogicNLG/refs/heads/master/data/table_to_page.json"


def safe_requests_get(url, retries=3, backoff_factor=1.0):
    for i in range(retries):
        response = requests.get(url)
        if response.status_code == 200:
            return response
        else:
            time.sleep(backoff_factor * (2 ** i))
    print(f"Failed after {retries} retries for: {url}")
    return None

# Fetch the JSON data
response = safe_requests_get(mapping_url)

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

#dataset = dataset.shuffle().select(range(10))  # FIXME: Remove after debugging!
dataset = dataset.map(split_table_column)

requests_cache.install_cache('wayback_cache', expire_after=86400)  # 1-day expiration

def get_earlier_snapshot(url, date):
    """
    Try to find the closest snapshot on or before the given date using the CDX API.
    """
    # Convert the date to a year range for CDX query
    from_year = "2014"
    to_year = date.year
    timestamp_str = date.strftime('%Y%m%d%H%M%S')

    cdx_url = (
        "http://web.archive.org/cdx/search/cdx"
        f"?url={url}&from={from_year}&to={to_year}"
        "&fl=timestamp&filter=statuscode:200&collapse=digest&output=json"
    )

    response = safe_requests_get(cdx_url)
    if response.status_code != 200:
        print(f"Error accessing CDX API for {url}: {response.status_code}")
        return None

    data = response.json()
    # data[0] will be the header row ['timestamp'], so actual timestamps start at data[1]
    if len(data) <= 1:
        print(f"No snapshots found for {url} in {from_year}")
        return None

    # Extract timestamps and convert to datetime
    snapshots = []
    for row in data[1:]:
        ts = row[0]
        snap_date = datetime.strptime(ts, '%Y%m%d%H%M%S')
        # Only consider snapshots that are on or before the requested date
        if snap_date <= date:
            snapshots.append((snap_date, ts))

    if not snapshots:
        # If none are before or on the date, consider after if you wish
        # or return None if you strictly want an earlier snapshot
        print(f"No earlier snapshot found for {url} at or before {timestamp_str}")
        return None

    # Find the snapshot closest to the desired date
    snapshots.sort(key=lambda x: x[0], reverse=True)
    closest_snap = snapshots[0]  # After sorting descending by date, first is the closest before date

    # Now retrieve the actual archived page
    snapshot_url = f"http://web.archive.org/web/{closest_snap[1]}/{url}"
    page_response = safe_requests_get(snapshot_url)
    if page_response.status_code == 200:
        return page_response.text
    else:
        print(f"Snapshot found but failed to retrieve content: {page_response.status_code}")
        return None

def get_archived_page_html(url, date):
    """
    Get the HTML content of a URL from the Wayback Machine as it was on or before a specific date.
    Fallbacks to the CDX API approach if the direct 'wayback/available' fails to return earlier snapshots.
    """
    timestamp = date.strftime('%Y%m%d%H%M%S')
    wayback_url = f"http://archive.org/wayback/available?url={url}&timestamp={timestamp}"
    response = safe_requests_get(wayback_url)
    if response.status_code != 200:
        print(f"Error accessing Wayback Machine for {url}: {response.status_code}")
        return get_earlier_snapshot(url, date)

    data = response.json()
    if 'archived_snapshots' in data and 'closest' in data['archived_snapshots']:
        snap = data['archived_snapshots']['closest']
        snap_ts = snap['timestamp']
        snap_date = datetime.strptime(snap_ts, '%Y%m%d%H%M%S')

        # Check if the snapshot is on or before the requested date
        if snap_date <= date:
            # This is acceptable; return it directly
            snapshot_url = snap['url']
            snapshot_response = safe_requests_get(snapshot_url)
            if snapshot_response.status_code == 200:
                return snapshot_response.text
            else:
                print(f"Failed to retrieve snapshot content: {snapshot_response.status_code}")
                return None
        else:
            # The closest snapshot is after the requested date, so fall back to CDX approach
            return get_earlier_snapshot(url, date)
    else:
        # No snapshot found directly, fall back to the CDX approach
        print(f"No archived version found for {url} at {timestamp} directly, checking CDX...")
        return get_earlier_snapshot(url, date)

# Convert the date string to a datetime object
scrape_time = datetime.strptime(WIKI_SCRAPE_DATE, "%Y-%m-%d")
dataset = dataset.map(lambda example: {
    "html_content": get_archived_page_html(example["wiki_url"], scrape_time) if example["wiki_url"] else None})
print()

def normalize_value(val):
    val_str = str(val).strip().lower()
    val_str = unicodedata.normalize('NFC', val_str)
    return val_str

def compute_cosine_similarity(vectorizer, text1: str, text2: str) -> float:
    vectors = vectorizer.transform([text1, text2]).toarray()
    return cosine_similarity(vectors)[0][1]

def extract_table_text_from_html(table_element):
    """
    Extract caption + all table cell texts from the given table_element.
    Return a normalized concatenated string.
    """
    parts = []
    # Extract caption if available
    caption = table_element.find('caption')
    if caption:
        parts.append(normalize_value(caption.get_text(strip=True)))

    # Extract all rows (including header rows)
    rows = table_element.find_all('tr')
    for row in rows:
        cells = row.find_all(['th', 'td'])
        for c in cells:
            parts.append(normalize_value(c.get_text(strip=True)))

    return " ".join(parts)

def extract_matched_table_html_and_similarity(html_content, title, table):
    if not html_content:
        return None

    # Create the desired string from dataset fields "title" and "table"
    desired_str = normalize_value(title) + normalize_value(table)

    soup = BeautifulSoup(html_content, 'html.parser')
    tables = soup.find_all('table', class_='wikitable')

    if not tables:
        return None

    # Fit vectorizer on desired_str to create vocabulary
    vectorizer = CountVectorizer().fit([desired_str])

    best_similarity = -1.0
    best_table_html = None

    for table_element in tables:
        extracted_str = extract_table_text_from_html(table_element)
        similarity = compute_cosine_similarity(vectorizer, desired_str, extracted_str)

        if similarity > best_similarity:
            best_similarity = similarity
            best_table_html = str(table_element)

    return best_table_html, best_similarity


def map_example(example):
    best_table_html, best_similarity = extract_matched_table_html_and_similarity(
        example.get("html_content"),
        example.get("title", ""),
        example.get("table", [])
    )
    return {
        "matched_table_html": best_table_html,
        "matched_table_similarity": best_similarity
    }

dataset = dataset.map(map_example)

print()
dataset.save_to_disk("../../data/logicnlg")
