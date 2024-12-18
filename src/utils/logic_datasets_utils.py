import time
import unicodedata
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def safe_requests_get(url, retries=3, backoff_factor=1.0):
    for i in range(retries):
        response = requests.get(url)
        if response.status_code == 200:
            return response
        else:
            time.sleep(backoff_factor * (2 ** i))
    print(f"Failed after {retries} retries for: {url}")
    return None


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


def fetch_snapshot_content(url, ts):
    """
    Given a timestamp 'ts' (string) and a URL, return the HTML content and the snapshot timestamp as datetime.
    """
    snapshot_url = f"http://web.archive.org/web/{ts}/{url}"
    page_response = safe_requests_get(snapshot_url)
    if page_response and page_response.status_code == 200:
        # Convert ts string (YYYYMMDDHHMMSS) to a datetime object
        snap_date = datetime.strptime(ts, '%Y%m%d%H%M%S')
        return page_response.text, snap_date
    return None, None


def get_closest_snapshot(url, date):
    from_year = "2014"
    current_year = datetime.now().year
    cdx_url = (
        "http://web.archive.org/cdx/search/cdx"
        f"?url={url}&from={from_year}&to={current_year}"
        "&fl=timestamp&filter=statuscode:200&collapse=digest&output=json"
    )

    response = safe_requests_get(cdx_url)
    if not response or response.status_code != 200:
        return None, None

    data = response.json()
    if len(data) <= 1:
        return None, None

    # Parse all snapshots and find the one closest to 'date'
    snapshots = []
    for row in data[1:]:
        ts = row[0]
        snap_date = datetime.strptime(ts, '%Y%m%d%H%M%S')
        time_diff = abs((snap_date - date).total_seconds())
        snapshots.append((time_diff, ts))

    if not snapshots:
        return None, None

    # Sort by the absolute time difference
    snapshots.sort(key=lambda x: x[0])
    closest_snap = snapshots[0]
    return fetch_snapshot_content(url, closest_snap[1])  # returns (html, dt)


def get_archived_page_html(url, date):
    if not url:
        return None, None

    timestamp = date.strftime('%Y%m%d%H%M%S')
    wayback_url = f"http://archive.org/wayback/available?url={url}&timestamp={timestamp}"
    response = safe_requests_get(wayback_url)
    if not response or response.status_code != 200:
        # Direct lookup failed, try closest snapshot
        closest_html, closest_dt = get_closest_snapshot(url, date)
        if closest_html:
            return closest_html, closest_dt
        # Fallback to live page
        live_page = safe_requests_get(url)
        if live_page and live_page.status_code == 200:
            return live_page.text, datetime.now()
        return None, None

    data = response.json()
    if 'archived_snapshots' in data and 'closest' in data['archived_snapshots']:
        snap = data['archived_snapshots']['closest']
        snap_ts = snap['timestamp']
        snap_date = datetime.strptime(snap_ts, '%Y%m%d%H%M%S')

        # If this snapshot is acceptable (before or after doesn't matter now, we just have one),
        # try fetching it directly.
        html, dt = fetch_snapshot_content(url, snap_ts)
        if html:
            return html, dt
        else:
            # If the direct snapshot failed, try the closest snapshot overall
            closest_html, closest_dt = get_closest_snapshot(url, date)
            if closest_html:
                return closest_html, closest_dt
            # Fallback to live page
            live_page = safe_requests_get(url)
            if live_page and live_page.status_code == 200:
                return live_page.text, datetime.now()
            return None, None
    else:
        # No direct snapshot from wayback/available, just use the closest snapshot
        closest_html, closest_dt = get_closest_snapshot(url, date)
        if closest_html:
            return closest_html, closest_dt
        # Fallback to live page
        live_page = safe_requests_get(url)
        if live_page and live_page.status_code == 200:
            return live_page.text, datetime.now()
        return None, None


def fetch_html(example, cache, scrape_time, wiki_url_column_name="wiki_url"):
    table_id = example["table_id"]
    if table_id in cache:
        html, dt = cache[table_id]
        dt_str = dt.isoformat() if dt else None
        return {"html_content": html, "snapshot_timestamp": dt_str}
    else:
        html, dt = get_archived_page_html(example[wiki_url_column_name], scrape_time)
        cache[table_id] = (html, dt)
        dt_str = dt.isoformat() if dt else None
        return {"html_content": html, "snapshot_timestamp": dt_str}


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
        return ("", -1.0)

    # Create the desired string from dataset fields "title" and "table"
    desired_str = normalize_value(title) + normalize_value(table)

    soup = BeautifulSoup(html_content, 'html.parser')
    tables = soup.find_all('table', class_='wikitable')

    if not tables:
        return ("", -1.0)

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
    if best_table_html is None:
        best_table_html = ""
    if best_similarity is None:
        best_similarity = -1.0

    return best_table_html, best_similarity
