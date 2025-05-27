import requests
from typing import Tuple, Optional
from bs4 import BeautifulSoup
from urllib.parse import urlparse


def get_html(url: str) -> Tuple[Optional[str]]:
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None


def parse_license_url(url: str) -> str:
    parsed_url = urlparse(url)
    path = parsed_url.path

    segments = path.split("licenses/")
    if len(segments) > 1:
        license_segment = segments[1].strip("/")
        formatted_license_info = license_segment.replace("/", " ")
        return formatted_license_info
    else:
        return url


def get_license_venue_info(url: str) -> Tuple[Optional[str], Optional[str]]:
    response = get_html(url)

    if response is None:
        license = "cc by"
        venue = "acl"
        return license, venue
    soup = BeautifulSoup(response, "html.parser")
    license_divs = soup.find_all("div", class_="abs-license")

    venue = "arxiv"
    if license_divs:
        licence_url = license_divs[0].find("a")["href"]
        parsed_license_url = parse_license_url(licence_url)
        return parsed_license_url, venue
    else:
        return None, venue


def add_license_venue_scigen(data: dict) -> dict:
    for key, item in data.items():
        paper_id = item.get("paper_id")
        url = f"https://arxiv.org/abs/{paper_id}"
        license, venue = get_license_venue_info(url)
        item["license"] = license
        item["venue"] = venue
    return data


def add_venue_license_numericnlg(example: dict) -> dict:
    example["venue"] = "acl"
    example["license"] = "cc by"
    return example
