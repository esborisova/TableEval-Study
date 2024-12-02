from bs4 import BeautifulSoup


def map_pmc_to_html_tags(soup):
    """Replace PMC XML specific tags with HTML equivalents."""
    tag_replacements = {"bold": "strong", "italic": "em", "sup": "sup", "sub": "sub"}

    for pmc_tag, html_tag in tag_replacements.items():
        for tag in soup.find_all(pmc_tag):
            tag.name = html_tag
    return soup


def replace_line_breaks(soup):
    for break_tag in soup.find_all("break"):
        br_tag = soup.new_tag("br")
        break_tag.replace_with(br_tag)
    return soup


def generate_html_content(soup):
    return f"<html><head><meta charset='UTF-8'></head><body>{soup.prettify()}</body></html>"


def pmc_tables_to_html(xml_table: str) -> str:
    soup = BeautifulSoup(xml_table, "xml")
    soup = map_pmc_to_html_tags(soup)
    soup = replace_line_breaks(soup)
    html_table = generate_html_content(soup)
    return html_table
