from bs4 import BeautifulSoup
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
from lxml import etree
from thefuzz import fuzz
import os
import pandas as pd
import re
from other import find_file, read_html


def map_xml_html_tags(soup, conversion_type="html_to_xml"):

    tag_replacements = {
        "html_to_xml": {
            "strong": "bold",
            "em": "italic",
            "br": "break",
        },
        "xml_to_html": {
            "bold": "strong",
            "italic": "em",
            "break": "br",
        },
    }

    replacements = tag_replacements[conversion_type]

    for html_tag, xml_tag in replacements.items():
        for tag in soup.find_all(html_tag):
            if html_tag == "br" or xml_tag == "break":
                new_tag = soup.new_tag(xml_tag)
                tag.replace_with(new_tag)
            else:
                tag.name = xml_tag
    return soup


def extract_footnotes(soup):
    """Extract footnotes content."""
    return [
        " ".join([p.get_text(strip=True) for p in footnote.find_all("p")])
        for footnote in soup.find_all("table-wrap-foot")
    ]


def add_footnotes_to_html_table(html_table: str, footnotes):
    footnotes_html = "".join([f"<tr><td>{content}</td></tr>" for content in footnotes])
    html_table = html_table.replace(
        "</table>", f"<tfoot>{footnotes_html}</tfoot></table>"
    )
    return html_table


def pmc_tables_to_html(xml_input: str, meta: bool = True) -> str:
    soup = BeautifulSoup(xml_input, "xml")
    soup = map_xml_html_tags(soup, "xml_to_html")

    table = soup.find("table")
    caption_html = ""

    if meta:
        caption_tag = soup.find("caption")
        if caption_tag:
            p_tag = caption_tag.find("p")
            if p_tag:
                p_tag.unwrap()
            caption_html = (
                f"<caption>{soup.find('label').get_text(strip=True)}: {caption_tag.get_text(strip=True)}</caption>"
                if soup.find("label")
                else f"<caption>{caption_tag.get_text(strip=True)}</caption>"
            )

        html_table = f'<table border="1" class="table">\n{caption_html}\n{table.prettify().strip()}'

        footnotes = extract_footnotes(soup)
        if footnotes:
            html_table = add_footnotes_to_html_table(html_table, footnotes)
    else:
        html_table = f'<table border="1" class="table">\n{table.prettify().strip()}'
    return html_table


def create_pmc_table_wrap(
    table_id: str = None, label_text: str = None, caption_text: str = None
):
    table_wrap = Element("table-wrap", position="float")
    if table_id:
        table_wrap.set("id", table_id)
    if label_text:
        SubElement(table_wrap, "label").text = label_text
    if caption_text:
        caption = SubElement(table_wrap, "caption")
        SubElement(caption, "p").text = caption_text
    return table_wrap


def process_html_table_row(row, parent_elem, row_type="td"):

    tr_elem = SubElement(parent_elem, "tr")

    for cell in row.find_all(row_type):
        cell_elem = SubElement(tr_elem, row_type, align="left")
        cell_text = cell.get_text(strip=True)

        cell_elem.text = cell_text if cell_text else " "

        for child in cell.contents:
            if child.name == "br":
                SubElement(cell_elem, "break")

    return tr_elem


def create_xml_table(xml_parent, html_table):
    pmc_table = SubElement(xml_parent, "table", frame="hsides", rules="groups")

    thead = SubElement(pmc_table, "thead")
    tbody = SubElement(pmc_table, "tbody")

    rows = html_table.find_all("tr")
    for tr in rows:
        if tr.find("th"):
            process_html_table_row(tr, thead, row_type="th")
        else:
            process_html_table_row(tr, tbody, row_type="td")


def html_to_xml_table(
    html_table: str,
    table_id: str = None,
    label_text: str = None,
    caption_text: str = None,
) -> str:

    soup = BeautifulSoup(html_table, "html.parser")

    soup = map_xml_html_tags(soup, conversion_type="html_to_xml")
    table_wrap = create_pmc_table_wrap(table_id, label_text, caption_text)
    html_table_elem = soup.find("table")
    if html_table_elem:
        create_xml_table(table_wrap, html_table_elem)

    rough_string = tostring(table_wrap, encoding="unicode", method="xml")
    pretty_xml = (
        minidom.parseString(rough_string)
        .toprettyxml(indent="  ", encoding="utf-8")
        .decode("utf-8")
    )

    return (
        pretty_xml.split("\n", 1)[1] if pretty_xml.startswith("<?xml") else pretty_xml
    )


def preprocess_target_caption(caption, format="html") -> str:
    if format == "html":
        return caption.get_text(strip=True).lower()
    else:
        caption = " ".join(caption.itertext()).strip()
        caption = caption.replace("\n", " ")
        caption = re.sub(r"\s+", " ", caption).lower()
        return caption


def preprocess_gold_caption(caption: str, format="html"):
    if format == "html":
        caption = caption.lower()
    else:
        caption = caption.strip()
        caption = caption.replace("\n", " ")
        caption = re.sub(r"\s+", " ", caption)
        caption = caption.lower()
    return caption


def extract_captions(file_path: str, format: str):
    namespaces = {"ltx": "http://dlmf.nist.gov/LaTeXML"}
    if format == "html":
        html_content = read_html(file_path)
        soup = BeautifulSoup(html_content, "lxml")
        captions = soup.find_all("figcaption")
    else:
        tree = etree.parse(file_path)
        root = tree.getroot()
        captions = root.findall(".//ltx:caption", namespaces=namespaces)
    return captions


def extract_table(caption, format):
    if format == "html":
        parent_table = str(caption.find_parent("figure").find("table"))
    else:
        parent_table = caption.getparent()
        while parent_table is not None and not parent_table.tag.endswith("table"):
            parent_table = parent_table.getparent()
        parent_table = (
            etree.tostring(parent_table, pretty_print=True).decode()
            if parent_table is not None
            else None
        )
    return parent_table


def find_best_match(df: pd.DataFrame, files_dir: str, format="html") -> pd.DataFrame:
    """
    Finds the best table caption match for each entry in a DataFrame.

    Args:
        dataframe (pd.DataFrame): DataFrame containing 'table_caption' and 'paper_id'.
        files_dir (str): Path to the folder containing HTML or XML files.
        file_type (str): Type of files to process, html or xml (default is 'html').
        format (str): Either xml or html. Default is 'html'.
    Returns:
        pd.DataFrame: DataFrame containing best match details for each row.
    """

    best_matches = []

    for _, row in df.iterrows():
        gold_caption = preprocess_gold_caption(row["table_caption"], format)
        paper_id = row["paper_id"]
        highest_score = 0
        if format == "html":
            table_format = f"table_{format}"
        else:
            table_format = f"table_{format}"
        best_table = {
            "paper_id": paper_id,
            "image_id": row["image_id"],
            "text": row["text"],
            "gold_caption": row["table_caption"],
            "target_caption": None,
            table_format: None,
            "fuzzy_score": None,
        }

        latexml_files = find_file(files_dir, paper_id, format)

        for file in latexml_files:
            file_path = os.path.join(files_dir, paper_id, file)
            captions = extract_captions(file_path, format)

            for caption in captions:
                target_caption = preprocess_target_caption(caption, format)
                fuzzy_match_score = fuzz.ratio(gold_caption, target_caption)

                if fuzzy_match_score > highest_score:
                    highest_score = fuzzy_match_score
                    best_table["fuzzy_score"] = fuzzy_match_score

                    if format == "html":
                        best_table["target_caption"] = (caption.get_text(strip=True),)
                        best_table[table_format] = extract_table(caption, format)

                    else:
                        best_table["target_caption"] = " ".join(
                            caption.itertext()
                        ).strip()
                        best_table[table_format] = extract_table(caption, format)

        best_matches.append(best_table)
    return pd.DataFrame(best_matches)


def add_table_metadata(html_table: str, caption_text: str, table_name=None) -> str:
    if table_name:
        caption = f"<caption>{table_name}: {caption_text}</caption>\n"
    else:
        caption = f"<caption>{caption_text}</caption>\n"
    if "<table" in html_table:
        table_parts = html_table.split(">", 1)
        modified_html = table_parts[0] + ">\n" + caption + table_parts[1]
    else:
        modified_html = caption + html_table
    return modified_html


def change_table_class(html: str, new_class="table") -> str:
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all("table"):
        table.attrs["class"] = [new_class]
    cleaned_html = str(soup).strip()
    return cleaned_html
