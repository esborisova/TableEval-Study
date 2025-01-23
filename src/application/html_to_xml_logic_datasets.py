"""Script for generating XML, filtering not fully matched tables and 
removing table type indication in HTML for LogicNLG and Logic2Text data."""

from datasets import load_from_disk
import pandas as pd
from bs4 import BeautifulSoup
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
from ..utils.other import create_and_save_dataset
from ..utils.xml_html_convertion import change_table_class


def process_cell_attributes(cell, cell_elem):
    colspan = cell.get("colspan", None)
    rowspan = cell.get("rowspan", None)

    if colspan:
        cell_elem.set("colspan", colspan)
    if rowspan:
        cell_elem.set("rowspan", rowspan)


def extract_visible_text(cell):
    visible_text = ""
    for child in cell.children:
        if child.name == "span" and child.get("style") == "display:none":
            continue
        visible_text += child.get_text(strip=True)
    return visible_text


def process_special_cell_children(cell, cell_elem):
    for child in cell.contents:
        if child.name == "span" and "legend-color" in child.get("class", []):
            color_elem = SubElement(cell_elem, "color-box")
            style = child.get("style", "")
            color_style = next(
                (s for s in style.split(";") if "background-color" in s), None
            )
            if color_style:
                color = color_style.split(":")[1].strip()
                color_elem.set("background-color", color)
            color_elem.text = " "

        elif child.name == "a":
            link_href = child.get("href", "")
            link_elem = SubElement(cell_elem, "link", href=link_href)
            link_elem.text = child.get_text(strip=True)


def process_html_table_row(row, parent_elem):
    tr_elem = SubElement(parent_elem, "tr")

    for cell in row.find_all(["td", "th"]):
        cell_elem = SubElement(tr_elem, cell.name, align="left")
        process_cell_attributes(cell, cell_elem)

        if cell.find("style"):
            continue

        visible_text = extract_visible_text(cell)
        if visible_text:
            cell_elem.text = visible_text

        process_special_cell_children(cell, cell_elem)

    return tr_elem


def create_xml_table(xml_parent, html_table):
    pmc_table = SubElement(xml_parent, "table", frame="hsides", rules="groups")

    thead = SubElement(pmc_table, "thead")
    tbody = SubElement(pmc_table, "tbody")

    rows = html_table.find_all("tr")
    for _, tr in enumerate(rows):
        if tr.find("th"):
            process_html_table_row(tr, thead)
        else:
            process_html_table_row(tr, tbody)


def create_pmc_table_wrap(table_id=None, label_text=None, caption_text=None):
    table_wrap = Element("table-wrap", position="float")
    if table_id:
        table_wrap.set("id", table_id)
    if label_text:
        SubElement(table_wrap, "label").text = label_text
    if caption_text:
        caption = SubElement(table_wrap, "caption")
        SubElement(caption, "p").text = caption_text
    return table_wrap


def html_to_xml_table(
    html_table: str, table_id=None, label_text=None, caption_text=None
) -> str:

    soup = BeautifulSoup(html_table, "html.parser")
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


def define_agg_rules(df: pd.DataFrame, annotation: str, merged_annotation: str):
    aggregation_rules = {
        col: "first" for col in df.columns if col not in [annotation, "table_id"]
    }
    aggregation_rules[merged_annotation] = " ".join
    return aggregation_rules


def main():
    data_paths = [
        "../../data/LogicNLG/logicnlg_updated_2025-01-22",
        "../../data/Logic2Text/logic2text_updated_2025-01-22",
    ]

    for path in data_paths:
        data = load_from_disk(path)
        data = data["test"].to_pandas()

        data["image_name"] = data["filename"].apply(lambda x: x + ".png")

        data["table_xml"] = data["matched_table_html"].apply(html_to_xml_table)
        data["table_html"] = data["matched_table_html"].apply(change_table_class)

        if "LogicNLG" in path:
            # save intermediate results
            create_and_save_dataset(
                data, "test", "../../data/LogicNLG/logicnlg_with_xml_updated"
            )

            annotation = "ref"
            merged_annotation = f"joined_{annotation}"
            save_path = "../../data/LogicNLG/logicnlg_filtered_merged_ref_updated"
        else:
            # save intermediate results
            create_and_save_dataset(
                data, "test", "../../data/Logic2Text/logic2text_with_xml_updated"
            )

            annotation = "sent"
            merged_annotation = f"joined_{annotation}"
            save_path = "../../data/Logic2Text/logic2text_filtered_merged_sent_updated"

        # keep only fully matched tables
        data = data[data["issue"] == "none"]

        # merge individual statements
        data[annotation] = data[annotation].astype(str).str.strip()
        data[annotation] = data[annotation].apply(
            lambda x: x + "." if not x.endswith(".") else x
        )
        data[merged_annotation] = data[annotation]
        agg_rules = define_agg_rules(data, annotation, merged_annotation)
        data_merged = data.groupby("table_id", as_index=False).agg(agg_rules)
        create_and_save_dataset(data_merged, "test", save_path)


if __name__ == "__main__":
    main()
