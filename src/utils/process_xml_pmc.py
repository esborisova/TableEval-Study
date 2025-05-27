import os
import re
import pandas as pd
import xml.etree.ElementTree as ET
from Bio import Entrez
from typing import List, Tuple, Optional
import logging
from html import unescape


class ProcessTableXML:
    def __init__(self, pmc_ids: List[str], save_dir: str, log_level=logging.INFO):
        self.pmc_ids = pmc_ids
        self.save_dir = save_dir
        self.tables_df = pd.DataFrame()
        os.makedirs(save_dir, exist_ok=True)

        logging.basicConfig(
            filename=os.path.join(save_dir, "xml_processing.log"),
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def get_full_text_xml(self, pmcid: str):
        try:
            handle = Entrez.efetch(db="pmc", id=pmcid, rettype="xml", retmode="xml")
            xml_data = handle.read()
            handle.close()
            return xml_data
        except Exception as e:
            self.logger.error(f"Failed to download {pmcid}: {e}")
            return None

    def save_xml_to_file(self, xml_content, filename: str):
        if xml_content is None:
            self.logger.warning(
                f"Skipping saving XML to {filename} because content is None."
            )
            return

        if isinstance(xml_content, bytes):
            xml_content = xml_content.decode("utf-8")
        with open(filename, "w", encoding="utf-8") as file:
            file.write(xml_content)

    def collect_xml(self):
        for pmcid in self.pmc_ids:
            xml_content = self.get_full_text_xml(pmcid)
            if xml_content is None:
                self.logger.warning(f"Skipping {pmcid} due to download error.")
                continue
            file_path = os.path.join(self.save_dir, f"{pmcid}.xml")
            self.save_xml_to_file(xml_content, file_path)

    def read_xml_file(self, file_path: str):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                xml_content = file.read()
                if not xml_content:
                    self.logger.warning(f"File {file_path} is empty.")
                    return None
                return xml_content
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return None

    def extract_tables_from_xml(self, xml_content) -> list:
        if xml_content is None:
            self.logger.warning("No XML content to extract tables from.")
            return []

        if "Error retrieving full-text XML" in xml_content:
            self.logger.warning(
                "Error message found in XML content, skipping table extraction."
            )
            return []

        try:
            root = ET.fromstring(xml_content)
            tables = root.findall(".//table-wrap")
            return tables
        except ET.ParseError as e:
            self.logger.error(f"Error parsing XML content: {e}")
            return []

    def extract_table_title(self, table) -> Optional[str]:
        title = table.find(".//label")
        return (
            ET.tostring(title, encoding="unicode", method="text").strip()
            if title is not None
            else None
        )

    def extract_table_caption(self, table) -> Optional[str]:
        caption = table.find(".//caption")
        return (
            ET.tostring(caption, encoding="unicode", method="text").strip()
            if caption is not None
            else None
        )

    def extract_table_footnote(self, table) -> Optional[str]:
        footnote = table.find(".//table-wrap-foot")
        return (
            ET.tostring(footnote, encoding="unicode", method="text").strip()
            if footnote is not None
            else None
        )

    def collect_table_meta(self, table) -> Tuple[str, str, str]:
        title = self.extract_table_title(table)
        caption = self.extract_table_caption(table)
        footnote = self.extract_table_footnote(table)
        return title, caption, footnote

    def extract_raw_table_xml(self, table) -> Optional[str]:
        table_body = table.find(".//table")
        return (
            ET.tostring(table_body, encoding="unicode").strip()
            if table_body is not None
            else None
        )

    def get_max_columns(self, root: ET.Element) -> int:
        max_columns = 0
        thead = root.find(".//thead")
        if thead is not None:
            header_rows = thead.findall(".//tr")
            for header_row in header_rows:
                header_cells = [
                    unescape("".join(cell.itertext()).strip())
                    for cell in header_row.findall("td")
                ]
                max_columns = max(max_columns, len(header_cells))
        return max_columns

    def extract_headers(
        self, root: ET.Element, max_columns: int
    ) -> Tuple[List[str], List[str]]:
        headers = []
        subheaders = []

        thead = root.find(".//thead")
        if thead is not None:
            header_rows = thead.findall(".//tr")
            for i, header_row in enumerate(header_rows):
                header_cells = [
                    unescape("".join(cell.itertext()).strip())
                    for cell in header_row.findall("td")
                ]
                while len(header_cells) < max_columns:
                    header_cells.append("")
                if i == 0:
                    headers = header_cells
                else:
                    subheaders.append(header_cells)
        return headers, subheaders

    def extract_rows(self, root: ET.Element, max_columns: int) -> List[List[str]]:
        rows = []

        tbody = root.find(".//tbody")
        if tbody is not None:
            for row in tbody.findall("tr"):
                cols = [
                    unescape("".join(col.itertext()).strip())
                    for col in row.findall("td")
                ]
                while len(cols) < max_columns:
                    cols.append("")
                rows.append(cols)
        return rows

    def parse_table(self, raw_table_xml: str):
        root = ET.fromstring(raw_table_xml)

        max_columns = self.get_max_columns(root)
        table_headers, table_subheaders = self.extract_headers(root, max_columns)
        table_rows = self.extract_rows(root, max_columns)

        return table_headers, table_subheaders, table_rows

    def get_tables_from_xml(self, file_path: str):
        xml_content = self.read_xml_file(file_path)
        pmc_id = re.search(r"PMC\d+", file_path).group()
        tables = self.extract_tables_from_xml(xml_content)

        data = {
            "id": [],
            "table_title": [],
            "table_caption": [],
            "table_footnote": [],
            "table_xml_no_meta": [],
            "table_xml": [],
            "table_headers": [],
            "table_subheaders": [],
            "table_rows": [],
        }

        for table in tables:
            title, caption, footnote = self.collect_table_meta(table)
            raw_table = ET.tostring(table, encoding="unicode").strip()
            table_clean_xml = self.extract_raw_table_xml(table)
            table_headers, table_subheaders, table_rows = self.parse_table(
                table_clean_xml
            )
            data["id"].append(pmc_id)
            data["table_title"].append(title)
            data["table_caption"].append(caption)
            data["table_footnote"].append(footnote)
            data["table_xml_no_meta"].append(table_clean_xml)
            data["table_xml"].append(raw_table)
            data["table_headers"].append(table_headers)
            data["table_subheaders"].append(table_subheaders)
            data["table_rows"].append(table_rows)

        df = pd.DataFrame(data)
        return df

    def process_xml_files(self):
        all_tables = []

        for pmcid in self.pmc_ids:
            file_path = os.path.join(self.save_dir, f"{pmcid}.xml")
            df_tables = self.get_tables_from_xml(file_path)
            if not df_tables.empty:
                all_tables.append(df_tables)

        self.tables_df = pd.concat(all_tables, ignore_index=True)
        return self.tables_df
