import pandas as pd
import xml.etree.ElementTree as ET
from html import unescape
from typing import Tuple, List

def get_max_columns(root: ET.Element) -> int:
    "Computes the max number of columns across all header rows"
    max_columns = 0
    thead = root.find(".//thead")
    if thead is not None:
        header_rows = thead.findall(".//tr")
        for header_row in header_rows:
            header_cells = [
                unescape("".join(cell.itertext()).strip()) for cell in header_row.findall("td")
            ]
            max_columns = max(max_columns, len(header_cells))
    return max_columns

def extract_headers(root: ET.Element, max_columns: int) -> Tuple[List[str], List[str]]:
    "Extracts headers and subheaders"
    headers = []
    subheaders = []
    
    thead = root.find(".//thead")
    if thead is not None:
        header_rows = thead.findall(".//tr")
        
        for i, header_row in enumerate(header_rows):
            header_cells = [
                unescape("".join(cell.itertext()).strip()) for cell in header_row.findall("td")
            ]
            
            # Fill in empty cells with empty str 
            while len(header_cells) < max_columns:
                header_cells.append("")
            if i == 0:
                headers = header_cells  
            else:
                subheaders.append(header_cells)  
    return headers, subheaders
def extract_rows(root: ET.Element, max_columns: int) -> List[str]:
    rows = []
    
    tbody = root.find(".//tbody")
    if tbody is not None:
        for row in tbody.findall("tr"):
            cols = [unescape("".join(col.itertext()).strip()) for col in row.findall("td")]
            # Fill in empty cells with empty str 
            while len(cols) < max_columns:
                cols.append("")
            rows.append(cols)
    return rows
def parse_table(xml_content: str) -> dict:
    "Extracts headers, subheaders, and rows from table xml"
    root = ET.fromstring(xml_content)
    
    table_data = {
        "table_headers": [],
        "table_subheaders": [],
        "table_rows": []
    }
    
    max_columns = get_max_columns(root)
    table_data["table_headers"], table_data["table_subheaders"] = extract_headers(root, max_columns)
    table_data["table_rows"] = extract_rows(root, max_columns)
    
    return table_data


def parsed_tab_xml_df(row: pd.Series, condition_col: str, condition_value: str, xml_col: str) -> pd.Series :
    """
    Parses table XML and saves into a series. 

    Args:
        row (pd.Series): A single row of the DataFrame being processed.
        condition_col (str): The name of the column in the DataFrame to check for the condition. E.g., "dataset".
        condition_value (str): The value in the `condition_col` that will trigger XML processing. E.g., if "dataset" == "PubTab1M".
        xml_col (str): The name of the column with table XML to be parsed.

    Returns:
        pd.Series: A Series containing the parsed table data.
    """
    if row[condition_col] == condition_value:
        table_data = parse_table(row[xml_col])
        return pd.Series({
            "table_headers": table_data["table_headers"],
            "table_subheaders": table_data["table_subheaders"],
            "table_rows": table_data["table_rows"]
        })
    else:
        return pd.Series({
            "table_headers": None,
            "table_subheaders": None,
            "table_rows": None
        })