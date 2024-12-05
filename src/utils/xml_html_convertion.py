from bs4 import BeautifulSoup
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

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
        }
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



def generate_html_content(soup):
    return f"<html><head><meta charset='UTF-8'></head><body>{soup.prettify()}</body></html>"


def pmc_tables_to_html(xml_table: str) -> str:
    soup = BeautifulSoup(xml_table, "xml")
    soup = map_xml_html_tags(soup, convertion_type="xml_to_html")
    html_table = generate_html_content(soup)
    return html_table

def create_pmc_table_wrap(table_id: str = None, 
                          label_text: str = None, 
                          caption_text: str = None):
    table_wrap = Element("table-wrap", position="float")
    if table_id:
        table_wrap.set("id", table_id)
    if label_text:
        SubElement(table_wrap, "label").text = label_text
    if caption_text:
        caption = SubElement(table_wrap, "caption")
        SubElement(caption, "p").text = caption_text
    return table_wrap


def process_html_table_row(row, 
                           parent_elem, 
                           row_type="td"):
    
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


def html_to_xml_table(html_table: str, 
                      table_id: str =None, 
                      label_text: str =None, 
                      caption_text: str =None) -> str:
    
    soup = BeautifulSoup(html_table, "html.parser")

    soup = map_xml_html_tags(soup, conversion_type="html_to_xml")
    table_wrap = create_pmc_table_wrap(table_id, label_text, caption_text)
    html_table_elem = soup.find("table")
    if html_table_elem:
        create_xml_table(table_wrap, html_table_elem)

    rough_string = tostring(table_wrap, encoding="unicode", method="xml")
    pretty_xml = minidom.parseString(rough_string).toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")

    return pretty_xml.split('\n', 1)[1] if pretty_xml.startswith('<?xml') else pretty_xml
