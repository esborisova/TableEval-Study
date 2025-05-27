from TexSoup import TexSoup
from bs4 import BeautifulSoup, Comment
from datasets import Dataset, load_from_disk, save_to_disk
import re
import argparse
from ..utils.other import create_and_save_dataset
from ..utils.xml_html_convertion import prettify_html, validate_html


def clean_latexml_html(html_content):
    """Automatically cleans LaTeXML-generated HTML tables by removing unnecessary tags while keeping content."""
    soup = BeautifulSoup(html_content, "html.parser")

    allowed_tags = {"table", "tbody", "tr", "th", "td"}

    for tag in soup.find_all():
        if tag.name not in allowed_tags:
            tag.unwrap()

    for tag in soup.find_all(allowed_tags):
        tag.attrs = {
            k: v for k, v in tag.attrs.items() if k in {"colspan", "rowspan", "align"}
        }

    return str(soup.prettify())


def clean_latexml_xml(xml_content):

    """Automatically cleans LaTeXML-generated XML tables by removing unnecessary tags while keeping captions and titles."""

    soup = BeautifulSoup(xml_content, "xml")

    # extract content inside the first <tag>
    tag = soup.find('tag')
    if tag:
        tag_content = tag.string

    #remove <tag> including its content
    for tag in soup.find_all('tag'):
      tag.decompose()

    allowed_tags = {"tabular", "table", "thead", "tfoot", "tbody", "tr", "th", "td", "caption", "title"}

    # remove toccaption tag, which includes the same as the caption tag
    toccaption = soup.find("toccaption")
    if toccaption:
        toccaption.decompose()

    caption = soup.find("caption")

    if caption:
        caption_text = " ".join(
            [tag.string for tag in caption.find_all(text=True)]
        )  # Extract all text content inside <caption>
        caption.clear()
        # add the tag_conent inside <caption>
        caption.append(tag_content)
        caption.append(caption_text)

    for tag in soup.find_all():
        if tag.name not in allowed_tags:
            tag.unwrap()

    for tag in soup.find_all(allowed_tags):
        tag.attrs = {
            k: v for k, v in tag.attrs.items() if k in {"colspan", "rowspan", "align", "thead"}
        }

    # remove comments <!-- -->
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()

    # remove XML declaration if it exists
    cleaned_xml = "\n".join(
        line
        for line in str(soup.prettify()).split("\n")
        if not line.startswith("<?xml")
    )

    return cleaned_xml

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path for the SciGen dataset")
    parser.add_argument(
        "--output_path", type=str, help="Path for the saving the resulting dataset"
    )
    args = parser.parse_args()

    data_path = args.data_path
    output_path = args.output_path

    scigen_data = load_from_disk(data_path)

    cleaned_html_tables = []
    cleaned_xml_tables = []

    for item in scigen_data:
        if item["table_html"] != None:
            cleaned_html_tables.append(clean_latexml_html(item["table_html"]))
        else:
            cleaned_html_tables.append(None)

        if item["table_xml"] != None:
            cleaned_xml_tables.append(clean_latexml_xml(item["table_xml"]))
        else:
            cleaned_xml_tables.append(None)

    scigen_data = scigen_data.to_pandas()

    cleaned_html_tables = [
        prettify_html(html) if html is not None else html
        for html in cleaned_html_tables
    ]

    scigen_data["table_html_cleaned"] = cleaned_html_tables
    scigen_data["table_xml_cleaned"] = cleaned_xml_tables

    validated_html = validate_html(
        scigen_data, "table_html_cleaned", "image_id", "scigen_html_val"
    )

    # Save as HF Dataset
    create_and_save_dataset(scigen_data, "test", output_path)


if __name__ == "__main__":
    main()
