"""Remove HTML indicators, validate and prettify HTML in numericNLG"""

from datasets import load_from_disk
import pandas as pd
from ..utils.xml_html_convertion import change_table_class, validate_html, prettify_html
from ..utils.other import create_and_save_dataset


def main():
    numericnlg = load_from_disk("../../data/numericNLG/numericnlg_updated_2025-12-27")
    numericnlg = numericnlg["test"].to_pandas()
    numericnlg["table_html_clean"] = numericnlg["table_html"].apply(change_table_class)
    list_of_html = numericnlg["table_html_clean"].tolist()
    pretty_html = [prettify_html(html) for html in list_of_html]
    numericnlg["table_html_clean"] = pretty_html
    validated_html = validate_html(
        numericnlg, "table_html_clean", "table_id_paper", "numericnlg_html_val"
    )
    create_and_save_dataset(
        numericnlg, "test", "../../data/numericNLG/numericnlg_updated"
    )


if __name__ == "__main__":
    main()
