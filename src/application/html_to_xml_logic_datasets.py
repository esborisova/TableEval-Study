"""Script for generating XML, filtering not fully matched tables and 
removing table type indication in HTML for LogicNLG and Logic2Text data."""

from datasets import load_from_disk
import pandas as pd
from ..utils.other import create_and_save_dataset
from ..utils.xml_html_convertion import (
    change_table_class,
    html_to_xml_table,
    prettify_xml,
)


def main():
    data_paths = [
        "../../data/LogicNLG/logicnlg_updated_2025-01-22",
        "../../data/Logic2Text/logic2text_updated_2025-01-22",
    ]

    for path in data_paths:
        data = load_from_disk(path)
        data = data["test"].to_pandas()

        data["image_name"] = data["filename"].apply(lambda x: x + ".png")
        data["table_html"] = data["matched_table_html"].apply(change_table_class)
        data["table_xml"] = data["table_html"].apply(html_to_xml_table)
        data["table_xml"] = data["table_xml"].apply(prettify_xml)

        if "LogicNLG" in path:
            save_path = "../../data/LogicNLG/logicnlg_filtered_updated"
        else:
            save_path = "../../data/Logic2Text/logic2text_filtered_updated"

        # keep only fully matched tables
        data = data[data["issue"] == "none"]
        create_and_save_dataset(data, "test", save_path)


if __name__ == "__main__":
    main()
