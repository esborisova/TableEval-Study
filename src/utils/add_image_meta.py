import re
import roman
import os
import json
from typing import Tuple, Optional, List


class ImageMetadataExtractor:
    def __init__(self, data: dict, images_path: str, save_directory: str):
        self.data = data
        self.images_path = images_path
        self.save_directory = save_directory

    def extract_image_paper_id(self, image: str) -> Tuple[Optional[str], Optional[str]]:
        match = re.match(r"^([a-zA-Z0-9\.\-v]+)-(Table(\d+|[IVXLCDM]+))-\d+", image)
        if match:
            table_paper_id = match.group(1)
            img_table_name = match.group(2).replace(" ", "").lower()
            return table_paper_id, img_table_name
        else:
            return None, None

    def collect_image_meta(self, images: List[str]) -> dict:
        image_metadata = {}
        for image in images:
            table_paper_id, img_table_name = self.extract_image_paper_id(image)
            if table_paper_id and img_table_name:
                image_metadata[(table_paper_id, img_table_name)] = image
        return image_metadata

    def convert_table_name(self, table_name: str) -> str:
        """Converts a table name from Arabic to Roman numerals or vice versa."""
        match = re.match(r"(Table)\s?(\d+|[ivxlcdm]+)", table_name, re.IGNORECASE)

        if match:
            table_prefix = match.group(1).lower()
            table_number = match.group(2)

            if table_number.isdigit():
                roman_number = roman.toRoman(int(table_number)).lower()
                return f"{table_prefix}{roman_number}"
            else:
                arabic_number = roman.fromRoman(table_number.upper())
                return f"{table_prefix}{arabic_number}"
        else:
            return table_name

    def extract_table_name(self, caption: str) -> Tuple[Optional[str], Optional[str]]:
        """Handles both Arabic and Roman numbers, returning both forms."""
        caption = caption.lower()
        match = re.search(r"(table)\s?(\d+|[ivxlcdm]+)", caption)

        if match:
            table_prefix = match.group(1).replace(" ", "")
            table_number = match.group(2)

            if table_number.isdigit():
                roman_number = roman.toRoman(int(table_number))
                return (
                    f"{table_prefix}{table_number}",
                    f"{table_prefix}{roman_number.lower()}",
                )
            else:
                arabic_number = roman.fromRoman(table_number.upper())
                return f"{table_prefix}{table_number}", f"{table_prefix}{arabic_number}"
        else:
            return None, None

    def add_image_id_numericnlg(self, example: dict, image_metadata: dict) -> dict:
        table_name = example.get("table_name", "").replace(" ", "").lower()
        transformed_table_name = self.convert_table_name(table_name)
        paper_id = example.get("paper_id", "")
        image_id = image_metadata.get(
            (paper_id, table_name), None
        ) or image_metadata.get((paper_id, transformed_table_name), None)

        example["image_id"] = image_id
        return example

    def add_image_id_to_scigen(self, image_metadata: dict) -> dict:
        for key, item in self.data.items():
            caption = item.get("table_caption", "")
            original_table_name, transfromed_table_name = self.extract_table_name(
                caption
            )
            paper_id = item.get("paper_id", "")
            if paper_id:
                image_id = image_metadata.get(
                    (paper_id, original_table_name), None
                ) or image_metadata.get((paper_id, transfromed_table_name), None)
                item["image_id"] = image_id
            else:
                item["image_id"] = None
        return self.data

    def process_numericnlg(self):
        image_names = [
            image
            for image in os.listdir(self.images_path)
            if ("table" in image.lower()) and (".DS_Store" not in image)
        ]
        image_metadata = self.collect_image_meta(image_names)
        ds = self.data.map(
            lambda example: self.add_image_id_numericnlg(example, image_metadata)
        )
        ds.save_to_disk(self.save_directory)

    def process_scigen(self):
        image_names = [
            image
            for image in os.listdir(self.images_path)
            if ("table" in image.lower()) and (".DS_Store" not in image)
        ]
        image_metadata = self.collect_image_meta(image_names)
        updated_json = self.add_image_id_to_scigen(image_metadata)
        with open(self.save_directory, "w") as file:
            json.dump(updated_json, file, indent=4)
