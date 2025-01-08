"Script for extracting table HTML from paper HTML files generated with latexML tool."
from datasets import load_from_disk
import os
import pandas as pd
from bs4 import BeautifulSoup
from thefuzz import fuzz
from ..utils.other import create_and_save_dataset
from typing import List


def find_file(root_folder: str, paper_id, file_type: str) -> List[str]:
    paper_folder = os.path.join(root_folder, paper_id)
    found_files = []
    if os.path.exists(paper_folder) and os.path.isdir(paper_folder):
        for root, _, files in os.walk(paper_folder):
            for file in files:
                if file_type == "html":
                    pattern = ".html"
                else:
                    pattern = ".xml"
                if file.lower().endswith(pattern):
                    found_files.append(file)
    return found_files


def read_html(file):
    with open(os.path.join(file), "r") as f:
        content = f.read()
    return content


def preprocess_target_caption(caption):
    target_caption = caption.get_text(strip=True)
    return target_caption.lower()


def find_best_match(df: pd.DataFrame, files_dir: str, format="html") -> pd.DataFrame:
    """
    Finds the best table caption match for each entry in a DataFrame.

    Args:
        dataframe (pd.DataFrame): DataFrame containing 'table_caption' and 'paper_id'.
        files_dir (str): Path to the folder containing HTML or XML files.
        file_type (str): Type of files to process, html or xml (default is 'html').
    Returns:
        pd.DataFrame: DataFrame containing best match details for each row.
    """

    best_matches = []

    for _, row in df.iterrows():
        gold_caption = row["table_caption"].lower()
        paper_id = row["paper_id"]
        highest_score = 0

        best_table = {
            "paper_id": paper_id,
            "image_id": row["image_id"],
            "text": row["text"],
            "gold_caption": row["table_caption"],
            "target_caption": None,
            "table_html": None,
            "fuzzy_score": None,
        }

        html_files = find_file(files_dir, paper_id, format)

        for file in html_files:
            folder_dir = os.path.join(files_dir, paper_id)
            html_content = read_html(os.path.join(folder_dir, file))
            soup = BeautifulSoup(html_content, "lxml")

            captions = soup.find_all("figcaption")
            for caption in captions:
                target_caption = caption
                target_caption_pre = preprocess_target_caption(target_caption)

                fuzzy_match_score = fuzz.ratio(gold_caption, target_caption_pre)

                if fuzzy_match_score > highest_score:
                    highest_score = fuzzy_match_score
                    best_table["target_caption"] = (
                        target_caption.get_text(strip=True),
                    )
                    best_table["table_html"] = str(
                        caption.find_parent("figure").find("table")
                    )
                    best_table["fuzzy_score"] = fuzzy_match_score

        best_matches.append(best_table)
    return pd.DataFrame(best_matches)


def main():

    data_paths = [
        "../../data/SciGen/test-CL/test_CL_all_meta_all_formats_2024_11_25.hf",
        "../../data/SciGen/test-Other/test_Other_all_meta_all_formats_2024_11_25.hf",
    ]
    save_paths = [
        "../../data/SciGen/test-CL/test_CL_updated",
        "../../data/SciGen/test-Other/test_Other_updated",
    ]

    tex_paths = [
        "../../data/SciGen/test-CL/latex_source_code/latexml/",
        "../../data/SciGen/test-Other/latex_source_code/latexml/",
    ]

    # ids of instances to assign none to
    img_ids_cl = [
        "1710.10380v3-Table4-1.png",
        "1904.02338v2-Table5-1.png",
        "1809.05157v2-Table1-1.png",
        "1908.08528v1-Table3-1.png",
    ]

    img_ids_other = [
        "1909.05379v2-Table3-1.png",
        "1909.05379v2-Table4-1.png",
        "2002.02618v1-Table7-1.png",
        "1812.06707v1-Table2-1.png",
        "1912.03820v3-Table4-1.png",
        "2003.06729v1-Table2-1.png",
        "1810.00319v6-Table1-1.png",
        "1810.00319v6-Table4-1.png",
        "1810.00319v6-Table5-1.png",
        "1912.10080v1-TableIII-1.png",
        "1809.08410v1-TableI-1.png",
        "1909.05125v1-Table1-1.png",
        "1912.04216v1-Table1-1.png",
    ]

    for data_path, save_path in zip(data_paths, save_paths):
        scigen_data = load_from_disk(data_path)
        scigen_data = scigen_data.to_pandas()

        if "test-Other" in data_path:
            scigen_data.loc[
                scigen_data["image_id"] == "1907.11281v1-Table2-1.png", "image_id"
            ] = "1907.11281v1-Table1-1.png"

            scigen_data.loc[
                scigen_data["image_id"] == "1907.11281v1-Table3-1.png", "image_id"
            ] = "1907.11281v1-Table2-1.png"
            scigen_data = scigen_data.drop_duplicates(
                subset=["paper_id", "table_caption", "image_id", "text"]
            )
            img_ids = img_ids_other
        else:
            img_ids = img_ids_cl

        arxiv_subset = scigen_data[scigen_data["venue"] == "arxiv"]

        for tex_path in tex_paths:
            best_matches_df = find_best_match(arxiv_subset, tex_path, "html")
            best_matches_df.loc[
                best_matches_df["fuzzy_score"] < 80, ["table_html", "target_caption"]
            ] = None
            best_matches_df.loc[
                best_matches_df["image_id"].isin(img_ids),
                ["table_html", "target_caption"],
            ] = None
            best_matches_df = best_matches_df.rename(
                columns={
                    "gold_caption": "table_caption",
                    "target_caption": "table_html_caption",
                }
            )
            scigen_data = scigen_data.drop(
                columns=["table_html", "table_html_source", "__index_level_0__"]
            )

            merged_df = scigen_data.merge(
                best_matches_df[
                    [
                        "paper_id",
                        "image_id",
                        "table_caption",
                        "table_html",
                        "table_html_caption",
                        "text",
                    ]
                ],
                on=["paper_id", "image_id", "table_caption", "text"],
                how="left",
            )

            create_and_save_dataset(merged_df, "test", save_path)


if __name__ == "__main__":
    main()
