"Script for extracting table HTML and XML from paper HTML/XML files generated with latexML tool."
from datasets import load_from_disk
import pandas as pd
from ..utils.other import create_and_save_dataset
from ..utils.xml_html_convertion import find_best_match
from ..utils.other import read_json


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

    img_ids = read_json("imgs_to_exclude_scigen.json")
    formats = ["html", "xml"]

    for data_path, save_path in zip(data_paths, save_paths):
        scigen_data = load_from_disk(data_path).to_pandas()

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

            img_ids_html = img_ids["img_ids_other_html"]
            img_ids_xml = img_ids["img_ids_other_xml"]
        else:
            img_ids_html = img_ids["img_ids_cl_html"]
            img_ids_xml = img_ids["img_ids_cl_xml"]

        arxiv_subset = scigen_data[scigen_data["venue"] == "arxiv"]

        for format in formats:
            combined_results = pd.DataFrame()

            for tex_path in tex_paths:
                best_matches_df = find_best_match(arxiv_subset, tex_path, format)

                best_matches_df.loc[
                    best_matches_df["fuzzy_score"] < 80,
                    [f"table_{format}", "target_caption"],
                ] = None

                img_ids = img_ids_html if format == "html" else img_ids_xml

                best_matches_df.loc[
                    best_matches_df["image_id"].isin(img_ids),
                    [f"table_{format}", "target_caption"],
                ] = None

                best_matches_df = best_matches_df.rename(
                    columns={
                        "gold_caption": "table_caption",
                        "target_caption": f"table_{format}_caption",
                    }
                )

                combined_results = pd.concat(
                    [combined_results, best_matches_df], axis=0
                )

                scigen_data = scigen_data.drop(
                    columns=[
                        "table_html",
                        "table_xml",
                        "table_html_source",
                        "table_xml_source",
                        "__index_level_0__",
                    ]
                )

                merged_df = scigen_data.merge(
                    combined_results[
                        [
                            "paper_id",
                            "image_id",
                            "table_caption",
                            "text",
                            "table_html",
                            "table_html_caption",
                            "table_xml",
                            "table_xml_caption",
                        ]
                    ],
                    on=["paper_id", "image_id", "table_caption", "text"],
                    how="left",
                )

                create_and_save_dataset(merged_df, "test", save_path)


if __name__ == "__main__":
    main()
