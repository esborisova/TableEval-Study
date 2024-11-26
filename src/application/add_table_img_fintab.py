import os
from PyPDF2 import PdfReader
import copy
from pdf2image import convert_from_path
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from datetime import datetime
from PIL import ImageDraw
import copy


def main():
    fintabnet_annot_path = "../../../data/fintabnet/FinTabNet_1.0.0_cell_test.jsonl"
    fintabnet_annotations = pd.read_json(fintabnet_annot_path, lines=True)
    pdf_files_path = "../../../data/fintabnet/pdfs/"

    ds = load_from_disk("../../../data/comtqa_updated_2024_10_24")
    fintab_subset = ds.filter(lambda x: x["dataset"] == "FinTabNet")
    table_ids = set(fintab_subset["train"]["table_id"])

    save_dir = "../../../data/fintabnet/images"
    save_dir_img_with_bbox = "../../../data/fintabnet/bbox_annotations"

    images_meta = []
    for _, row in fintabnet_annotations.iterrows():
        table_id = str(row["table_id"])
        if table_id in table_ids:
            pdf_path = os.path.join(pdf_files_path, row["filename"])
            try:
                pdf_page = PdfReader(open(pdf_path, "rb")).pages[0]
                if pdf_page:
                    pdf_shape = pdf_page.mediabox
                    pdf_height = pdf_shape[3] - pdf_shape[1]
                    pdf_width = pdf_shape[2] - pdf_shape[0]

                    converted_images = convert_from_path(
                        pdf_path, dpi=700, size=(pdf_width, pdf_height)
                    )
                    img_pdf = converted_images[0]

                    orig_annotation = copy.copy(row["bbox"])
                    row["bbox"][3] = float(pdf_height) - float(orig_annotation[1])
                    row["bbox"][1] = float(pdf_height) - float(orig_annotation[3])

                    img_table = img_pdf.crop(
                        (row["bbox"][0], row["bbox"][1], row["bbox"][2], row["bbox"][3])
                    )

                    draw = ImageDraw.Draw(img_pdf)
                    bbox = row["bbox"]
                    draw.rectangle(
                        [bbox[0], bbox[1], bbox[2], bbox[3]], outline="red", width=5
                    )

                    img_file_name = f"{row['filename'].replace('/', '_').split('.pdf')[0]}_{str(row['table_id'])}.png"
                    image_save_dir = os.path.join(save_dir, img_file_name)
                    bbox_annot_save_dir = os.path.join(
                        save_dir_img_with_bbox, img_file_name
                    )
                    img_table.save(image_save_dir, "PNG")
                    img_pdf.save(bbox_annot_save_dir, "PNG")
                    images_meta.append(
                        {
                            "table_id": table_id,
                            "image_name": img_file_name,
                        }
                    )

            except Exception as e:
                print(e)

    df_images_meta = pd.DataFrame(images_meta)
    ds_df = ds["train"].to_pandas()

    merged_df = pd.merge(
        ds_df,
        df_images_meta[["table_id", "image_name"]],
        on="table_id",
        how="left",
        suffixes=("_df2", "_df1"),
    )
    merged_df["image_name"] = merged_df["image_name_df1"].combine_first(
        merged_df["image_name_df2"]
    )
    merged_df = merged_df.drop(columns=["image_name_df2", "image_name_df1"])

    hf_dataset = Dataset.from_pandas(merged_df.reset_index(drop=True))
    hf_dataset_dict = DatasetDict({"train": hf_dataset})
    date = datetime.now().strftime("%Y-%m-%d")
    hf_dataset_dict.save_to_disk(f"../../../data/comtqa_updated_{date}")


if __name__ == "__main__":
    main()
