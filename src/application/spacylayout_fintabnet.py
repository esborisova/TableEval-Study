"""Pipeline for extracting tables from fintabnet pdfs using spaCy layout."""
import pandas as pd
import spacy
import os
from spacy_layout import spaCyLayout
from datasets import load_from_disk
import pickle
from ..utils.other import display_table


def main():
    fintabnet_annot_path = (
        "../../data/ComTQA_data/fintabnet/source_data/FinTabNet_1.0.0_cell_test.jsonl"
    )
    fintabnet_annotations = pd.read_json(fintabnet_annot_path, lines=True)

    pdf_files_path = "../../data/ComTQA_data/fintabnet/pdfs/"
    save_dir = "../../data/ComTQA_data/fintabnet/spacylayout_output.pkl"

    ds = load_from_disk(f"../../data/ComTQA_data/comtqa_updated_2024-11-27")
    fintab_subset = ds.filter(lambda x: x["dataset"] == "FinTabNet")
    table_ids = set(fintab_subset["train"]["table_id"])

    nlp = spacy.blank("en")
    layout = spaCyLayout(nlp, display_table=display_table)

    table_dict = []
    for _, row in fintabnet_annotations.iterrows():
        table_id = str(row["table_id"])
        if table_id in table_ids:
            pdf_path = os.path.join(pdf_files_path, row["filename"])
            doc = layout(pdf_path)

            table_entry = {"table_id": table_id, "pdf_path": pdf_path}

            for idx, table in enumerate(doc._.tables):
                table_entry[f"table_df_{idx + 1}"] = table._.get(layout.attrs.span_data)

            table_dict.append(table_entry)

    with open(save_dir, "wb") as f:
        pickle.dump(table_dict, f)


if __name__ == "__main__":
    main()
