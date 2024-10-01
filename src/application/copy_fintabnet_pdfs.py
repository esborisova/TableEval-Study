import pandas as pd
import shutil
from datasets import load_dataset
import os

def main():
    fintabnet_doc_path = "/netscratch/borisova/eval_study/data/fintab/fintabnet/FinTabNet_1.0.0_cell_test.jsonl"
    pdfs_dest_dir = "/netscratch/borisova/eval_study/data/fintab/fintabnet/comtqa_pdfs/"
    pdfs_rootdir = "/netscratch/borisova/eval_study/data/fintab/fintabnet/pdf/"

    ds = load_dataset("ByteDance/ComTQA")
    fintab_subset = ds.filter(lambda x: x["dataset"] == "FinTabNet")
    table_ids = list(set(fintab_subset["train"]["table_id"]))

    fintabnet_annotations = pd.read_json(path_or_buf=fintabnet_doc_path, lines=True)

    for index, _ in fintabnet_annotations.iterrows():
        table_id = str(fintabnet_annotations["table_id"][index])
        if table_id in table_ids:
            filename = fintabnet_annotations["filename"][index]
            source_file = os.path.join(
                pdfs_rootdir, filename
            )
            if os.path.exists(source_file):
                relative_path = fintabnet_annotations["filename"][index]
                dest_file_path = os.path.join(pdfs_dest_dir, relative_path)
                os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
                shutil.copy2(source_file, dest_file_path)
                
if __name__ == "__main__":
    main()

