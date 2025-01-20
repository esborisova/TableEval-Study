import pandas as pd
from datasets import Dataset, load_from_disk

dataset_name = "logic2text"  # "logic2text" / "logicnlg"
dataset = load_from_disk(f"../../data/{dataset_name}").to_pandas()

if dataset_name == "logicnlg":
    anno_one = pd.read_csv("../../data/logicnlg_annos_part_1.csv")
    anno_two = pd.read_csv("../../data/logicnlg_annos_part_2.csv")
    anno = pd.concat([anno_one, anno_two])
elif dataset_name == "logic2text":
    anno = pd.read_csv("../../data/logic2text_annos.csv")
else:
    raise ValueError(f"Dataset {dataset_name} not found.")

anno["table_id"] = anno["filename"]
merged_df = dataset.merge(anno, how="left", on="table_id")

annotated_df = merged_df[merged_df["usable"].notna()]
final_annotated_df = annotated_df[annotated_df["usable"] == True]

final_annotated_dataset = Dataset.from_pandas(final_annotated_df)

final_annotated_dataset.save_to_disk(f"../../data/{dataset_name}_filtered")

print("Finished dataset.")
