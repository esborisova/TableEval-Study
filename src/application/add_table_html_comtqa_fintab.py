import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from datetime import datetime


def main():
    fintabnet_annot_path = "../../data/fintabnet/FinTabNet_1.0.0_cell_test.jsonl"
    fintabnet_annotations = pd.read_json(path_or_buf=fintabnet_annot_path, lines=True)

    ds = load_from_disk("../../data/comtqa_updated_2024_10_24")
    fintab_subset = ds.filter(lambda x: x["dataset"] == "FinTabNet")
    table_ids = set(fintab_subset["train"]["table_id"])

    table_html = []
    for _, row in fintabnet_annotations.iterrows():
        table_id = str(row["table_id"])
        if table_id in table_ids:
            html = ""
            cnt = 0
            for token in row["html"]["structure"]["tokens"]:
                html += token
                if token == "<td>":
                    html += "".join(row["html"]["cells"][cnt]["tokens"])
                    cnt += 1

            table_html.append({"table_id": table_id, "html": html})

    table_html_df = pd.DataFrame(table_html)
    ds_df = ds["train"].to_pandas()
    ds_df = ds_df.merge(table_html_df[["table_id", "html"]], on="table_id", how="left")
    ds_df.rename(columns={"html": "table_html"}, inplace=True)

    ds_df = ds_df.reset_index(drop=True)
    hf_dataset = Dataset.from_pandas(ds_df.reset_index(drop=True))
    hf_dataset_dict = DatasetDict({"train": hf_dataset})
    date = datetime.now().strftime("%Y-%m-%d")
    hf_dataset_dict.save_to_disk(f"../../data/comtqa_updated_{date}")


if __name__ == "__main__":
    main()
