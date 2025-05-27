"""Script for collecting and saving into csv scores per dataset and model"""

import pandas as pd
import json
import os
import re
from ..utils import read_json


def main():
    scores_rootdir = ""

    list_of_dfs = []
    for file in os.listdir(scores_rootdir):
        if not file.startswith(".DS_Store"):
            file_path = os.path.join(scores_rootdir, file)

            match = re.match(
                r"scores_([a-zA-Z]+)_(\w+)_([\w\.\-]+)_(\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2})\.json",
                file,
            )
            if match:
                format = match.group(1)
                dataset_name = match.group(2)
                model_name = match.group(3)
            else:
                format, dataset_name, model_name = None, None, None

            data = read_json(file_path)
            df = pd.json_normalize(data)

            df["format"] = format
            df["dataset"] = dataset_name
            df["model"] = model_name

            column_order = ["format", "dataset", "model"] + [
                col for col in df.columns if col not in ["format", "dataset", "model"]
            ]
            df = df[column_order]
            list_of_dfs.append(df)

    scores_df = pd.concat(list_of_dfs, ignore_index=True)
    scores_df = scores_df.sort_values(by=["format", "dataset"], ascending=[True, True])
    scores_df = scores_df.reset_index(drop=True)
    save_path = os.path.join(scores_rootdir, "scores.csv")
    scores_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
