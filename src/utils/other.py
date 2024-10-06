import json


def read_json(file_path: str) -> list:
    with open(file_path) as f:
        return json.load(f)
