"""Script for adding license and venue metadata to SciGen and numericNLG datasets."""
from datasets import load_from_disk
import json
from ..utils.prepare_data import load_scigen_dataset
from ..utils.license_venue_meta import (
    add_venue_license_numericnlg,
    add_license_venue_scigen,
)


def main():
    file_names = ["numericnlg", "scigen"]
    numeric_nlg_path = "../../data/numericNLG/data_with_imgs_meta"
    scigen_jsons = [
        "../../data/SciGen/test-CL/test_CL_with_imgs_meta_updated_2024-09-04.json",
        "../../data/SciGen/test-Other/test_Other_with_imgs_meta.json",
    ]

    scigen_cl_save_dir = (
        "../../data/SciGen/test-CL/test_CL_with_imgs_license_venue.json"
    )
    scigen_other_save_dir = (
        "../../data/SciGen/test-Other/test_Other_with_imgs_license_venue.json",
    )

    for file_name in file_names:
        if "numericnlg" in file_name:
            data = load_from_disk(numeric_nlg_path)
            updated_data = data.map(add_venue_license_numericnlg)
            updated_data.save_to_disk(
                "../../data/numericNLG/data_with_imgs_license_venue"
            )
        else:
            for json_file in scigen_jsons:
                data = load_scigen_dataset(json_file)
                updated_data = add_license_venue_scigen(data)
                if "CL" in json_file:
                    save_dir = scigen_cl_save_dir
                else:
                    save_dir = scigen_other_save_dir
                with open(save_dir, "w") as file:
                    json.dump(updated_data, file, indent=4)


if __name__ == "__main__":
    main()
