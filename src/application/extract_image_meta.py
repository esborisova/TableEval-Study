"""Pipeline for collecting images metadata (id, name, etc.) in SciGen and numericNLC datasets."""
import datasets
from datasets import load_dataset
from ..utils.prepare_data import load_scigen_dataset
from ..utils.add_image_meta import ImageMetadataExtractor


def main():
    file_names = ["numericnlg", "scigen"]

    numeric_nlg_imgs = "../../data/numericNLG/generated_imgs/"
    save_numeric_nlg = "../../data/numericNLG/"

    scigen_imgs = [
        "../../data/SciGen/test-CL/generated_imgs_cl/",
        "../../data/SciGen/test-Other/generated_imgs_other/",
    ]
    scigen_jsons = [
        "../../data/SciGen/test-CL/updated_test-CL.json",
        "../../data/SciGen/test-Other/updated_test-Other.json",
    ]
    save_scigen_cl = "../../data/SciGen/test-CL/test_CL_with_imgs_meta.json"
    save_scigen_other = "../../data/SciGen/test-Other/test_Other_with_imgs_meta.json"

    for file_name in file_names:
        if "numericnlg" in file_name:
            ds = load_dataset("kasnerz/numericnlg", split="test")
            extractor = ImageMetadataExtractor(ds, numeric_nlg_imgs, save_numeric_nlg)
            extractor.process_numericnlg()

        else:
            for images_folder, json_file in zip(scigen_imgs, scigen_jsons):
                data = load_scigen_dataset(json_file)
                if "CL" in json_file:
                    save_dir = save_scigen_cl
                else:
                    save_dir = save_scigen_other
                extractor = ImageMetadataExtractor(data, images_folder, save_dir)
                extractor.process_scigen()


if __name__ == "__main__":
    main()
