"""Collecting images from pubtables1m used on ComTQA dataset"""
from datasets import load_dataset
import tarfile
import os


def copy_files(tar_dir, output_dir, image_files):
    with tarfile.open(tar_dir, "r:gz") as tar:
        os.makedirs(output_dir, exist_ok=True)
        tar.extractall(path=output_dir, members=image_files)


def main():
    tar_path = "/netscratch/borisova/eval_study/data/pubmed/PubTables-1M-Structure_Images_Train.tar.gz"
    ds = load_dataset("ByteDance/ComTQA")
    pubtab1m_data = ds.filter(lambda x: x["dataset"] == "PubTab1M")
    pubtab1m_imgs = [item["image_name"] for item in pubtab1m_data["train"]]
    unique_pubtab1m_imgs = list(set(pubtab1m_imgs))
    save_images_dir = "/netscratch/borisova/eval_study/data/pubmed/images/"
    copy_files(tar_path, save_images_dir, unique_pubtab1m_imgs)


if __name__ == "__main__":
    main()
