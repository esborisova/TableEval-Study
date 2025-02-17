"Script for converting SciGen source tex files into HTML and XML with latexML tool"
from datasets import load_from_disk
import os
import subprocess
from ..utils.other import create_dir


def find_main_tex_by_document(root_folder: str, paper_id):
    paper_folder = os.path.join(root_folder, paper_id)

    tex_files = []
    if os.path.exists(paper_folder) and os.path.isdir(paper_folder):
        for root, _, files in os.walk(paper_folder):
            for file in files:
                if file.endswith(".tex"):
                    with open(os.path.join(root, file), "r") as f:
                        content = f.read()
                        if "\\begin{document}" in content:
                            tex_files.append(os.path.join(root, file))

    tex_files = list(set(tex_files))
    return tex_files


def convert_tex_to_html_xml(tex_file: str, output_dir: str):
    """Converts a .tex file to HTML and XML using LaTeXML."""
    create_dir(output_dir)

    xml_output = os.path.join(
        output_dir, os.path.basename(tex_file).replace(".tex", ".xml")
    )
    html_output = os.path.join(
        output_dir, os.path.basename(tex_file).replace(".tex", ".html")
    )
    try:
        subprocess.run(["latexml", "--dest", xml_output, tex_file], check=True)
    except Exception as e:
        print("Could not convert to xml. Error:", e)
    try:
        subprocess.run(["latexmlpost", "--dest", html_output, xml_output], check=True)
    except Exception as e:
        print("Could not convert to html. Error:", e)


def main():
    data_paths = [
        "../../data/SciGen/test-CL/test_CL_all_meta_all_formats_2024_11_25.hf",
        "../../data/SciGen/test-Other/test_Other_all_meta_all_formats_2024_11_25.hf",
    ]
    save_paths = [
        "../../data/SciGen/test-CL/latex_source_code/latexml/",
        "../../data/SciGen/test-Other/latex_source_code/latexml/",
    ]

    tex_paths = [
        "../../data/SciGen/test-CL/latex_source_code/uzipped/"
        "../../data/SciGen/test-Other/latex_source_code/unzipped/"
    ]

    for data_path, save_path in zip(data_paths, save_paths):

        scigen_data = load_from_disk(data_path)
        scigen_data = scigen_data.to_pandas()
        arxiv_ids = list(
            set(scigen_data.loc[scigen_data["venue"] == "arxiv", "paper_id"])
        )
        for tex_path in tex_paths:
            for id in arxiv_ids:
                tex_files = find_main_tex_by_document(tex_path, id)
                for tex_file in tex_files:
                    output_dir = os.path.join(save_path, id)
                    try:
                        convert_tex_to_html_xml(tex_file, output_dir)
                        print("Processed paper id:", id)
                    except Exception as e:
                        print("Faled to generate html/xml", e)


if __name__ == "__main__":
    main()
