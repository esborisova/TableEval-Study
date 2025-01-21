import json
import requests
import tarfile
import shutil
import os
import re
from TexSoup import TexSoup
from thefuzz import fuzz
import argparse
import subprocess
import glob
from ..utils.other import create_and_save_dataset
import pandas as pd


"""
A script that processes the SciGen data (both CL and Other subsets) and gets the source code of the 
LaTeX tables where possible. 
To run the script, the argument 'data_path' must be passed to indicate the path to the SciGen data.

The script iterates over the papers, downloads the LaTeX fils from arXiv, and finds the tables 
based on 80% fuzzy matching the caption. 
The SciGen data is saved as a JSON file with the following keys added to each item "table_latex", which contains the source code of the table if found, and None otherwise. 

"""

def download_arxiv_source(arxiv_id, download_path='zip_files'):
    """Downloads the source of the given arXiv ID as a zip file."""

    arxiv_url = f'https://arxiv.org/e-print/{arxiv_id}'
    print(arxiv_url)

    # Ensure the download path exists
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # Download the zip file containing the source
    response = requests.get(arxiv_url)
    response.raise_for_status()

    zip_path = os.path.join(download_path, f'{arxiv_id}.tar.gz')

    with open(zip_path, 'wb') as f:
        f.write(response.content)

    return zip_path


def extract_tex_files(zip_path, extract_to='extracted/'):
    """Extracts all .tex files from the downloaded zip file."""

    os.makedirs(extract_to, exist_ok=True)

    with tarfile.open(zip_path, "r:gz") as tar:
        tar.extractall(path=extract_to)

    tex_files = []
    # Find all .tex files
    for root, dirs, files in os.walk(extract_to):
        for file in files:
            if file.endswith('.tex'):
                tex_files.append(os.path.join(root, file))

    return tex_files

def normalize_caption(caption):
    """
    Removes LaTeX formatting and special commands from the caption.
    """
    # Remove common LaTeX formatting commands such as \textbf{}, \textsc{}, \citet{}, \refapp{}
    caption = re.sub(r'\\text\w+\{([^}]*)\}', r'\1', caption)  # Removes \textbf{}, \textsc{}
    caption = re.sub(r'\\citet?\{([^}]*)\}', '', caption)  # Removes \citet{}
    caption = re.sub(r'\\refapp\{([^}]*)\}', '', caption)  # Removes \refapp{}

    # Remove any remaining LaTeX commands (starting with \)
    caption = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', caption)  # Removes any \command{}
    caption = re.sub(r'\\[a-zA-Z]+\b', '', caption)  # Removes any \command

    # Remove extra spaces and line breaks
    caption = caption.replace('\n', ' ').strip()
    caption = caption.replace('{', '').strip()
    caption = caption.replace('}', '').strip()

    return caption

def find_table_by_caption(tex_file, caption):
    """Searches for a LaTeX table in the .tex file with the given caption."""
    with open(tex_file, 'r', encoding='utf-8') as file:
        content = file.read()

    content = remove_commented_lines(content)
    # remove $ from content
    content = content.replace('$', '')
    # Regex to find all tables and captions
    table_pattern = re.compile(r'\\begin\{table\*?\}(.*?)\\end\{table\*?\}', re.DOTALL)

    for table_match in table_pattern.finditer(content):
        table_code = table_match.group(0)
        soup = TexSoup(table_code)

        table_soup_1 = soup.find_all('table')
        table_soup_2 = soup.find_all('table*')

        for item in table_soup_1:
          if item.caption != None:
            normalized_caption = normalize_caption(item.caption.string)
            fuzzy_match_score = fuzz.ratio(normalized_caption, caption)
            if fuzzy_match_score >= 80:
              return table_code

        for item in table_soup_2:
          if item.caption != None:
            normalized_caption = normalize_caption(item.caption.string)
            fuzzy_match_score = fuzz.ratio(normalized_caption, caption)
            if fuzzy_match_score >= 80:
              return table_code

    return None

def cleanup_files(download_path, extract_path):
    """Removes the downloaded and extracted files."""
    if os.path.exists(download_path):
        os.remove(download_path)
    if os.path.exists(extract_path):
        for root, dirs, files in os.walk(extract_path, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(extract_path)


def process_arxiv_paper(arxiv_id, table_caption):
    # Paths for download and extraction
    extract_path = f'extracted/{arxiv_id}/'

    # Step 1: Download the source
    zip_path = download_arxiv_source(arxiv_id)
    if zip_path is None:
        return None

    # Step 2: Extract the .tex files
    tex_files = extract_tex_files(zip_path, extract_to=extract_path)

    # Step 3: Search for the table in each .tex file
    for tex_file in tex_files:
        table_code = find_table_by_caption(tex_file, table_caption)
        if table_code:
            print(f"Found table for {arxiv_id}:")
            return table_code
    else:
        print(f"Table not found for {arxiv_id}")

    # Step 4: Cleanup the downloaded and extracted files
    cleanup_files(zip_path, extract_path)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path for the SciGen dataset")
    parser.add_argument("--output_path", type=str, help="Path for the saving the resulting dataset")
    args = parser.parse_args()
    

    data_path = args.data_path
    output_path = args.output_path

    with open(data_path) as json_file:
        data = json.load(json_file)

    extracted_latex_tables = []
    
    for key, value in data.items():
        arxiv_id = value['paper_id']
        caption = value['table_caption']
        venue = item['venue']

        if venue == 'arxiv':
            print(key)
            try:
                caption = caption.split(':', 1)[1].strip()
            except:
                try:
                    caption = caption.split('.', 1)[1:].strip()
                except:
                    caption = caption
            #Get LaTeX
            try:
                latex_table_code = process_arxiv_paper(arxiv_id, caption)
                if latex_table_code:
                    extracted_latex_tables.append((key, latex_table_code))
                else:
                    extracted_latex_tables.append((key, None))       
            except:
                extracted_latex_tables.append((key, None))
 
    
    # Add extracted info to the data 
    for idx, item in enumerate(extracted_latex_tables):
        key = item[0]
        value = item[1]
        data[key]['table_latex'] = value

    # Convert data for pandas DataFrame
    data_df = pd.DataFrame.from_dict(data, orient='index')
    # Save as HF Dataset
    create_and_save_dataset(data_df, "test", output_path)
    

if __name__ == "__main__":
    main()
