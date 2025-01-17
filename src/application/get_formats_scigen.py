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

"""
A script that processes the SciGen data (both CL and Other subsets) and gets the source code of the 
LaTeX tables where possible. After that, the LatexML library is used to convert the tables into HTML and XML formats. 
To run the script, the argument 'data_path' must be passed to indicate the path to the SciGen data.

The script iterates over the papers, downloads the LaTeX fils from arXiv, and finds the tables 
based on 80% fuzzy matching the caption. 
The SciGen data is saved as a JSON file with the following keys added to each item:
1. "table_latex": containing the source code of the table if found, and none otherwise. 
2. "table_latex_source": containing the string "source_code" for tables that were found, none otherwise.
3. "table_html": containing output of HTML created by LatexML.
4. "table_html_source": containing the string "LatexML" for tables that were found, none otherwise.
5. "table_xml": containing output of XML created by LatexML.
6. "table_xml_source": containing the string "LatexML" for tables that were found, none otherwise.

Note that LaTeXML should be installed using '!sudo apt-get install latexml' before running the code.
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

def convert_tex_to_html_xml(tex_file, output_dir='converted'):
    """Converts a .tex file to HTML and XML using LaTeXML."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert .tex to .xml using LaTeXML
    xml_output = os.path.join(output_dir, os.path.basename(tex_file).replace('.tex', '.xml'))
    html_output = os.path.join(output_dir, os.path.basename(tex_file).replace('.tex', '.html'))

    # Run LaTeXML
    try:
      subprocess.run(['latexml', '--dest', xml_output, tex_file], check=True)
    except:
      xml_output = None

    try:
      subprocess.run(['latexmlpost', '--dest', html_output, xml_output], check=True)
    except:
      html_output = None

    return xml_output, html_output

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

def tex_to_html_xml(arxiv_id, target_caption):

    # Paths for download and extraction
    extract_path = f'extracted/{arxiv_id}/'

    # Step 1: Get the zip path (already downloaded when getting LaTeX)
    zip_path = os.path.join('zip_files', f'{arxiv_id}.tar.gz')

    # Step 2: Extract the .tex files
    tex_files = extract_tex_files(zip_path, extract_to=extract_path)
    print(f"Extracted {len(tex_files)} .tex files.")

    # Step 3: Convert each .tex file to HTML and XML
    converted_files = []
    for tex_file in tex_files:
      xml_file, html_file = convert_tex_to_html_xml(tex_file, output_dir=os.path.join('converted', arxiv_id))
      print(f"Converted {tex_file} to XML and HTML.")

      # Step 4: Search for table by caption in both XML and HTML
      html_table_found = False
      xml_table_found = False
      html_table = None
      xml_table = None

      # Read html_file
      if html_file:
        with open(html_file, 'r', encoding='utf-8') as file:
          html_content = file.read()

        if not html_table_found:
          html_table = extract_tables_by_caption_html(html_content, target_caption)
          if html_table != None:
            html_table_found = True

      # Read xml_file
      if xml_file:
        with open(xml_file, 'r', encoding='utf-8') as file:
          xml_content = file.read()

        if not xml_table_found:
          xml_table = extract_tables_by_caption_xml(xml_content, target_caption)
          if xml_table != None:
            xml_table_found = True

      for xml_file, html_file in converted_files:
        os.remove(xml_file)
        os.remove(html_file)

      if html_table_found and xml_table_found:
        break

    # Step 5: Cleanup the downloaded and extracted files as well as HTML and XML
    cleanup_files(zip_path, extract_path)

    return html_table, xml_table

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path for the SciGen dataset")
    args = parser.parse_args()

    data_path = args.data_path

    with open(data_path) as json_file:
        data = json.load(json_file)

    extracted_latex_tables = []
    extracted_latex_tables_source = []
    html_tables = []
    html_tables_sources = []
    xml_tables = []
    xml_tables_sources = []
    
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
                    extracted_latex_tables_source.append((key, 'source_code'))
                
                else:
                    extracted_latex_tables.append((key, None))
                    extracted_latex_tables_source.append((key, None))
            
            except:
                extracted_latex_tables.append((key, None))
                extracted_latex_tables_source.append((key, None))

            # Convert to HTML and XML
            html_table, xml_table = tex_to_html_xml(arxiv_id, caption)
            if html_table != None:
                html_tables.append((key, html_table))
                html_tables_sources.append((key, 'latexML'))
            
            else:
                html_tables.append((key, None))
                html_tables_sources.append((key, None))
                
            if xml_table != None:
                xml_tables.append((key, xml_table))
                xml_tables_sources.append((key, 'latexML'))
            
            else:
                xml_tables.append((key, None))
                xml_tables_sources.append((key, None))
 
    
    # Add extracted info to the data and save a new JSON file
    for idx, item in enumerate(extracted_latex_tables):
        key = item[0]
        value = item[1]
        data[key]['table_latex'] = value
        data[key]['table_latex_source'] = extracted_latex_tables_source[idx][1]

    for idx, item in enumerate(html_tables):
        key = item[0]
        value = item[1]
        data[key]['table_html'] = value
        data[key]['table_html_source'] = html_tables_sources[idx][1]

    for idx, item in enumerate(xml_tables):
        key = item[0]
        value = item[1]
        data[key]['table_xml'] = value
        data[key]['table_xml_source'] = html_tables_sources[idx][1]

    with open('data_with_source_latex.json', 'w') as json_file:
        json.dump(cl_data, json_file, indent=4)


if __name__ == "__main__":
    main()
