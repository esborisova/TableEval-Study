import imgkit
import os
import re
import requests
import tempfile
import time
from bs4 import BeautifulSoup
from datasets import Dataset, load_from_disk
from urllib.parse import urlparse, unquote


imgkit.config(wkhtmltoimage='/usr/local/bin/wkhtmltoimage')

# Global set to track processed table_ids
processed_ids = set()


def rewrite_and_download_images(table_html: str, image_dir: str) -> str:
    """
    Parse `table_html`, find all <img> tags, download images locally,
    and rewrite `src` to point to the downloaded file.

    :param table_html: The raw HTML string containing <img> tags.
    :param image_dir:  Path to the local directory where images will be saved.
    :return:           Modified HTML with local image paths instead of remote URLs.
    """
    # Ensure the directory for images exists
    os.makedirs(image_dir, exist_ok=True)

    soup = BeautifulSoup(table_html, "html.parser")

    # Track whether we actually downloaded any images
    downloaded_any = False

    for img_tag in soup.find_all("img"):
        src = img_tag.get("src")
        if not src:
            continue  # Skip <img> with no src

        # Fix protocol-relative references: //example.com -> https://example.com
        if src.startswith("//"):
            src = "https:" + src

        # Optionally remove Wayback prefixes (if you often see web.archive.org)
        # Example: https://web.archive.org/web/12345im_/https://upload.wikimedia.org/...
        # We can do a simple replacement or a more robust regex approach
        if "web.archive.org" in src:
            # One simplistic approach:
            # pattern: https://web.archive.org/web/<digits>im_/<actual https://...>
            match = re.match(r"https?://web\.archive\.org/web/\d+im_/(https?://.*)", src)
            if match:
                src = match.group(1)

        # Try downloading the image
        try:
            response = requests.get(src, timeout=10)
            response.raise_for_status()  # Raise an error for 4xx/5xx

            # Build a stable local filename from the URL
            # Derive filename using unquote
            parsed_url = urlparse(src)
            filename = os.path.basename(parsed_url.path)  # e.g. "23px-Flag_of_Canada_%28Pantone%29.svg.png"
            filename = unquote(filename)  # becomes "23px-Flag_of_Canada_(Pantone).svg.png"

            # If filename is empty or something weird, fallback
            if not filename:
                filename = "downloaded_image.png"

            local_path = os.path.join(image_dir, filename)

            # Write to local file
            with open(local_path, "wb") as f:
                f.write(response.content)

            # Rewrite the <img> tag to the local path
            # We can use a relative path if wkhtmltoimage is invoked from the same directory
            # e.g. "images/filename.png"
            # Or an absolute path. Either can work with --enable-local-file-access
            rel_path_for_html = os.path.abspath(local_path)

            img_tag["src"] = rel_path_for_html

        except (requests.RequestException, IOError) as e:
            # If download fails, remove this <img> or replace with a placeholder
            img_tag.decompose()  # remove the <img> tag entirely
            # OR:
            # img_tag["src"] = "placeholder.png"

    return str(soup)

def remove_wayback_prefix(html: str) -> str:
    # Remove the Wayback Machine prefix if present
    pattern = re.compile(r'https://web\.archive\.org/web/\d+im_/(https://.*)')
    html = pattern.sub(r'\1', html)
    return html

def html_to_image(table_html, output_path="table_image.png"):
    # Create a minimal HTML document
    html_content = f"""
    <html>
    <head>
      <meta charset="UTF-8">
      <style>
        table.wikitable {{
          border: 1px solid #aaa;
          border-collapse: collapse;
        }}
        .wikitable th, .wikitable td {{
          border: 1px solid #aaa;
          padding: 5px;
        }}
      </style>
    </head>
    <body>
      {table_html}
    </body>
    </html>
    """
    # Create a temporary HTML file
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        f.write(html_content.encode("utf-8"))
        temp_path = f.name

    # Convert the file to image
    # --enable-local-file-access is required for wkhtmltoimage >= 0.12.6
    imgkit.from_file(
        temp_path,
        output_path,
        options={
            "enable-local-file-access": "",
        }
    )

    # Remove the temp file
    os.remove(temp_path)
    return output_path


def map_example(example):
    global processed_ids

    # 1) If no matched_table_html, skip
    if example.get("matched_table_html") is None:
        return {"matched_table_image_path": None}

    # 2) Check if table_id was processed before
    table_id = example["table_id"]
    if table_id in processed_ids:
        # We already have an image for this table_id; skip re-creating
        return {"matched_table_image_path": None}

    # 3) Build output path and check if file exists
    output_dir = f"../../data/{dataset_name}_table_images"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{table_id}.png"

    # 4) Skip if file already exists
    if os.path.exists(output_path):
        processed_ids.add(table_id)
        return {"matched_table_image_path": output_path}

    # 5) Pre-download images and rewrite HTML
    #    We'll store all images in a subfolder named after the table_id (optional)
    images_dir = os.path.join(output_dir, f"{table_id}_images")
    replaced_html = rewrite_and_download_images(example["matched_table_html"], images_dir)

    # 6) Create the image
    html_to_image(replaced_html, output_path=output_path)

    # 7) Mark this table_id as processed
    processed_ids.add(table_id)

    time.sleep(5)

    return {"matched_table_image_path": output_path}


def generate_html_from_dataframe(df, output_file):
    # Start HTML structure with CSS for image resizing
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>HTML Table</title>
        <style>
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                border: 1px solid black;
                padding: 8px;
                text-align: center;
            }
            img {
                max-width: 200px;   /* Adjust image width */
                max-height: 200px;  /* Adjust image height */
            }
        </style>
    </head>
    <body>
        <table>
            <thead>
                <tr>
                    <th>Filename</th>
                    <th>Table</th>
                    <th>Image</th>
                </tr>
            </thead>
            <tbody>
    """

    # Iterate through DataFrame rows
    for _, row in df.iterrows():
        image_path = f"../../data/{dataset_name}_table_images/{row['image_id']}"

        # Debugging: Print the image path to check if it's correct
        print(f"Image path: {image_path}")

        html_content += f"""
        <tr>
            <td>{row['image_id']}</td>
            <td>{row['table_xml']}</td>
            <td><img src="{image_path}" alt="Image"></td>
        </tr>
        """

    # Close HTML structure
    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """

    # Save to an HTML file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(html_content)

dataset_name = "logicnlg"  # "logic2text" / "logicnlg"
dataset = load_from_disk(f"../../data/{dataset_name}").to_pandas()
dataset.loc[dataset["matched_table_similarity"] < 0.90,["matched_table_html"]] = None
filtered_dataset = dataset[dataset["matched_table_html"].notna()]
dataset = Dataset.from_pandas(filtered_dataset)
dataset = dataset.map(map_example)
