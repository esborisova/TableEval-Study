import imgkit
import os
import re
import tempfile
import time
from datasets import load_from_disk


imgkit.config(wkhtmltoimage='/usr/local/bin/wkhtmltoimage')

# Global set to track processed table_ids
processed_ids = set()

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
    if os.path.exists(output_path):
        # Already have an image file at this location; skip creation
        processed_ids.add(table_id)
        return {"matched_table_image_path": output_path}

    # 4) Modify HTML to fix protocol-relative // URLs (example)
    replaced_html = example["matched_table_html"]
    replaced_html = replaced_html.replace('src="//', 'src="https://')
    replaced_html = replaced_html.replace('href="//', 'href="https://')

    # 5) Create the image
    try:
        html_to_image(replaced_html, output_path=output_path)
    except ConnectionRefusedError:
        replaced_html = remove_wayback_prefix(example["matched_table_html"])
        replaced_html = replaced_html.replace('src="//', 'src="https://')
        replaced_html = replaced_html.replace('href="//', 'href="https://')
        html_to_image(replaced_html, output_path=output_path)

    # 6) Mark this table_id as processed
    processed_ids.add(table_id)

    time.sleep(30)

    return {"matched_table_image_path": output_path}


dataset_name = "logic2text"  # "logic2text"
dataset = load_from_disk(f"../../data/{dataset_name}")
dataset = dataset.map(map_example)
