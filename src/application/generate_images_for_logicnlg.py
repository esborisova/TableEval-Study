import imgkit
import os
import tempfile
from datasets import load_from_disk


imgkit.config(wkhtmltoimage='/usr/local/bin/wkhtmltoimage')

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
    imgkit.from_file(temp_path, output_path, options={"enable-local-file-access": ""})

    # Remove the temp file
    os.remove(temp_path)
    return output_path


def map_example(example):
    if example.get("matched_table_html") is None:
        return {"matched_table_image_path": None}

    table_id = example["table_id"]
    output_dir = "../../data/logicnlg_table_images"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{table_id}.png"
    html_to_image(example["matched_table_html"], output_path=output_path)
    return {"matched_table_image_path": output_path}


dataset = load_from_disk("../../data/logicnlg")
dataset = dataset.map(map_example)
