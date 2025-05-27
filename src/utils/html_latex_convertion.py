import pandas as pd
import re
import os
import subprocess
from bs4 import BeautifulSoup
from typing import Dict, List
from ..utils.other import create_dir, save_table_to_file


def replace_with_latex_symbols(latex_table: str, replacements: Dict[str, str]) -> str:
    for symbol, latex_symbol in replacements.items():
        latex_table = latex_table.replace(symbol, latex_symbol)
    return latex_table


def clean_column_names(cols):
    cleaned_cols = []
    for col in cols:
        if isinstance(col, tuple):
            cleaned_col = " ".join(
                [item if "Unnamed" not in str(item) else "" for item in col]
            )
        elif isinstance(col, str):
            cleaned_col = "" if "Unnamed" in col else col
        else:
            cleaned_col = col

        cleaned_cols.append(cleaned_col)
    return cleaned_cols


def fix_auto_generated_headers(df: pd.DataFrame) -> pd.DataFrame:
    if list(df.columns) == list(range(len(df.columns))):
        df.columns = [""] * len(df.columns)
    return df


def remove_footnote_row(html: str):
    soup = BeautifulSoup(html, "html.parser")
    for tfoot in soup.find_all("tfoot"):
        tfoot.extract()
    return soup


def html_to_df(html: str) -> pd.DataFrame:
    soup = remove_footnote_row(html)
    html_df = pd.read_html(str(soup))[0]
    html_df = fix_auto_generated_headers(html_df)
    html_df.columns = clean_column_names(html_df.columns)
    return html_df


def html_to_latex(html: pd.DataFrame) -> str:
    html = html.fillna("")
    table_latex = html.to_latex(index=False, float_format="{:g}".format)
    return table_latex


def clean_latex_string(latex_str: str) -> str:
    latex_str = re.sub(r"\\\$(?=[^\d\w])", "$", latex_str)
    return latex_str


def get_table_numb(table_title: str) -> int:
    table_number = int(re.search(r"Table (\d+)", table_title).group(1))
    return table_number


def add_meta_table_tex(
    table_latex: str,
    table_title: str = None,
    table_caption: str = None,
    table_footnote: str = None,
) -> str:

    latex_table = f"\\begin{{table}}[ht]\n\\centering\n"

    if table_title:
        table_number = get_table_numb(table_title) if table_title else 1
        latex_table += f"\\setcounter{{table}}{{{table_number - 1}}}\n"

    if table_caption and table_footnote:
        latex_table += (
            f"\\captionsetup{{justification=raggedright, singlelinecheck=false}}\n"
        )
        latex_table += f"\\begin{{threeparttable}}\n \\caption{{{table_caption}}}\n"

    elif table_caption:
        latex_table += f"\\captionsetup{{justification=raggedright, singlelinecheck=false}}\n \\caption{{{table_caption}}}\n"

    latex_table += f"{table_latex}\n"

    if table_footnote:
        latex_table += f"\\begin{{tablenotes}}\n \\footnotesize\n \\item {table_footnote}\n \\end{{tablenotes}}\n \\end{{threeparttable}}\n"

    latex_table += "\\end{table}\n"

    return latex_table


def convert_html_to_latex_tables(
    df: pd.DataFrame,
    html_column: str,
    html_replacements: Dict[str, str],
    tex_replacements: Dict[str, str],
    all_replacements: Dict[str, str],
    title_column: str = None,
    caption_column: str = None,
    footnote_column: str = None,
) -> List[str]:

    tables_latex = []
    for _, row in df.iterrows():
        if row[html_column] is not None:

            preprocessed_html = replace_with_latex_symbols(
                row[html_column], html_replacements
            )
            html_df = html_to_df(preprocessed_html)
            table_latex = html_to_latex(html_df)
            table_latex = replace_with_latex_symbols(table_latex, tex_replacements)
            table_latex = clean_latex_string(table_latex)

            table_title = row.get(title_column, None) if title_column else None

            table_caption = row.get(caption_column, None) if caption_column else None
            if table_caption is not None:
                table_caption = replace_with_latex_symbols(
                    table_caption, all_replacements
                )

            table_footnote = row.get(footnote_column, None) if footnote_column else None
            if table_footnote is not None:
                table_footnote = replace_with_latex_symbols(
                    table_footnote, all_replacements
                )

            table_tex_with_meta = add_meta_table_tex(
                table_latex, table_title, table_caption, table_footnote
            )

            tables_latex.append(table_tex_with_meta)

    return tables_latex


def generate_latex_content(table_latex: str) -> str:
    latex_begin = f"\\documentclass{{article}}\n \\usepackage{{wasysym}}\n \\usepackage{{graphicx}}\n \\usepackage{{array}}\n \\usepackage{{booktabs}}\n \\usepackage{{caption}}\n \\usepackage{{tablefootnote}}\n \\usepackage{{threeparttable}}\n \\begin{{document}}\n"
    latex_end = "\\end{document}"

    complete_latex = latex_begin + table_latex + latex_end
    return complete_latex


def process_and_save_table_tex(
    df: pd.DataFrame, id_col: str, table_latex_col: str, root_dir: str
):
    create_dir(root_dir)

    for _, row in df.iterrows():
        filename = row[id_col] + ".tex"
        file_path = os.path.join(root_dir, filename)
        if row[table_latex_col] is not None:
            table_latex = generate_latex_content(row[table_latex_col])
            save_table_to_file(file_path, table_latex)


def compile_tex_file(file_path, output_dir, error_log_file):
    process = subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-output-directory",
            output_dir,
            file_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        output = process.stdout.decode("utf-8")
    except UnicodeDecodeError:
        output = process.stdout.decode("latin1")

    if process.returncode != 0 or "!" in output:
        log_error(file_path, output, error_log_file)
        return False
    else:
        return True


def log_error(file_name, error_output, error_log_file):
    with open(error_log_file, "a") as log:
        log.write(f"File: {file_name}\n")
        log.write("Error Output:\n")
        log.write(error_output)
        log.write("\n" + "-" * 40 + "\n")


def compile_tex_files_in_dir(tex_dir, output_dir, error_log_file):
    create_dir(output_dir)

    log_dir = os.path.dirname(error_log_file)
    create_dir(log_dir)

    with open(error_log_file, "w") as log:
        log.write("Errors encountered during compilation:\n\n")

    for file_name in os.listdir(tex_dir):
        if file_name.endswith(".tex"):
            file_path = os.path.join(tex_dir, file_name)
            print(f"Compiling {file_path}...")

            success = compile_tex_file(file_path, output_dir, error_log_file)

            if success:
                print(f"{file_name} compiled successfully.")
            else:
                print(f"Error in {file_name}, logging error...")

    print(f"Compilation completed. Errors logged in {error_log_file}")


def escape_non_math_dollar(latex_text: str):
    """
    Escapes dollar signs ($) only if they are not part of math mode in LaTeX.
    """
    math_mode_regex = re.compile(r"\$(.*?)\$")
    math_matches = math_mode_regex.findall(latex_text)
    temp_placeholder = "\uFFFF"
    temp_latex = math_mode_regex.sub(temp_placeholder, latex_text)
    temp_latex = temp_latex.replace("$", r"\$")
    for math_content in math_matches:
        temp_latex = temp_latex.replace(temp_placeholder, f"${math_content}$", 1)
    return temp_latex
