
"""
Script to review extracted LaTeX tables in SciGen.
The script reconstructs the gold data and compares it to the LaTeX code after cleaning it.
These are then compared using cosine similarity and any items with a similarity lower than 0.3 are checked manually.
As a result, 16 LaTeX tables were removed. 
"""
from datasets import load_from_disk, Dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import re

def get_gold_reconstruced(table_column_names, table_content_values):
  table_reconstructed = ""
  for idx, col in enumerate(table_column_names):
    table_reconstructed += col + ' '

  for row in table_content_values:
    for cell in row:
      table_reconstructed += cell + ' '

  # remove anything between brackets [], including the brackets
  table_reconstructed = re.sub(r'\[.*?\]', '', table_reconstructed)

  # Remove special characters
  table_reconstructed = re.sub(r'[^a-zA-Z0-9.\s]', '', table_reconstructed)

  # remove extra whitespaces
  table_reconstructed = ' '.join(table_reconstructed.split())

  return table_reconstructed

def remove_latex_commands(table_content):
    """
    Cleans LaTeX table content by:
    - Removing LaTeX commands related to tables while keeping table content.
    - Removing unnecessary formatting, keeping only meaningful data.

    Parameters:
        table_content (str): The LaTeX table content as a string.

    Returns:
        str: The cleaned and formatted table content.
    """

    # Remove caption commands and their content
    table_content = re.sub(r'\\caption\{.*?\}', '', table_content, flags=re.DOTALL)

    # Remove citation commands (\cite, \citep, \citet, etc.) and their content
    table_content = re.sub(r'\\cite[tp]?\*?(?:\[[^\]]*\])?\{.*?\}', '', table_content, flags=re.DOTALL)
    
    # Remove label commands and their content
    table_content = re.sub(r'\\label\{.*?\}', '', table_content)

    # Remove [h] in table environments
    table_content = table_content.replace(r'[h|ht|htb|t]', '')

    # Remove numbers in multirow and multicolumn commands (e.g., \multirow{6}{*}{Content})
    table_content = re.sub(r'\\multicolumn\{\d+\}\{.*?\}\{(.*?)\}', r'\1', table_content)
    table_content = re.sub(r'\\multirow\{\d+\}\{.*?\}\{(.*?)\}', r'\1', table_content)


    # Remove LaTeX table and tabular environment wrappers but keep the content inside
    table_content = re.sub(r'\\begin\{table\*?\}.*?|\\end\{table\*?\}', '', table_content, flags=re.DOTALL)
    table_content = re.sub(r'\\begin\{tabular\*?\}.*?|\\end\{tabular\*?\}', '', table_content, flags=re.DOTALL)
    table_content = re.sub(r'\\begin\{scriptsize\*?\}.*?|\\end\{scriptsize\*?\}', '', table_content, flags=re.DOTALL)

    # Remove standalone LaTeX commands but keep content inside braces {}
    table_content = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}', r'\1', table_content, flags=re.DOTALL)

    table_content = re.sub(r'\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}', r'\1', table_content, flags=re.DOTALL)

    # Remove specific table-related design commands while keeping meaningful content
    table_content = re.sub(r'\\(table|t|htb|ht|h|scriptsize|centering|center|resizebox|maxwidth|adjustbox|linewidth|textwidth|columnwidth|arraystretch|tabcolsep|toprule|midrule|bottomrule|hline|cline\{.*?\})', '', table_content)

    # Remove column formatting inside {}
    table_content = re.sub(r'\{[lcr| \t]+\}', '', table_content)  # Removes {ll|ccc| } or spaces in {}
    
    # Remove "number + px" and "number + em"
    table_content = re.sub(r'\b\d+(?:\.\d+)?(?:px|em)\b', '', table_content)

    # Remove '\\' and replace with space
    table_content = table_content.replace('\\', '')

    # Remove '&' and replace with space
    table_content = table_content.replace('&', ' ')

    # Remove leftover special characters (excluding alphanumerics, dots, and spaces)
    table_content = re.sub(r'[^\w\s\.\-\(\),]', '', table_content)

    # Clean extra whitespace
    table_content = re.sub(r'\s+', ' ', table_content).strip()

    return table_content



def main():
	sicgen_cl = load_from_disk("/netscratch/borisova/TableEval/data/SciGen/test-CL/test_CL_updated_2025-01-16")

	# remove 'table_latex_source' from features
	sicgen_cl = sicgen_cl.remove_columns(['table_latex_source'])

	suspected_incorrect_latex = []

	for idx, item in enumerate(sicgen_cl):

		if item['table_latex'] != None:

			table_column_names = item['table_column_names']
			table_content_values = item['table_content_values']
			table_latex = item['table_latex']

			table_reconstructed = get_gold_reconstruced(table_column_names, table_content_values)
			table_latex_reconstructed = remove_latex_commands(table_latex)

			vectorizer = CountVectorizer().fit([table_reconstructed] + [table_latex_reconstructed])
			vectors = vectorizer.transform([table_reconstructed] + [table_latex_reconstructed])
			similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

			similarity_scores.append(similarities[0])

			if similarities[0] < 0.3:
				suspected_incorrect_latex.append(item)
				print(idx)
				print(similarities[0])

	# Suspected items were reviewed manually and the following indices are those where a wrong latex table was extracted
	indices_to_remove = [51, 58, 67, 78, 169, 173, 177, 193, 194, 195, 224, 293, 294, 308, 316, 423]

	for idx in indices_to_remove:
		sicgen_cl = sicgen_cl.to_list()  # Convert to mutable format
		sicgen_cl[idx]["table_latex"] = None
		sicgen_cl = Dataset.from_list(sicgen_cl)  # Convert back to Dataset

	sicgen_cl.save_to_disk("/netscratch/borisova/TableEval/data/SciGen/test-CL/test_CL_updated_2025-01-17")

if __name__ == "__main__":
	main()