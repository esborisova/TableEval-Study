import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from ..utils.table_similarity import (
    prepare_tables,
    calculate_similarity_for_row,
    substitute_table,
)


def main():
    comtqa_with_tables = pd.read_csv("../../data/pubmed/comtqa_df.csv")
    jsons_rootdir = "/Users/ekbo01/Downloads/PubTables-1M-Structure_Table_Words/"
    pmc_all_tables = pd.read_csv("../../data/pubmed/pubmed_tables.csv")

    vectorizer = CountVectorizer()

    comtqa_with_tables = prepare_tables(
        df=comtqa_with_tables,
        header="table_header",
        content="table_content",
        merged_content=True,
    )

    merged_content_list = comtqa_with_tables[
        comtqa_with_tables["dataset"] == "PubTab1M"
    ]["merged_content"].tolist()
    vectorizer.fit(merged_content_list)

    comtqa_with_tables["similarity_score"] = comtqa_with_tables.apply(
        lambda row: calculate_similarity_for_row(row, jsons_rootdir, vectorizer), axis=1
    )

    comtqa_with_tables["similarity_score"] = pd.to_numeric(
        comtqa_with_tables["similarity_score"], errors="coerce"
    )
    comtqa_with_tables["similarity_score"] = comtqa_with_tables[
        "similarity_score"
    ].round(2)

    pubmed_subset = comtqa_with_tables[comtqa_with_tables["dataset"] == "PubTab1M"]
    low_similarity_indices = pubmed_subset[
        pubmed_subset["similarity_score"] < 0.90
    ].index

    low_similarity_indices_list = low_similarity_indices.tolist()

    pmc_all_tables = prepare_tables(
        df=pmc_all_tables, header="table_header", content="table_content"
    )

    columns_to_update = [
        "table_header",
        "table_content",
        "table_title",
        "table_caption",
        "table_footnote",
        "table_xml",
        "similarity_score",
    ]

    updated_comtqa_df = substitute_table(
        comtqa_with_tables,
        low_similarity_indices_list,
        pmc_all_tables,
        jsons_rootdir,
        vectorizer,
        columns_to_update,
    )

    updated_comtqa_df = updated_comtqa_df.drop(
        columns=[
            "cleaned_table_title",
            "clean_headers",
            "clean_content",
            "merged_content",
        ]
    )

    updated_comtqa_df["table_header"] = updated_comtqa_df["table_header"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    updated_comtqa_df["table_content"] = updated_comtqa_df["table_content"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    updated_comtqa_df.to_pickle("../../data/pubmed/comtqa_with_pmc_tables.pkl")


if __name__ == "__main__":
    main()
