"""Pipeline for resolving incorrectly added PMC tables (relying on tables titles) 
based on source annotations and cosine similarity score."""
import pandas as pd
import ast
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from ..utils.table_similarity import (
    prepare_tables,
    calculate_similarity_for_row,
    substitute_table,
)
from ..utils.other import create_and_save_dataset


def main():
    comtqa_with_tables = pd.read_csv(
        "../../data/ComTQA_data/pubmed/utils/comtqa_df_updated_2014-12-01.csv"
    )
    jsons_rootdir = (
        "../../data/ComTQA_data/pubmed/source_data/PubTables-1M-Structure_Table_Words/"
    )
    pmc_all_tables = pd.read_csv(
        "../../data/ComTQA_data/pubmed/utils/pubmed_tables_updated_2014-12-01.csv"
    )

    vectorizer = CountVectorizer()

    comtqa_with_tables = prepare_tables(
        df=comtqa_with_tables,
        header="table_headers",
        subheader="table_subheaders",
        content="table_rows",
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
        df=pmc_all_tables,
        header="table_headers",
        subheader="table_subheaders",
        content="table_rows",
    )

    columns_to_update = [
        "table_headers",
        "table_subheaders",
        "table_rows",
        "table_title",
        "table_caption",
        "table_footnote",
        "table_xml",
        "table_xml_no_meta",
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
            "clean_subheaders",
            "clean_rows",
            "merged_content",
        ]
    )

    updated_comtqa_df["table_headers"] = updated_comtqa_df["table_headers"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    updated_comtqa_df["table_subheaders"] = updated_comtqa_df["table_subheaders"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    updated_comtqa_df["table_rows"] = updated_comtqa_df["table_rows"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    date = datetime.now().strftime("%Y-%m-%d")
    updated_comtqa_df.to_pickle(
        f"../../data/ComTQA_data/pubmed/utils/comtqa_with_pmc_tables_updated_{date}.pkl"
    )

    create_save_dataset(updated_comtqa_df, "train", "../../data/ComTQA_data/comtqa_updated")



if __name__ == "__main__":
    main()
