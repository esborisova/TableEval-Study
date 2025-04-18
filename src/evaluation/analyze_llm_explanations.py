import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import re
import seaborn as sns
import spacy
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer, logging as hf_logging


max_files_to_process = 10  # np.inf # FIXME: Debug limit set

# Optional: Suppress some Hugging Face warnings if they are too noisy
hf_logging.set_verbosity_error()


# Global variables for models to load them only once
nlp = None
tokenizer = None
special_tokens_set = None

def load_models(tokenizer_name="mistralai/Mistral-Nemo-Instruct-2407"):
    """Loads spaCy and Hugging Face models globally."""
    global nlp, tokenizer, special_tokens_set

    # Load spaCy model
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
            print("spaCy model 'en_core_web_sm' loaded.")
        except OSError:
            print("Downloading en_core_web_sm...")
            try:
                spacy.cli.download("en_core_web_sm")
                nlp = spacy.load("en_core_web_sm")
                print("spaCy model 'en_core_web_sm' loaded after download.")
            except Exception as e:
                print(f"Error loading spaCy model: {e}")
                nlp = None # Ensure it's None if loading fails

    # Load Hugging Face tokenizer
    if tokenizer is None:
        print(f"Loading Hugging Face tokenizer '{tokenizer_name}'...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            # Get the set of special tokens for efficient lookup
            special_tokens_set = set(tokenizer.all_special_tokens)
            print(f"Tokenizer '{tokenizer_name}' loaded. Special tokens identified: {len(special_tokens_set)}")
            # print(f"Special tokens: {special_tokens_set}") # Optional: print the set
        except Exception as e:
            print(f"Error loading tokenizer '{tokenizer_name}': {e}")
            tokenizer = None
            special_tokens_set = set() # Ensure it's an empty set if loading fails


def load_pickle(file_path):
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        # Basic check if data is list-like and not empty
        if isinstance(data, (list, tuple)) and data:
            return data
        else:
            print(f"Warning: Data in {file_path} is not a non-empty list/tuple or format is unexpected.")
            return None # Return None to indicate an issue
    except Exception as e:
        print(f"Error loading pickle file {file_path}: {e}")
        return None


def clean_word(word):
    # Remove leading special characters often introduced by tokenizers like SentencePiece (Ġ) or BPE (Ċ)
    word = re.sub(r'^[ĠĊ]+', '', word)
    # Keep only alphanumeric words, return None otherwise
    # Added check for word being non-empty after stripping special chars
    return word.strip() if word and word.isalnum() else None


def analyze_words(data):
    if nlp is None or tokenizer is None:
        print("Error: Models not loaded. Cannot analyze words.")
        return {}, {}, []  # Return empty structures

    word_type_scores = defaultdict(float)
    word_type_counts = defaultdict(int)
    word_details = []

    for sample in data:
        if "target_attributions" not in sample or not isinstance(sample["target_attributions"], dict):
            continue

        for (token_id, word), attributions in sample["target_attributions"].items():
            # --- Special Token Check ---
            # Check the *original* word against the special tokens set
            is_special = word in special_tokens_set
            pos_tag = None  # Reset pos_tag for each word

            if is_special:
                pos_tag = "SPECIAL"
                word_representation = word  # Use the original special token string
            else:
                # --- Regular Word Processing ---
                clean_w = clean_word(word)
                if clean_w:
                    # Get the word type (POS tag) using spaCy only if it's not special and is clean
                    doc = nlp(clean_w)
                    if doc and len(doc) > 0:
                        pos_tag = doc[0].pos_
                        word_representation = clean_w  # Use the cleaned word
                    # else: word is clean but spaCy failed? Skip maybe.
                # If clean_w is None (not alphanumeric after cleaning), skip this word
                # If spaCy failed, skip this word

            # --- Score Extraction and Aggregation ---
            if pos_tag:  # Proceed only if we have a valid tag (SPECIAL or spaCy POS)
                if isinstance(attributions, dict) and attributions:
                    try:
                        total_score = next(iter(attributions.values()))
                        scaled_score = total_score * 100

                        # Add details
                        word_details.append((word_representation, scaled_score, pos_tag))

                        # Update aggregates
                        word_type_scores[pos_tag] += scaled_score
                        word_type_counts[pos_tag] += 1

                    except (StopIteration, TypeError):
                        # Issue getting score from attributions dict
                        continue
                # else: Invalid attributions format

        # --- Calculate Averages and Top Words ---
    avg_scores = {
        word_type: word_type_scores[word_type] / word_type_counts[word_type]
        for word_type in word_type_scores if word_type_counts.get(word_type, 0) > 0
    }

    # Sort all collected word details by saliency DESCENDING, take top 10
    # The list now contains both cleaned regular words and original special tokens
    top_words = sorted(word_details, key=lambda x: x[1], reverse=True)[:10]

    if not avg_scores and not top_words:
        # print("\nNo valid words found in this file's data.") # Keep concise
        return {}, {}, []

    return avg_scores, word_type_counts, top_words


def calculate_overall_averages(results):
    aggregated_type_scores = defaultdict(float)
    aggregated_type_counts = defaultdict(int)
    aggregated_word_saliencies = defaultdict(float)
    word_occurrence_counts = defaultdict(int)

    for result in results:
        for word_type, avg_score in result["avg_scores"].items():
            count = result["word_type_counts"].get(word_type, 0)
            if count > 0:
                aggregated_type_scores[word_type] += avg_score * count
                aggregated_type_counts[word_type] += count

        for word, saliency, _ in result["top_words"]:
            aggregated_word_saliencies[word] += saliency
            word_occurrence_counts[word] += 1

    overall_type_averages = {
        word_type: aggregated_type_scores[word_type] / aggregated_type_counts[word_type]
        for word_type in aggregated_type_scores if aggregated_type_counts.get(word_type, 0) > 0
    }

    overall_word_averages = {
        word: aggregated_word_saliencies[word] / word_occurrence_counts[word]
        for word in aggregated_word_saliencies if word_occurrence_counts.get(word, 0) > 0
    }

    return overall_type_averages, overall_word_averages, word_occurrence_counts


def process_folder(folder_path):
    results = []
    print(f"Starting analysis in folder: {folder_path}")
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return []

    file_list = [f for f in os.listdir(folder_path) if f.endswith("_dic.pickle")]
    print(f"Found {len(file_list)} pickle files to process.")

    processed_count = 0

    for file_name in tqdm(file_list, total=len(file_list)):
        if processed_count >= max_files_to_process: # Apply debug limit
            print(f"\nReached debug limit of {max_files_to_process} files.")
            break

        file_path = os.path.join(folder_path, file_name)
        # print(f"Processing file: {file_name}") # Reduced verbosity

        data = load_pickle(file_path)

        if data:
            analysis_result = analyze_words(data)
            if analysis_result and any(analysis_result):
                avg_scores, word_type_counts, top_words = analysis_result
                results.append({
                    "file_name": file_name,
                    "avg_scores": avg_scores,
                    "word_type_counts": word_type_counts,
                    "top_words": top_words
                })
                processed_count += 1 # Increment counter only on successful processing
            # else:
                # print(f"  Skipping results for {file_name} due to no valid words found.") # Reduced verbosity
        # else:
             # print(f"  Skipping file {file_name} due to load error or empty data.") # Reduced verbosity

    print(f"\nSuccessfully processed {processed_count} files.")
    return results


# --- Plotting Functions ---

# Visualization 1: Overall Averages by Word Type
def plot_overall_type_averages(overall_type_averages):
    if not overall_type_averages:
        print("No overall type averages to plot.")
        return
    # Convert overall averages to a DataFrame
    avg_df = pd.DataFrame(list(overall_type_averages.items()), columns=["Word Type", "Average Saliency"])
    avg_df = avg_df.sort_values(by="Average Saliency", ascending=False)
    # print("\nOverall Average Saliency by Word Type (DataFrame):")
    # print(avg_df)

    # Plot the overall averages
    plt.figure(figsize=(12, 7)) # Adjusted size
    sns.barplot(x="Average Saliency", y="Word Type", data=avg_df, palette="viridis") # Added palette
    plt.title("Overall Average Saliency by Word Type", fontsize=16)
    plt.xlabel("Average Saliency Score (Scaled x100)", fontsize=12) # Clarified scale
    plt.ylabel("Word Type (POS Tag)", fontsize=12) # Clarified y-axis
    plt.tight_layout()
    plt.show()

# Visualization 2: Aggregated Word Type Counts
def plot_word_type_counts(file_results):
    if not file_results:
        print("No file results to plot word type counts from.")
        return
    # Prepare data for visualization
    counts_data = []
    for result in file_results:
        for word_type, count in result["word_type_counts"].items():
            counts_data.append({"Word Type": word_type, "Count": count})

    if not counts_data:
        print("No word type counts found in the results.")
        return

    # Convert to a DataFrame
    counts_df = pd.DataFrame(counts_data)

    # Aggregate counts by word type
    counts_df = counts_df.groupby("Word Type", as_index=False).sum()
    counts_df = counts_df.sort_values(by="Count", ascending=False)

    # Plot the word type counts
    plt.figure(figsize=(12, 7)) # Adjusted size
    sns.barplot(x="Count", y="Word Type", data=counts_df, palette="magma") # Changed palette, removed errorbar
    plt.title("Aggregated Word Type Counts Across All Files", fontsize=16) # Clarified title
    plt.xlabel("Total Count", fontsize=12)
    plt.ylabel("Word Type (POS Tag)", fontsize=12) # Clarified y-axis
    plt.tight_layout()
    plt.show()

# Visualization 3: Comparison of Word Type Counts and Saliency
def plot_comparison_saliency_and_counts(overall_type_averages, file_results):
    if not overall_type_averages or not file_results:
        print("Missing data for comparison plot.")
        return

    # Prepare data for visualization
    counts_data = defaultdict(int)
    for result in file_results:
        for word_type, count in result["word_type_counts"].items():
            counts_data[word_type] += count

    if not counts_data:
        print("No counts data available for comparison plot.")
        return

    # Combine saliency and counts into a single DataFrame, ensuring alignment
    comparison_data_list = []
    for word_type, avg_saliency in overall_type_averages.items():
        count = counts_data.get(word_type, 0) # Get count, default to 0 if type not found (shouldn't happen if calculated correctly)
        if count > 0: # Only include types that actually occurred
             comparison_data_list.append({
                 "Word Type": word_type,
                 "Average Saliency": avg_saliency,
                 "Count": count
             })

    if not comparison_data_list:
        print("No overlapping data between saliency and counts for comparison plot.")
        return

    comparison_data = pd.DataFrame(comparison_data_list)
    # Sort by Count for the bar plot base
    comparison_data = comparison_data.sort_values(by="Count", ascending=False)

    # print("\nComparison Data (Saliency vs. Count):")
    # print(comparison_data)

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(14, 7)) # Adjusted size

    # Plot the counts on the primary y-axis
    sns.barplot(x="Word Type", y="Count", data=comparison_data, palette="coolwarm", ax=ax1, alpha=0.8) # Changed palette
    ax1.set_ylabel("Total Count", fontsize=12, color='blue') # Color coordination
    ax1.set_xlabel("Word Type (POS Tag)", fontsize=12)
    ax1.set_title("Comparison of Word Type Counts and Average Saliency", fontsize=16, weight="bold")
    ax1.tick_params(axis="y", labelsize=10, colors='blue')
    ax1.tick_params(axis="x", labelsize=10, rotation=45) # Rotate labels if needed
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    # Create a secondary y-axis for saliency
    ax2 = ax1.twinx()
    # Use pointplot or lineplot - pointplot can be clearer with markers
    sns.pointplot(x="Word Type", y="Average Saliency", data=comparison_data, color="green", ax=ax2, markers="o", linestyles="-") # Changed color, style
    ax2.set_ylabel("Average Saliency Score", fontsize=12, color="green") # Color coordination
    ax2.tick_params(axis="y", labelsize=10, colors="green")
    ax2.set_ylim(bottom=0) # Ensure saliency axis starts at 0 if appropriate

    # Adjust layout
    fig.tight_layout()
    plt.show()

# Visualization 4: Correlation between Count and Saliency
def analyze_correlation(overall_type_averages, file_results):
    if not overall_type_averages or not file_results:
        print("Missing data for correlation analysis.")
        return
    # Prepare data for correlation analysis
    counts_data = defaultdict(int)
    for result in file_results:
        for word_type, count in result["word_type_counts"].items():
            counts_data[word_type] += count

    # Combine saliency and counts into a single DataFrame, ensuring alignment
    correlation_data_list = []
    for word_type, avg_saliency in overall_type_averages.items():
        count = counts_data.get(word_type, 0)
        if count > 0: # Only include types that actually occurred
             correlation_data_list.append({
                 "Word Type": word_type,
                 "Average Saliency": avg_saliency,
                 "Count": count
             })

    if len(correlation_data_list) < 2: # Need at least 2 points for correlation
        print("Not enough data points for correlation analysis.")
        return

    correlation_data = pd.DataFrame(correlation_data_list)

    # Calculate Pearson and Spearman correlation coefficients
    # Check for constant data which causes NaN in correlation
    if correlation_data["Count"].nunique() == 1 or correlation_data["Average Saliency"].nunique() == 1:
        print("Cannot calculate correlation: Count or Average Saliency data is constant.")
        pearson_corr, spearman_corr = float('nan'), float('nan')
    else:
        pearson_corr, p_pearson = pearsonr(correlation_data["Count"], correlation_data["Average Saliency"])
        spearman_corr, p_spearman = spearmanr(correlation_data["Count"], correlation_data["Average Saliency"])
        print(f"\nPearson Correlation: {pearson_corr:.3f} (p-value: {p_pearson:.3f})")
        print(f"Spearman Correlation: {spearman_corr:.3f} (p-value: {p_spearman:.3f})")

    # Plot the relationship between count and saliency
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Count", y="Average Saliency", data=correlation_data, s=100, color="purple", alpha=0.7) # Changed color
    # Optionally add labels to points
    # for i in range(correlation_data.shape[0]):
    #     plt.text(x=correlation_data.Count[i]+0.3, y=correlation_data['Average Saliency'][i]+0.3, s=correlation_data['Word Type'][i],
    #              fontdict=dict(color='black',size=9))
    plt.title("Correlation Between Word Type Count and Average Saliency", fontsize=16, weight="bold")
    plt.xlabel("Total Count", fontsize=12)
    plt.ylabel("Average Saliency Score", fontsize=12)
    plt.grid(linestyle="--", alpha=0.7)
    # Add correlation values to the plot
    plt.annotate(f'Pearson r={pearson_corr:.2f}\nSpearman ρ={spearman_corr:.2f}',
                 xy=(0.05, 0.9), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))
    plt.tight_layout()
    plt.show()


# --- NEW Plotting Function ---
# Visualization 5: Overall Average Saliency for Top Words
def plot_top_overall_words(overall_word_averages, word_occurrence_counts, top_n=20):
    """
    Plots the top N words based on their overall average saliency score
    across all files where they appeared in the top 10.
    """
    if not overall_word_averages:
        print("No overall word averages to plot.")
        return

    # Convert the dictionary to a DataFrame
    word_avg_df = pd.DataFrame(list(overall_word_averages.items()), columns=["Word", "Average Saliency"])

    # Add occurrence count for context (optional, but useful)
    word_avg_df["OccurrencesInTop10"] = word_avg_df["Word"].map(word_occurrence_counts)

    # Sort by Average Saliency in descending order
    word_avg_df = word_avg_df.sort_values(by="Average Saliency", ascending=False)

    # Select the top N words
    top_words_df = word_avg_df.head(top_n)

    # print(f"\nTop {top_n} Words by Overall Average Saliency:")
    # print(top_words_df)

    # Create the plot (horizontal bar chart is good for word labels)
    plt.figure(figsize=(10, 8)) # Adjust figsize as needed
    barplot = sns.barplot(x="Average Saliency", y="Word", data=top_words_df, palette="rocket") # Using y="Word" for horizontal

    # Add labels with occurrence counts onto the bars
    for index, row in top_words_df.iterrows():
        barplot.text(row["Average Saliency"] / 2, # Position text in the middle of the bar
                     index, # Use index for y position in horizontal barplot
                     f'{row["Average Saliency"]:.2f} (n={row["OccurrencesInTop10"]})', # Text content
                     color='white', ha="center", va="center", fontsize=9, weight='bold')


    plt.title(f"Top {top_n} Words by Overall Average Saliency", fontsize=16)
    plt.xlabel("Overall Average Saliency Score (Scaled x100)", fontsize=12)
    plt.ylabel("Word", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    # --- Load Models Once ---
    # Specify the tokenizer you want to use here
    hf_tokenizer_name = "mistralai/Mistral-Nemo-Instruct-2407"
    load_models(tokenizer_name=hf_tokenizer_name)

    # Check if models loaded successfully before proceeding
    if nlp is None or tokenizer is None:
        print("Exiting due to model loading failure.")
        exit()

    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_folder = os.path.abspath(os.path.join(script_dir, "../../explanations/inseq/"))
    output_file = "word_analysis_results.json"

    # --- Process Data or Load Existing Results ---
    if os.path.exists(output_file):
        print(f"Loading pre-computed results from {output_file}...")
        try:
            with open(output_file, "r") as f:
                output_data = json.load(f)
            overall_type_averages = output_data.get("overall_type_averages", {})
            overall_word_averages = output_data.get("overall_word_averages", {})
            file_results = output_data.get("file_results", [])
            # Recalculate occurrence counts from loaded results
            _, _, word_occurrence_counts = calculate_overall_averages(file_results)
            print("Results loaded successfully.")
            if not file_results:
                print("Warning: Loaded file contains no individual file results. Attempting re-processing.")
                results = process_folder(target_folder)  # Will use loaded models
                if results:
                    overall_type_averages, overall_word_averages, word_occurrence_counts = calculate_overall_averages(
                        results)
                    file_results = results
                else:
                    print("Processing failed. Cannot proceed.")
                    exit()
        except Exception as e:  # Catch broader exceptions during load/recalc
            print(f"An error occurred loading or processing JSON data: {e}. Attempting re-processing.")
            results = process_folder(target_folder)  # Will use loaded models
            if not results:
                print("Processing failed. Cannot proceed.")
                exit()
            overall_type_averages, overall_word_averages, word_occurrence_counts = calculate_overall_averages(results)
            file_results = results
    else:
        print(f"{output_file} not found. Processing folder...")
        results = process_folder(target_folder)  # Will use loaded models
        if not results:
            print("No data processed. Exiting.")
            exit()
        overall_type_averages, overall_word_averages, word_occurrence_counts = calculate_overall_averages(results)
        file_results = results
        # Save results
        output_data = {
            "overall_type_averages": overall_type_averages,
            "overall_word_averages": overall_word_averages,
            "file_results": results,
            # "word_occurrence_counts": word_occurrence_counts # Optionally save this too
        }
        try:
            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=4)
            print(f"\nResults saved to {output_file}")
        except Exception as e:
            print(f"\nError saving results to JSON: {e}")

    # --- Run Visualizations ---
    print("\nGenerating plots...")

    # Call plotting functions with checks
    if overall_type_averages:
        plot_overall_type_averages(overall_type_averages)
    else:
        print("Skipping overall type averages plot: No data.")

    if file_results:
        plot_word_type_counts(file_results)
    else:
        print("Skipping word type counts plot: No data.")

    if overall_type_averages and file_results:
        plot_comparison_saliency_and_counts(overall_type_averages, file_results)
        analyze_correlation(overall_type_averages, file_results)
    else:
        print("Skipping comparison and correlation plots: Missing data.")

    if overall_word_averages and word_occurrence_counts:
        plot_top_overall_words(overall_word_averages, word_occurrence_counts, top_n=25)  # Show top 25
    else:
        print("Skipping top overall words plot: Missing data.")

    print("\nAnalysis complete.")