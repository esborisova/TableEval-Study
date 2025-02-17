import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_from_disk


df = load_from_disk("../../data/logicnlg").to_pandas()
df['matched_table_similarity'] = df['matched_table_similarity'].fillna(-1.0)

# Automatically calculate bin edges with more bins
num_bins = 50  # Adjust this number to increase/decrease bins
bin_edges = np.linspace(df['matched_table_similarity'].min(), df['matched_table_similarity'].max(), num_bins + 1)

# Bin the values and count occurrences
df['binned'] = pd.cut(df['matched_table_similarity'], bins=bin_edges)
bin_counts = df['binned'].value_counts().sort_index()

# Plot the bar chart
plt.figure(figsize=(12, 8))
bin_counts.plot(kind='bar', edgecolor='black')
plt.title('Distribution of Matched Table Similarity')
plt.xlabel('Similarity Range')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
