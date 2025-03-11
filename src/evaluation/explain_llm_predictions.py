import inseq
import pandas as pd
import pickle
import torch
from tqdm import tqdm
from transformers import AutoTokenizer


df = pd.read_json("../../predictions/mistralai/results_html_comtqa_fin_Mistral-7B-v0.1_2025-01-25_13_54_51.json")

model_id = "mistralai/Mistral-7B-v0.1" #"Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

explanations_save_dir = "../../explanations"

# Determine the device to use
if torch.cuda.is_available():
    device = "cuda"  # CUDA (NVIDIA GPU)
elif torch.backends.mps.is_available():
    device = "mps"   # MPS (Apple GPU)
else:
    device = "cpu"   # CPU fallback

inseq_model = inseq.load_model(
    model=model_id,
    tokenizer=tokenizer,
    attribution_method="attention",
    device=device,
)

for i, instance in tqdm(df.iterrows(), total=len(df)):
    input_text = instance["input"]
    generated_text = instance["prediction"]

    filename = instance["example"]["id"]

    if not generated_text:
        continue

    attribution_output = inseq_model.attribute(
        input_text,
        input_text + generated_text
    )

    # Accessing the input tokens for modification
    for seq in attribution_output.sequence_attributions:  # Iterate over sequences
        for token_attrib in seq.source:  # Iterate over source tokens
            # Clean up special tokens by replacing 'Ġ' with a space
            token_attrib.token = token_attrib.token.replace('Ġ', ' ')

    # Save attribution_output to file
    result = attribution_output.show(return_html=True)
    dic = attribution_output.get_scores_dicts()

    if device == "cuda":
        torch.cuda.empty_cache()

    with open(f"{explanations_save_dir}/{filename}_result.html", "w", encoding="utf-8") as f:
        f.write(result)

    with open(f"{explanations_save_dir}/{filename}_dic.pickle", "wb") as f: # Use 'wb' for writing binary data
        pickle.dump(dic, f)  # Save dic using pickle

