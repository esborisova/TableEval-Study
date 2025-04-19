import argparse
import datasets
import inseq
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer


mpl.rcParams['text.usetex']        = False
mpl.rcParams['mathtext.default']   = 'regular'
mpl.rcParams['mathtext.fontset']    = 'dejavusans'  # or whatever font you like

_escape_re = re.compile(r'([\\\$\_\%\&\#\{\}])')
def escape_for_matplotlib(s: str) -> str:
    return _escape_re.sub(r'\\\1', s)

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True,
                    help="The input file with predictions to process.")
parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B",
                    help="The model to use.")
parser.add_argument("--attribution_method", type=str, default="saliency")
parser.add_argument("--source_data_path", type=str,
                    default="../../data/LogicNLG/logicnlg_updated_2025-03-13")
parser.add_argument("--instance_ids", action="append", type=str, default=[])
parser.add_argument("--output_dir", type=str, default="../../explanations/inseq")
args = parser.parse_args()

# --- Load Data ---
print("Loading data...")
predictions_df = pd.read_json(args.input_file)
if 'example' not in predictions_df.columns or 'prediction' not in predictions_df.columns:
    raise ValueError("Input file must contain 'example' and 'prediction' columns.")

# --- Determine Merge Key ---
if isinstance(predictions_df['example'].iloc[0], dict) and 'instance_id' in predictions_df['example'].iloc[0]:
    predictions_df['instance_id'] = predictions_df['example'].apply(lambda x: x.get('instance_id'))
    merge_on = 'instance_id'
elif 'id' in predictions_df.columns:
    merge_on = 'id'
elif 'instance_id' in predictions_df.columns:
    merge_on = 'instance_id'
else:
    raise ValueError("Cannot find a unique ID ('instance_id', 'id') to merge on.")
print(f"Using merge key: '{merge_on}'")

# --- Ensure Merge Key is String in predictions_df ---
if merge_on not in predictions_df.columns:
    raise ValueError(f"Determined merge key '{merge_on}' not found in predictions_df after extraction.")
print(
    f"Converting '{merge_on}' to string in predictions_df (dtype before: {predictions_df[merge_on].dtype})...")
# Convert to string, handling potential errors if column contains mixed types/NaNs gracefully
try:
    predictions_df[merge_on] = predictions_df[merge_on].astype(str)
except Exception as e_astype:
    print(
        f"Warning: Failed to directly convert {merge_on} to string in predictions_df: {e_astype}. Trying apply(str).")
    # Fallback for complex cases, might be slower
    predictions_df[merge_on] = predictions_df[merge_on].apply(lambda x: str(x) if pd.notna(x) else x)

print(f" -> dtype after: {predictions_df[merge_on].dtype}")

# --- Load original dataset ---
original_ds = datasets.load_from_disk(args.source_data_path)
split = next((s for s in ['test', 'validation', 'train'] if s in original_ds), list(original_ds.keys())[0])
print(f"Using split '{split}' from source dataset.")
od = original_ds[split].to_pandas()

# --- Ensure Merge Key exists and is String in od ---
original_od_cols = od.columns.tolist()  # Keep track of original columns
key_found_in_od = False
if merge_on in od.columns:
    print(f"Found merge key '{merge_on}' directly in source data.")
    key_found_in_od = True
# Try renaming common alternative ID columns if direct key not found
elif merge_on == 'instance_id' and 'id' in od.columns:
    print(f"Merge key '{merge_on}' not found, renaming 'id' column in source data.")
    od = od.rename(columns={'id': 'instance_id'})
    key_found_in_od = True
elif merge_on == 'id' and 'instance_id' in od.columns:
    print(f"Merge key '{merge_on}' not found, renaming 'instance_id' column in source data.")
    od = od.rename(columns={'instance_id': 'id'})
    key_found_in_od = True

if not key_found_in_od:
    raise ValueError(
        f"Merge key '{merge_on}' not found in source dataset columns (tried renaming): {original_od_cols}")

# Now, unconditionally convert the key in 'od' to string
print(f"Converting '{merge_on}' to string in source data (dtype before: {od[merge_on].dtype})...")
try:
    od[merge_on] = od[merge_on].astype(str)
except Exception as e_astype:
    print(
        f"Warning: Failed to directly convert {merge_on} to string in source data: {e_astype}. Trying apply(str).")
    od[merge_on] = od[merge_on].apply(lambda x: str(x) if pd.notna(x) else x)
print(f" -> dtype after: {od[merge_on].dtype}")

# --- Perform Merge ---
print(
    f"Attempting merge on key '{merge_on}' (type in pred_df: {predictions_df[merge_on].dtype}, type in "
    f"od: {od[merge_on].dtype})...")
merged_df = pd.merge(predictions_df, od, on=merge_on, how='inner')  # NOW THIS SHOULD WORK
print(f"Merged {len(merged_df)} examples.")

df = merged_df

model_id = args.model_id
tokenizer = AutoTokenizer.from_pretrained(model_id)
input_format = args.input_file.split("/")[-1].split("_")[1]
dataset_name = args.input_file.split("/")[-2]
explanations_save_dir = args.output_dir + "/" + model_id.split("/")[-1] + "_" + input_format + "/" + dataset_name

if not os.path.exists(explanations_save_dir):
    os.makedirs(explanations_save_dir)
    print(f"Created directory: {explanations_save_dir}")

# Determine the device to use
if torch.cuda.is_available():
    device = "cuda"  # CUDA (NVIDIA GPU)
elif torch.backends.mps.is_available():
    device = "mps"   # MPS (Apple GPU)
else:
    device = "cpu"   # CPU fallback

attribution_method = args.attribution_method

inseq_model = inseq.load_model(
    model=model_id,
    tokenizer=tokenizer,
    attribution_method=attribution_method,
    device=device,
)

if attribution_method in ["attention", "value_zeroing"]:
    step_scores = []
else:
    step_scores = ["probability"]

for i, instance in tqdm(df.iterrows(), total=len(df)):
    iid = str(instance["instance_id"])
    if args.instance_ids and iid not in args.instance_ids:
        continue
    print(f"Processing instance ID: {instance['instance_id']}, table ID: {instance['table_id']}")

    filename = f"{instance['table_id']}-id-{instance['instance_id']}"
    output_figure_path_proxy = f"{explanations_save_dir}/{filename}-context-{attribution_method}.pdf"
    if os.path.exists(output_figure_path_proxy):
        continue

    pickle_out = f"{explanations_save_dir}/{filename}_dic.pickle"
    if os.path.exists(pickle_out):
        continue

    input_text = instance["input"]
    generated_text = instance["prediction"]
    if not generated_text:
        continue

    if type(input_text) == dict:
        input_text = input_text["content"]

    """
    if "meta-llama" in model_id:
        input_message = [{"role": "user", "content": input_text}]
        input_text = "<s>" + tokenizer.apply_chat_template(input_message,
                                                           tokenize=False,
                                                           add_generation_prompt=True)
        generated_message = [{"role": "assistant", "content": generated_text}]
        generated_text = "<s>" + tokenizer.apply_chat_template(generated_message,
                                                               tokenize=False,
                                                               add_generation_prompt=True)
        skip_special_tokens = False
    else:
        skip_special_tokens = True
    """
    skip_special_tokens = True

    try:
        attribution_output = inseq_model.attribute(
            input_text,
            input_text + generated_text,
            skip_special_tokens=skip_special_tokens,
            clean_special_chars=True,
            step_scores=step_scores,
        )
    except torch.cuda.OutOfMemoryError:
        print("Out of memory for file: {}".format(filename))
        torch.cuda.empty_cache()
        continue

    # Save attribution_output to file
    result = attribution_output.show(display=False, return_html=True)
    dic = attribution_output.get_scores_dicts()

    aggregated_ctx_attributions = {}
    first_generated_token_pos = list(dic[0]["target_attributions"].keys())[0][0]
    for tgt, tgt_attributions in dic[0]["target_attributions"].items():
        for ctx_token, ctx_attribution in tgt_attributions.items():
            aggregated_ctx_attributions[ctx_token] = aggregated_ctx_attributions.get(ctx_token, 0) + ctx_attribution

    if device == "cuda":
        torch.cuda.empty_cache()

    def plot_heatmap(    d,
            colormap_fn=plt.cm.coolwarm,
            figsize=(6, 4),
            fontsize=15,
            alpha_bg=0.4,
            pad=0.2,
            line_spacing=2,
            gap_frac=0.01,
            target="context",
        ):
        """
        d:          dict mapping (pos, token) -> score or -> {…:score}
        colormap_fn: matplotlib colormap
        figsize:    (width, height) in inches
        fontsize:   in points
        alpha_bg:   facecolor alpha
        pad:        boxstyle pad
        line_spacing: multiplier on text height for line spacing
        gap_frac:   extra frac of fig width between tokens
        target:     "context" or "output"
        """
        output_figure_path = f"{explanations_save_dir}/{filename}-{target}-{attribution_method}.pdf"

        # 1) extract + normalize
        toks, scs = [], []
        for (_, tok), raw in sorted(d.items(), key=lambda kv: kv[0][0]):
            # unwrap dicts like {'probability':…}
            if isinstance(raw, dict):
                vals = [v for v in raw.values()
                        if isinstance(v, (int, float, np.generic, torch.Tensor))]
                if not vals: continue
                v = float(vals[0])
            else:
                try:
                    v = float(raw)
                except:
                    continue
            if math.isnan(v): continue
            toks.append(tok);
            scs.append(v)

        if not toks:
            raise ValueError("No valid tokens to plot!")

        arr = np.array(scs)
        norm = (arr - arr.min()) / (arr.max() - arr.min())

        # 2) prep figure & measure dims
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fig_w_px, fig_h_px = fig.bbox.width, fig.bbox.height
        # text height in pixels: fontsize (pts) * dpi / 72 (pts/inch)
        text_h_px = fontsize * fig.dpi / 72
        # line height in axis‐fraction coords
        line_h_frac = (text_h_px * line_spacing) / fig_h_px

        # 3) place tokens
        x_frac = 0.0
        y_frac = 1.0 - (text_h_px / fig_h_px)  # start just below top
        for tok, s in zip(toks, norm):
            tok = escape_for_matplotlib(tok)
            # desired colours
            col = colormap_fn(s)
            face = (col[0], col[1], col[2], alpha_bg)
            edge = (col[0], col[1], col[2], 1.0)

            # draw a temp text at 0,0 just to measure its width
            txt = ax.text(0, 0, tok,
                          fontsize=fontsize,
                          bbox=dict(boxstyle=f"round,pad={pad}",
                                    facecolor=face,
                                    edgecolor=edge,
                                    linewidth=2))
            bb = txt.get_window_extent(renderer=renderer)
            w_px = bb.width
            # remove the measuring text
            txt.remove()

            w_frac = w_px / fig_w_px

            # if it would overflow the right edge, wrap
            if x_frac + w_frac > 1.0:
                x_frac = 0.0
                y_frac -= line_h_frac

            # now draw in final position
            ax.text(x_frac, y_frac, tok,
                    fontsize=fontsize,
                    ha="left", va="top",
                    bbox=dict(boxstyle=f"round,pad={pad}",
                              facecolor=face,
                              edgecolor=edge,
                              linewidth=1))

            # advance x by text width + small gap
            x_frac += w_frac + gap_frac

        ax.margins(0)  # no extra data‐space margins
        ax.set_xlim(0, 1)  # exactly span [0,1] in x
        ax.set_ylim(0, 1)  # exactly span [0,1] in y

        plt.tight_layout(pad=0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(output_figure_path, dpi=300, bbox_inches='tight', pad_inches=0.125)

    plot_heatmap(
        aggregated_ctx_attributions,
        colormap_fn=plt.cm.coolwarm,
        target="context",
    )
    plot_heatmap(
        dic[0]["step_scores"],
        colormap_fn=plt.cm.Greens,
        figsize=(5, 2),
        gap_frac=0.03,
        target="output"
    )
    print()

    """…
    with open(f"{explanations_save_dir}/{filename}_result.html", "w", encoding="utf-8") as f:
        f.write(result)
    

    with open(pickle_out, "wb") as f: # Use 'wb' for writing binary data
        pickle.dump(dic, f)  # Save dic using pickle
    """
