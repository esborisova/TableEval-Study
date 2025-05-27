import argparse
import datasets
import json
import math
import numpy as np
import os
import pandas as pd
import pickle
import random
import shap
import torch
import traceback # For detailed error printing
from scipy import spatial, stats, special
from sklearn import metrics
from tqdm import tqdm
from PIL import Image

from evaluation.models import HFModel
from evaluator import generate_prompt


def compute_mm_score(num_text_tokens, shap_values):
    """ Compute Multimodality Score based on SHAP values. """
    num_text_tokens = int(num_text_tokens)  # Ensure integer type

    # Check if shap_values.values exists and is numpy array
    if not hasattr(shap_values, 'values') or not isinstance(shap_values.values, np.ndarray):
        print(f"Warning: shap_values.values is missing or not a numpy array. Type: {type(getattr(shap_values, 'values', None))}. Returning MM-Score of 0.5.")
        return 0.5

    # shap_values.values shape is expected to be (1, num_input_tokens, num_output_tokens)
    if len(shap_values.values.shape) != 3:
        print(
            f"Warning: Unexpected shap_values.values shape: {shap_values.values.shape}. "
            f"Expected (1, num_input_tokens, num_output_tokens). Trying to adapt, but result might be incorrect.")
        # Attempt to handle cases where SHAP might return 2D (e.g., if only one output token explained)
        if len(shap_values.values.shape) == 2:
            print(" -> Reshaping 2D SHAP values to 3D assuming single batch and summing over outputs.")
            # This assumes shape (inputs, outputs) or (inputs). Need num_output_tokens.
            # Let's assume it summed outputs -> (batch, inputs). Add output dim=1.
            # This might be incorrect depending on how SHAP aggregated.
            # shap_values.values = np.expand_dims(shap_values.values, axis=0) # Add batch
            shap_values.values = np.expand_dims(shap_values.values, axis=2) # Add output dim
            # Recheck shape
            if len(shap_values.values.shape) != 3:
                 print(" -> Failed to reshape to 3D. Returning MM-Score 0.5.")
                 return 0.5
        else:
            print(" -> Cannot adapt shape. Returning MM-Score 0.5.")
            return 0.5

    if shap_values.values.shape[0] != 1:
        print(f"Warning: Expected batch size of 1 in shap_values, got {shap_values.values.shape[0]}. Using index 0.")

    # Assuming image patch tokens come *before* text tokens in the concatenated input X
    num_input_tokens = shap_values.values.shape[1]
    num_image_tokens = num_input_tokens - num_text_tokens
    if num_image_tokens < 0:
        print(
            f"Error: Calculated negative number of image tokens ({num_image_tokens}). "
            f"Check num_text_tokens ({num_text_tokens}) and shap_values shape ({shap_values.values.shape}). Returning MM-Score 0.5.")
        return 0.5

    image_contrib = np.abs(shap_values.values[0, :num_image_tokens, :]).sum()
    text_contrib = np.abs(shap_values.values[0, num_image_tokens:, :]).sum()

    total_contrib = text_contrib + image_contrib
    if total_contrib == 0 or not np.isfinite(total_contrib):
        print(f"Warning: Total contribution is zero or non-finite ({total_contrib}). Returning MM-Score of 0.5.")
        return 0.5  # Avoid division by zero, return neutral score

    # MM-Score: Proportion of contribution from TEXT
    mm_score = text_contrib / total_contrib
    return mm_score


def aggregate_values_explanation(shap_values, tokenizer, to_marginalize=' Why?', model_family="generic"):
    """ Aggregate SHAP values, marginalizing trailing explanation prompt tokens. """

    # Check if shap_values.values exists and is numpy array
    if not hasattr(shap_values, 'values') or not isinstance(shap_values.values, np.ndarray):
        print(f"Warning: shap_values.values is missing or not a numpy array in aggregate_values_explanation. Type: {type(getattr(shap_values, 'values', None))}. Returning empty array.")
        return np.array([])

    # Ensure 3D shape (handle potential SHAP aggregation issues)
    if len(shap_values.values.shape) != 3:
        print(f"Warning: Unexpected shape {shap_values.values.shape} in aggregate_values_explanation. Attempting reshape.")
        if len(shap_values.values.shape) == 2:
            shap_values.values = np.expand_dims(shap_values.values, axis=2)
            if len(shap_values.values.shape) != 3:
                 print(" -> Failed reshape. Returning empty.")
                 return np.array([])
        else:
            print(" -> Cannot adapt shape. Returning empty.")
            return np.array([])

    # Need to tokenize the marginalization string *without* special tokens added automatically
    marginalize_tokens = tokenizer.encode(to_marginalize, add_special_tokens=False)
    len_to_marginalize = len(marginalize_tokens)

    # Refinement based on model family (Example using Qwen's potential behavior)
    if model_family == "qwen": # Example: Qwen might add prefix space/handle differently
        # Note: Exact tokenization can be tricky, might need inspection of explanation outputs
        # Try tokenizing with space, check if different
        marginalize_tokens_with_space = tokenizer.encode(" " + to_marginalize, add_special_tokens=False)
        # Heuristic: if adding space changes tokenization meaningfully, use that length
        if marginalize_tokens_with_space != marginalize_tokens and len(marginalize_tokens_with_space) > 0:
             print(f"Adjusting len_to_marginalize based on space prefix for {model_family}")
             len_to_marginalize = len(marginalize_tokens_with_space)
        # Fallback if still zero
        if len_to_marginalize == 0 and len(to_marginalize) > 0:
             print(f"Warning: Tokenization of '{to_marginalize}' resulted in 0 tokens. Cannot marginalize.")


    num_output_tokens = shap_values.values.shape[2]
    if len_to_marginalize == 0:
         print("Warning: len_to_marginalize is 0. Aggregating all output tokens.")
         # Proceed as if marginalizing nothing
    elif len_to_marginalize >= num_output_tokens:
        print(
            f"Warning: len_to_marginalize ({len_to_marginalize}) >= num_output_tokens ({num_output_tokens}). "
            f"Aggregating all output tokens instead of marginalizing.")
        len_to_marginalize = 0 # Effectively aggregate all

    if len_to_marginalize == 0: # Aggregating all
        mean_contrib_all = np.mean(shap_values.values[0], axis=1) # Mean over all output tokens
        total_abs_contrib_all = np.abs(mean_contrib_all).sum()
        ratios = mean_contrib_all / (total_abs_contrib_all + 1e-9) * 100
        return ratios
    else: # Marginalize last 'len_to_marginalize' tokens
        print(f"Marginalizing last {len_to_marginalize} output tokens.")
        # Mean contribution over non-marginalized output tokens
        mean_contrib_non_marginalized = np.mean(shap_values.values[0, :, :-len_to_marginalize], axis=1)
        # Total absolute contribution *excluding* marginalized tokens (sum over non-marg outputs and all inputs)
        total_abs_contrib_non_marginalized = np.abs(shap_values.values[0, :, :-len_to_marginalize]).sum()

        # Normalize: contribution / total_abs_contribution (of non-marginalized outputs)
        ratios = mean_contrib_non_marginalized / (total_abs_contrib_non_marginalized + 1e-9) * 100
        return ratios



def aggregate_values_prediction(shap_values):
    """ Aggregate SHAP values for the prediction task (mean over output tokens). """
    # Check if shap_values.values exists and is numpy array
    if not hasattr(shap_values, 'values') or not isinstance(shap_values.values, np.ndarray):
        print(f"Warning: shap_values.values is missing or not a numpy array in aggregate_values_prediction. Type: {type(getattr(shap_values, 'values', None))}. Returning empty array.")
        return np.array([])

    # Ensure 3D shape (handle potential SHAP aggregation issues)
    if len(shap_values.values.shape) != 3:
        print(f"Warning: Unexpected shape {shap_values.values.shape} in aggregate_values_prediction. Attempting reshape.")
        if len(shap_values.values.shape) == 2:
             shap_values.values = np.expand_dims(shap_values.values, axis=2)
             if len(shap_values.values.shape) != 3:
                  print(" -> Failed reshape. Returning empty.")
                  return np.array([])
        else:
            print(" -> Cannot adapt shape. Returning empty.")
            return np.array([])

    if shap_values.values.shape[0] != 1:
        print(f"Warning: Expected batch size of 1, got {shap_values.values.shape[0]}. Using index 0.")


    # Mean contribution over all output tokens for each input token
    mean_values_per_input = np.mean(shap_values.values[0], axis=1)

    # Normalize: contribution / total_absolute_contribution
    total_abs = np.abs(mean_values_per_input).sum()
    if total_abs == 0 or not np.isfinite(total_abs):
         print("Warning: Total absolute contribution for prediction is zero or non-finite. Returning zeros.")
         return np.zeros_like(mean_values_per_input)

    ratios = mean_values_per_input / (total_abs + 1e-9) * 100
    return ratios


def cc_shap_score(ratios_prediction, ratios_explanation):
    """ Calculate consistency metrics between prediction and explanation ratios. """
    # Ensure inputs are numpy arrays
    ratios_prediction = np.asarray(ratios_prediction)
    ratios_explanation = np.asarray(ratios_explanation)

    if ratios_prediction.size == 0 or ratios_explanation.size == 0:
        print("Warning: Empty ratios passed to cc_shap_score. Returning default scores (NaNs).")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    if ratios_prediction.shape != ratios_explanation.shape:
        # This can happen if aggregation failed for one but not the other
        print(
            f"Warning: Shape mismatch in cc_shap_score: Pred {ratios_prediction.shape}, Expl {ratios_explanation.shape}. "
            "Cannot compute scores. Returning NaNs.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Handle potential NaNs or Infs if normalization failed
    if not np.all(np.isfinite(ratios_prediction)) or not np.all(np.isfinite(ratios_explanation)):
        print("Warning: Non-finite values detected in ratios. Replacing NaNs/Infs with 0 for CC-SHAP.")
        ratios_prediction = np.nan_to_num(ratios_prediction)
        ratios_explanation = np.nan_to_num(ratios_explanation)

    # Add small epsilon if all values are zero after nan_to_num
    if np.all(ratios_prediction == 0) or np.all(ratios_explanation == 0):
        print("Warning: All ratio values are zero after cleaning. Returning default scores (0).")
        # Return actual zeros for metrics where 0 implies perfect match (MSE, VarDiff) or no divergence (KL, JS)
        # Return 1 for correlations (cosine, distance) assuming perfect match at zero.
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 # CosineDist, DistCorr, MSE, VarDiff, KL, JS

    # --- Calculate Metrics ---
    try:
        cosine_dist = spatial.distance.cosine(ratios_prediction, ratios_explanation)
    except Exception as e:
        print(f"Warning: Cosine distance failed: {e}. Setting to NaN.")
        cosine_dist = np.nan

    try:
        # Distance correlation requires std dev > 0
        if np.std(ratios_prediction) > 1e-9 and np.std(ratios_explanation) > 1e-9:
             distance_correlation = spatial.distance.correlation(ratios_prediction, ratios_explanation)
        else:
             print("Warning: Zero standard deviation in ratios. Setting distance correlation to NaN.")
             distance_correlation = np.nan
    except Exception as e:
        print(f"Warning: Distance correlation failed: {e}. Setting to NaN.")
        distance_correlation = np.nan

    try:
        mse = metrics.mean_squared_error(ratios_prediction, ratios_explanation)
    except Exception as e:
        print(f"Warning: MSE failed: {e}. Setting to NaN.")
        mse = np.nan

    try:
        var_diff = np.var(ratios_prediction - ratios_explanation)
    except Exception as e:
        print(f"Warning: Variance of difference failed: {e}. Setting to NaN.")
        var_diff = np.nan

    # KL and Jensen-Shannon divergence require probability distributions
    try:
        prob_pred = special.softmax(ratios_prediction)
        prob_expl = special.softmax(ratios_explanation)
        epsilon = 1e-9
        # Ensure no negative probabilities after softmax (shouldn't happen but check)
        if np.any(prob_pred < 0) or np.any(prob_expl < 0):
            print("Warning: Negative values after softmax. Clamping to zero.")
            prob_pred = np.maximum(prob_pred, 0)
            prob_expl = np.maximum(prob_expl, 0)
            # Renormalize if needed
            prob_pred /= (prob_pred.sum() + epsilon)
            prob_expl /= (prob_expl.sum() + epsilon)

        kl_div = stats.entropy(prob_pred + epsilon, prob_expl + epsilon) # Use P, Q order (Expl vs Pred) for KL(P || Q)
        js_div = spatial.distance.jensenshannon(prob_pred, prob_expl, base=2) # JS divergence squared
        js_dist = np.sqrt(js_div) # JS distance

    except Exception as e:
        print(f"Warning: KL/JS divergence calculation failed: {e}. Setting to NaN.")
        kl_div = np.nan
        js_dist = np.nan # Use JS distance

    # Return distances/divergences (lower is better/more consistent)
    return cosine_dist, distance_correlation, mse, var_diff, kl_div, js_dist


def compute_cc_shap(values_prediction, values_explanation, tokenizer,
                    num_patches_p, # Changed name back to reflect p value
                    num_text_tokens_pred, num_text_tokens_expl,
                    marg_pred_str='', marg_expl_str=' Why?', model_family="generic"):
    """ Computes CC-SHAP scores and prepares data for plotting/analysis. """

    # --- Aggregate SHAP values (remains the same) ---
    ratios_prediction = aggregate_values_prediction(values_prediction)
    ratios_explanation = aggregate_values_explanation(values_explanation, tokenizer, marg_expl_str, model_family)

    # --- Handle potential aggregation failures (remains the same) ---
    if ratios_prediction.size == 0 or ratios_explanation.size == 0:
        # ... return NaNs and empty plot info ...
        nan_scores = (np.nan,) * 6
        empty_plot_info = {'error': 'Aggregation failed'} # Simplified error info
        return *nan_scores, empty_plot_info

    # --- Align ratios (remains the same) ---
    if len(ratios_prediction) != len(ratios_explanation):
        # ... warning and truncate ...
        min_len = min(len(ratios_prediction), len(ratios_explanation))
        ratios_prediction = ratios_prediction[:min_len]
        ratios_explanation = ratios_explanation[:min_len]

    # --- Calculate Consistency Scores (remains the same) ---
    cosine_dist, dist_correl, mse, var_diff, kl_div, js_div = cc_shap_score(ratios_prediction, ratios_explanation)

    # --- Prepare Plotting Info ---
    shap_plot_info = {}
    num_image_patches = num_patches_p * num_patches_p  # Calculate from p
    try:
        # Check prediction SHAP values data
        if hasattr(values_prediction, 'data') and isinstance(values_prediction.data, (torch.Tensor, np.ndarray)):
             input_data = values_prediction.data
             if isinstance(input_data, torch.Tensor): input_data = input_data.cpu().numpy()
             if input_data.ndim == 1: input_data = np.expand_dims(input_data, axis=0)

             if input_data.shape[0] > 0:
                 # Get text token IDs (values >= 0) after the image placeholder(s)
                 input_ids_pred_np = input_data[0, num_image_placeholders:] # Use placeholder count
                 input_ids_pred_np = input_ids_pred_np[input_ids_pred_np >= 0].astype(int)
                 input_ids_pred_list = input_ids_pred_np.tolist()
                 text_tokens_pred = [tokenizer.decode([x], skip_special_tokens=False) for x in input_ids_pred_list]
             else: text_tokens_pred = []
        else: text_tokens_pred = []

        # Create image placeholder labels
        image_labels = [f"IMAGE_FEAT_{i}" for i in range(num_image_placeholders)] # Label for the feature block

        # Combine labels
        num_ratios = len(ratios_prediction)
        combined_labels = image_labels + text_tokens_pred

        # Adjust labels length (remains the same)
        if len(combined_labels) > num_ratios: combined_labels = combined_labels[:num_ratios]
        elif len(combined_labels) < num_ratios: combined_labels.extend([f"UNKNOWN_{i}" for i in range(num_ratios - len(combined_labels))])

        shap_plot_info = {
            'input_labels': combined_labels,
            'ratios_prediction': ratios_prediction.astype(float).round(2).tolist(),
            'ratios_explanation': ratios_explanation.astype(float).round(2).tolist(),
            'num_image_patches': num_image_patches,  # Store total patch count
            'num_patches_p': num_patches_p,  # Store p value
            'num_image_placeholders': num_image_placeholders, # Store this info
            'num_text_tokens_pred': num_text_tokens_pred,
            'num_text_tokens_expl': num_text_tokens_expl,
        }

    # ... (error handling for plot info) ...
    except Exception as e_plot:
        print(f"Error preparing SHAP plot info: {e_plot}")
        traceback.print_exc()
        shap_plot_info = {'error': f'Failed to create plot info: {e_plot}'}

    return cosine_dist, dist_correl, mse, var_diff, kl_div, js_div, shap_plot_info


def custom_masker(mask, x, num_patches, num_text_tokens, pad_token_id, image_sequence_ids, bos_id, eos_id):
    """ SHAP masker: Masks patch IDs and text tokens. Protects BOS/EOS/Image sequence. """
    mask_tensor = torch.from_numpy(mask).bool() if isinstance(mask, np.ndarray) else mask.bool()
    # Keep on CPU as x should be CPU
    # mask_tensor = mask_tensor.to(x.device) # x is CPU here

    if x.ndim == 1: x = x.unsqueeze(0)
    masked_X = x.clone() # x is on CPU
    current_mask = mask_tensor

    # Expand mask if necessary
    if mask_tensor.ndim == 1: current_mask = mask_tensor.unsqueeze(0).expand_as(masked_X)
    elif mask_tensor.shape[0] == 1 and masked_X.shape[0] > 1: current_mask = mask_tensor.expand_as(masked_X)
    elif mask_tensor.shape[0] != masked_X.shape[0]:
        print(f"ERROR Masker: Mask batch size mismatch. Using first element.")
        current_mask = mask_tensor[0:1].expand_as(masked_X)

    # --- Create a mask where True means DO NOT MASK ---
    dont_mask_flags = torch.zeros_like(current_mask, dtype=torch.bool)

    # --- Flag Special Text Tokens for preservation ---
    text_token_indices = torch.arange(num_patches, num_patches + num_text_tokens, device=x.device)
    tokens_to_always_keep_ids = []
    if bos_id is not None and bos_id >= 0: tokens_to_always_keep_ids.append(bos_id)
    if eos_id is not None and eos_id >= 0: tokens_to_always_keep_ids.append(eos_id)
    if image_sequence_ids: tokens_to_always_keep_ids.extend(image_sequence_ids)
    tokens_to_always_keep_ids = sorted(list(set(tokens_to_always_keep_ids)))

    original_text_tokens = x[:, num_patches:]
    for keep_id in tokens_to_always_keep_ids:
         keep_mask = (original_text_tokens == keep_id)
         dont_mask_flags[:, num_patches:][keep_mask] = True # Flag text tokens to keep

    # --- Combine SHAP's mask with our preservation flags ---
    # Keep if SHAP wants (current_mask=True) OR if it's a special token (dont_mask_flags=True)
    final_keep_mask = current_mask | dont_mask_flags

    # --- Apply the combined mask ---
    mask_value_text = 0 if pad_token_id == -100 else pad_token_id
    mask_value_img = 0 # Use 0 for masked image patch IDs

    # Mask elements where final_keep_mask is False
    masked_X.masked_fill_(~final_keep_mask, mask_value_text) # Default PAD for all initially
    # Specifically set image patch IDs to 0 if masked
    masked_X[:, :num_patches].masked_fill_(~final_keep_mask[:, :num_patches], mask_value_img)

    # NOTE: We no longer conditionally mask image tokens here. The predictor handles everything.

    return masked_X # Return CPU tensor


def get_model_prediction(x, model, device,
                         original_inputs_cpu, # Full original processor output dict
                         num_image_placeholders,
                         num_text_tokens,
                         target_output_ids, pad_token_id):
    """ SHAP prediction function: Feeds masked inputs directly to the model. """
    if isinstance(x, np.ndarray): x = torch.from_numpy(x)
    if x.ndim == 1: x = x.unsqueeze(0)
    x = x.to('cpu') # Process masking logic on CPU

    num_permutations = x.shape[0]
    num_output_tokens = target_output_ids.shape[1]
    results = np.zeros((num_permutations, num_output_tokens))

    target_output_ids_cpu = target_output_ids.cpu() # Ensure target IDs are on CPU

    # Identify vision-related keys and prepare them on device once if they exist
    vision_keys = [k for k in original_inputs_cpu.keys() if 'pixel' in k or 'image' in k or 'vision' in k] # Broader search
    if 'pixel_values' not in vision_keys and 'pixel_values' in original_inputs_cpu:
        vision_keys.append('pixel_values')
    # Add potential Idefics keys if known, e.g., 'pixel_attention_mask' ?
    # For now, rely on general search

    original_vision_inputs_on_device = {}
    for v_key in vision_keys:
         if v_key in original_inputs_cpu and original_inputs_cpu[v_key] is not None:
              if isinstance(original_inputs_cpu[v_key], torch.Tensor):
                   try:
                        # Use model's dtype for vision features if different from default float32
                        target_dtype = model.dtype # Use the loaded model's dtype
                        original_vision_inputs_on_device[v_key] = original_inputs_cpu[v_key].to(device).to(target_dtype)
                   except Exception as e_move:
                        print(f"Warning: Error moving vision key '{v_key}' to device/dtype: {e_move}. Skipping this key.")
              else:
                   original_vision_inputs_on_device[v_key] = original_inputs_cpu[v_key] # Keep non-tensors

    # --- Print Vision Inputs (keep this debug) ---
    # print("\n--- Vision Inputs Going to Model (when image included) ---")
    # ... (print shapes/types) ...
    # print("-----------------------------------------------------------\n")

    with torch.no_grad():
        batch_size = 16 # Process in batches
        for i_start in tqdm(range(0, num_permutations, batch_size), desc="SHAP Permutations", leave=False):
            i_end = min(i_start + batch_size, num_permutations)
            current_batch_indices = range(i_start, i_end)
            actual_batch_size = len(current_batch_indices)

            # Process items individually to handle conditional vision inputs correctly
            for k, global_idx in enumerate(current_batch_indices):
                current_x = x[global_idx : global_idx + 1, :] # Shape (1, 1 + num_text_tokens)
                # input_ids potentially have PAD tokens where image sequence was
                item_input_ids = current_x[:, num_image_placeholders:].to(torch.long) # Shape [1, SeqLen]
                image_placeholder_value = current_x[0, 0].item()
                include_image = (image_placeholder_value != 0)
                item_attn_mask = (item_input_ids != pad_token_id).long() # Shape [1, SeqLen]
                item_input_len = item_input_ids.shape[1]

                try:
                    # --- REMOVE PROCESSOR CALL ---
                    # batch = processor( # THIS WAS WRONG
                    #     text=masked_input_texts,
                    #     images=current_pixel_values, # Incorrect type
                    #     return_tensors="pt",
                    #     padding=True,
                    #     truncation=True,
                    # ).to(device)

                    # --- CORRECT BATCH CONSTRUCTION ---
                    # Build batch directly for the model using masked IDs and original vision features
                    batch = {
                        "input_ids": item_input_ids.to(device),
                        "attention_mask": item_attn_mask.to(device),
                    }
                    if include_image:
                        for v_key, v_tensor_on_device in original_vision_inputs_on_device.items():
                            batch[v_key] = v_tensor_on_device # Pass original vision features
                    # --- END CORRECTION ---

                    outputs = model(**batch, return_dict=True, output_attentions=False, output_hidden_states=False)

                    # --- Logit Extraction ---
                    item_logits = outputs.logits.detach().cpu()[0] if hasattr(outputs, 'logits') and outputs.logits is not None else None
                    if item_logits is None: raise ValueError("Model output missing logits")
                    target_ids_np = target_output_ids_cpu.numpy()[0]
                    item_max_seq_len = item_logits.shape[0]
                    start_logit_idx = item_input_len - 1
                    available_logits_count = item_max_seq_len - start_logit_idx
                    effective_num_output_tokens = min(num_output_tokens, available_logits_count)

                    if start_logit_idx < 0 or start_logit_idx >= item_max_seq_len or effective_num_output_tokens <= 0 :
                         results[global_idx] = np.zeros(num_output_tokens); continue
                    sliced_logits = item_logits[start_logit_idx : start_logit_idx + effective_num_output_tokens, :]
                    try:
                        logits_for_available_tokens = np.array([sliced_logits[t, target_ids_np[t]] for t in range(effective_num_output_tokens)])
                        results[global_idx, :effective_num_output_tokens] = logits_for_available_tokens
                        if effective_num_output_tokens < num_output_tokens: results[global_idx, effective_num_output_tokens:] = 0.0
                    except IndexError as e_fancy: print(f"ERROR Predict (idx {global_idx}): IndexError: {e_fancy}"); results[global_idx] = np.zeros(num_output_tokens)

                except Exception as e_pred_item:
                     print(f"ERROR Predict (Item idx {global_idx}): {e_pred_item}")
                     traceback.print_exc()
                     results[global_idx] = np.zeros(num_output_tokens)

    return results


def explain_mllm(prompt, raw_image, model_wrapper: HFModel,
                 max_new_tokens=100, target_output_ids=None, p=None,
                 num_evals=None):
    """ Computes SHAP values for a given MLLM prediction. """
    model = model_wrapper.model
    processor = model_wrapper.processor
    tokenizer = model_wrapper.tokenizer
    device = model_wrapper.device
    image_token = model_wrapper.image_token

    # 1. Preprocess input and generate target output if needed
    full_prompt_text = image_token + prompt
    try:
        original_inputs_cpu = processor(
            text=full_prompt_text,
            images=raw_image,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
    except Exception as e:
         print(f"ERROR: Processor failed for prompt: '{full_prompt_text[:100]}...'")
         print(f"Processor Error: {e}")
         raise RuntimeError(f"Processor failed for instance: {e}") from e

    # --- Find the ACTUAL image token ID SEQUENCE from processor output ---
    actual_image_token_ids = []
    input_ids_list = original_inputs_cpu['input_ids'][0].tolist()
    try:
        potential_image_sequence = tokenizer.encode(image_token, add_special_tokens=False)
        # print(f"DEBUG: Encoded '{image_token}' -> {potential_image_sequence}") # Optional
        seq_len = len(potential_image_sequence)
        found_sequence = False
        if seq_len > 0:
            for k in range(len(input_ids_list) - seq_len + 1):
                if input_ids_list[k: k + seq_len] == potential_image_sequence:
                    actual_image_token_ids = potential_image_sequence
                    print(f"Found image token sequence {actual_image_token_ids} matching '{image_token}' at index {k}")
                    found_sequence = True;
                    break
        if not found_sequence: print(f"WARNING: Could not find the ID sequence for '{image_token}'.")
    except Exception as e:
        print(f"ERROR searching for image sequence: {e}")

    # --- Define Special Token IDs (BOS/EOS/PAD) ---
    bos_token_id = getattr(tokenizer, 'bos_token_id', None)
    eos_token_id = getattr(tokenizer, 'eos_token_id', None)
    pad_token_id = getattr(tokenizer, 'pad_token_id', -100)
    if pad_token_id is None: pad_token_id = -100
    print(f"DEBUG: bos={bos_token_id}, eos={eos_token_id}, pad={pad_token_id}")

    # --- Identify unmaskable tokens (BOS, EOS, and Image Sequence) ---
    unmaskable_token_ids = []
    if bos_token_id is not None and bos_token_id >= 0: unmaskable_token_ids.append(bos_token_id)
    if eos_token_id is not None and eos_token_id >= 0: unmaskable_token_ids.append(eos_token_id)
    if actual_image_token_ids:
        unmaskable_token_ids.extend([tid for tid in actual_image_token_ids if tid is not None and tid >= 0])
    else:
        print("WARNING: No image token sequence IDs found to add to unmaskable list.")
    unmaskable_token_ids = sorted(list(set(unmaskable_token_ids)))
    print(f"Identified unmaskable token IDs (incl. image sequence): {unmaskable_token_ids}")

    # --- Prepare for SHAP: Patches + Text Tokens ---
    input_ids_cpu = original_inputs_cpu['input_ids'].cpu()  # Get original IDs on CPU
    pixel_values_cpu = original_inputs_cpu.get('pixel_values')  # Get original features on CPU

    if pixel_values_cpu is None:
        raise ValueError("Original processor output missing 'pixel_values', cannot use patch-based SHAP.")

    num_text_tokens = input_ids_cpu.shape[1]

    # Calculate p for patches (heuristic) - Use TEXT tokens ONLY for calculation
    # This requires careful thought - does Qwen's feature count relate to patches?
    # Let's start with the original heuristic based on text tokens.
    # We might need to adjust 'p' based on the feature dimension if this fails.
    if p is None:
        p = int(math.ceil(np.sqrt(max(1, num_text_tokens))))  # Heuristic based on text length
        print(f"Calculated p={p} based on {num_text_tokens} text tokens.")
        if p == 0: p = 1
    else:
        print(f"Using provided p={p}")

    # Determine patch size based on RAW IMAGE dimensions
    img_height, img_width = raw_image.height, raw_image.width  # Use PIL image size
    patch_size_h = img_height // p
    patch_size_w = img_width // p
    if patch_size_h == 0 or patch_size_w == 0:
        raise ValueError(
            f"Calculated patch size is zero ({patch_size_h}x{patch_size_w}) for p={p} and image size {img_height}x{img_width}.")

    num_patches = p * p
    # Create patch IDs (-1 to -num_patches)
    image_patch_ids = torch.tensor(range(-1, -num_patches - 1, -1), dtype=torch.long).unsqueeze(
        0)  # Shape (1, num_patches)

    # Create the combined input X for SHAP: (patch_ids, text_input_ids)
    # Move input_ids_cpu just before concat if needed, ensure both CPU
    X = torch.cat((image_patch_ids.cpu(), input_ids_cpu),
                  dim=1)  # Shape (1, num_patches + num_text_tokens) - Ensure on CPU for SHAP library
    print(f"DEBUG: SHAP input X shape: {X.shape}")

    # --- Define SHAP Helper Functions ---

    # Masker needs to know patch/text split
    shap_masker = lambda mask, x: custom_masker(mask, x,
                                                num_patches=num_patches,
                                                #num_image_placeholders=num_image_placeholders,  # Use placeholder count
                                                num_text_tokens=num_text_tokens,
                                                pad_token_id=pad_token_id,
                                                image_sequence_ids=actual_image_token_ids,  # Found sequence
                                                bos_id=bos_token_id,
                                                eos_id=eos_token_id)

    # Predictor needs info to reconstruct inputs from scratch
    shap_predictor = lambda x: get_model_prediction(x, model=model, device=device,
                                                    original_inputs_cpu=original_inputs_cpu,
                                                    # Pass the original processed inputs
                                                    num_image_placeholders=num_image_placeholders,
                                                    # Use placeholder count
                                                    num_text_tokens=num_text_tokens,
                                                    target_output_ids=target_output_ids,  # Target for explanation
                                                    pad_token_id=pad_token_id)

    # --- Run SHAP Explainer ---
    if num_evals is None:
        num_features = X.shape[1]  # Now 1 + num_text_tokens
        max_evals = 2 * num_features + 2048
    else:
        max_evals = num_evals

    print(f"Running SHAP Explainer with {num_features} features and max_evals={max_evals}...")
    explainer = shap.Explainer(shap_predictor, shap_masker)

    try:
        shap_values = explainer(X, max_evals=max_evals)

        # Ensure shap_values has the expected 3D structure
        if hasattr(shap_values, 'values') and isinstance(shap_values.values, np.ndarray):
             if len(shap_values.values.shape) == 2:
                  print("Warning: SHAP values returned 2D, expanding output dimension.")
                  shap_values.values = np.expand_dims(shap_values.values, axis=2)
             # Ensure batch dimension exists (should be 1 based on input X)
             if len(shap_values.values.shape) == 3 and shap_values.values.shape[0] != 1 :
                 # This case shouldn't happen if X input was (1, F)
                 print(f"Warning: SHAP values batch dim {shap_values.values.shape[0]} != 1. Taking first element.")
                 shap_values.values = shap_values.values[0:1] # Take first batch element
                 # Update base_values and data if they exist and have batch dim
                 if hasattr(shap_values, 'base_values') and isinstance(shap_values.base_values, np.ndarray) and shap_values.base_values.ndim > 0 and shap_values.base_values.shape[0] > 1:
                     shap_values.base_values = shap_values.base_values[0:1]
                 if hasattr(shap_values, 'data') and isinstance(shap_values.data, np.ndarray) and shap_values.data.ndim > 1 and shap_values.data.shape[0] > 1:
                      shap_values.data = shap_values.data[0:1]

        else:
             print("Warning: SHAP explanation did not produce expected .values attribute.")
             # Create dummy Explanation object to avoid downstream errors
             dummy_values = np.zeros((1, X.shape[1], target_output_ids.shape[1]))
             dummy_base_values = np.zeros(target_output_ids.shape[1])
             shap_values = shap.Explanation(values=dummy_values, base_values=dummy_base_values, data=X.cpu().numpy())


    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        traceback.print_exc() # Print full traceback for SHAP errors
        print("Attempting with increased max_evals or different settings might help.")
        # Provide dummy values to allow script continuation
        dummy_values = np.zeros((1, X.shape[1], target_output_ids.shape[1]))
        dummy_base_values = np.zeros(target_output_ids.shape[1])
        # Ensure data passed to Explanation is NumPy
        dummy_data = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        shap_values = shap.Explanation(values=dummy_values, base_values=dummy_base_values, data=dummy_data)

    # --- Calculate MM-Score ---
    mm_score = compute_mm_score(num_text_tokens, shap_values)

    p_placeholder = 1  # Representing the single image block

    return shap_values, mm_score, p, num_text_tokens, target_output_ids


# --- Prompting Functions ---

def create_prediction_prompt(sample, task):
    """ Creates the initial prompt for prediction using generate_prompt """
    # Use the lambda that returns [image, text] for multi-modal
    prompt_pair = generate_prompt(
        [sample], few_shot_samples=[], num_fewshot=0, task=task, prompt_template=False)[0]
    if not isinstance(prompt_pair, (list, tuple)) or len(prompt_pair) != 2:
        raise ValueError(f"generate_prompt returned unexpected format: {prompt_pair}")
    return prompt_pair[1] # Return only the text part


def create_explanation_prompt(original_prompt, prediction_text, model_family="generic"):
    """ Creates a prompt asking the model to explain its prediction. """
    # Basic explanation request
    explanation_request = "Why did you generate the previous response? Please explain your reasoning step by step."

    # Use model-specific chat templating if available, otherwise basic format
    # Qwen format might look something like this (needs verification):
    if model_family == "qwen":
         # This is a guess - check Qwen docs for exact multi-turn chat format
         # It might involve system prompts or specific role tags
         full_expl_prompt = (
             f"User: {original_prompt}\n"
             f"Assistant: {prediction_text}\n"
             f"User: {explanation_request}\n"
             f"Assistant:"
         )
         # Alternatively, the processor might handle templating if text is passed differently
    else: # Generic format
        full_expl_prompt = f"{original_prompt}\nASSISTANT: {prediction_text}\nUSER: {explanation_request}\nASSISTANT:"

    return full_expl_prompt


# --- Serialization ---
def serialize_shap(shap_expl):
    """ Serializes SHAP explanation object to a dictionary for JSON storage. """
    # Handle potential None values gracefully
    values = shap_expl.values.tolist() if hasattr(shap_expl, 'values') and shap_expl.values is not None else None
    base_values = shap_expl.base_values.tolist() if hasattr(shap_expl, 'base_values') and shap_expl.base_values is not None else None
    data = shap_expl.data.tolist() if hasattr(shap_expl, 'data') and shap_expl.data is not None else None
    return {
        "values": values,
        "base_values": base_values,
        "data": data,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file with 'example', 'prediction' fields.")
    parser.add_argument("--source_data_path", type=str, default="../../data/ComTQA_data/comtqa_pmc_updated_2025-03-07")
    parser.add_argument("--image_path", type=str, default="../../data/ComTQA_data/pubmed/images/png")
    # parser.add_argument("--model_id", type=str, default="google/paligemma-3b-mix-224")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="The model to use.") # Changed default
    parser.add_argument("--model_family", type=str, default="qwen", help="Specify model family ('qwen', 'llava', 'paligemma', 'generic').") # Changed default
    parser.add_argument("--output_dir", type=str, default="../../explanations/mm-shap")
    parser.add_argument("--subset_size", type=int, default=10, help="Number of examples to process.")
    parser.add_argument("--max_new_tokens_pred", type=int, default=50, help="Max new tokens for original prediction.")
    parser.add_argument("--max_new_tokens_expl", type=int, default=100, help="Max new tokens for explanation generation.")
    parser.add_argument("--shap_num_evals", type=int, default=None, help="Max evaluations for SHAP (default: 2*N+2048).")
    parser.add_argument("--save_shap_values", action="store_true", help="Save detailed SHAP values (large files).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # --- Set Seed ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    elif torch.backends.mps.is_available(): torch.mps.manual_seed(args.seed) # Seed MPS if available

    # --- Load Data ---
    print("Loading data...")
    try:
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
            f"Attempting merge on key '{merge_on}' (type in pred_df: {predictions_df[merge_on].dtype}, type in od: {od[merge_on].dtype})...")
        merged_df = pd.merge(predictions_df, od, on=merge_on, how='inner')  # NOW THIS SHOULD WORK
        print(f"Merged {len(merged_df)} examples.")
        # ... rest of checks and error handling ...

    except Exception as e:
        print(f"Error loading or merging data: {e}")
        traceback.print_exc()  # Make sure traceback is imported
        exit(1)

    # --- Select Image Parser ---
    print("Selecting image parser...")
    # Simplified parser selection logic
    if "ComTQA_data/fintabnet" in args.source_data_path:
        from evaluation.tasks.ComTQA.fintabnet.image_parser import parse as parse_func
    elif "ComTQA_data/pubmed" in args.source_data_path:
        from evaluation.tasks.ComTQA.pubmed.image_parser import parse as parse_func
    elif "LogicNLG" in args.source_data_path:
        from evaluation.tasks.LogicNLG.image_parser import parse as parse_func
    else:
        print(f"Warning: Unknown dataset path '{args.source_data_path}'. Using default/dummy parser logic if available.")
        # Provide a dummy parser or raise error if no default
        # def dummy_parser(records, image_path):
        #     print("Using dummy image parser.")
        #     return [[Image.new('RGB', (60, 30), color = 'red'), "Dummy prompt"] for _ in records]
        # parse_func = dummy_parser
        raise ValueError('Invalid or unconfigured dataset path for image parsing: {}'.format(args.source_data_path))


    # --- Parse Images ---
    print("Parsing images...")
    try:
        # Convert dataframe rows to list of records for parser
        records = merged_df.to_dict('records')
        parsed_data = parse_func(records, image_path=args.image_path) # Pass records
        if len(parsed_data) != len(merged_df):
            raise ValueError(f"Parser returned {len(parsed_data)} items, expected {len(merged_df)}")
        merged_df['parsed_image_obj'] = [item[0] for item in parsed_data]
        merged_df['parsed_text_prompt'] = [item[1] for item in parsed_data]
    except Exception as e:
        print(f"Error parsing images: {e}")
        traceback.print_exc()
        exit(1)


    # --- Sort and Subsample ---
    def compute_sort_key(row):
        raw_image = row['parsed_image_obj']
        prompt = row['parsed_text_prompt']
        prompt_length = len(prompt) if isinstance(prompt, str) else 0
        pixel_count = raw_image.size[0] * raw_image.size[1] if isinstance(raw_image, Image.Image) else 0
        return prompt_length + pixel_count

    merged_df["sort_key"] = merged_df.apply(compute_sort_key, axis=1)
    merged_df = merged_df.sort_values("sort_key", ascending=True)
    merged_df = merged_df.head(args.subset_size).copy()
    print(f"Processing subset of {len(merged_df)} examples.")

    # --- Load Model ---
    print(f"Loading model: {args.model_id}")
    if torch.cuda.is_available(): device = "cuda:0"
    elif torch.backends.mps.is_available(): device = "mps"
    else: device = "cpu"
    print(f"Using device: {device}")

    # Adjust dtype based on device
    model_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    print(f"Using model dtype: {model_dtype}")

    try:
        model_wrapper = HFModel(
            model_name=args.model_id,
            model_args={"torch_dtype": model_dtype, "trust_remote_code": True}, # Needed for some models like Qwen
            multi_modal=True,
            device=device,
        )
        # Ensure processor and tokenizer are loaded
        if not hasattr(model_wrapper, 'processor') or model_wrapper.processor is None:
             raise AttributeError("HFModel failed to load processor.")
        if not hasattr(model_wrapper, 'tokenizer') or model_wrapper.tokenizer is None:
             # Often processor contains the tokenizer
             if hasattr(model_wrapper.processor, 'tokenizer'):
                  model_wrapper.tokenizer = model_wrapper.processor.tokenizer
                  print("Assigned processor.tokenizer to model_wrapper.tokenizer")
             else:
                  raise AttributeError("HFModel failed to load tokenizer and processor has no tokenizer attribute.")

        # Ensure padding token is set (use EOS if PAD is None)
        if model_wrapper.tokenizer.pad_token_id is None:
            model_wrapper.tokenizer.pad_token_id = model_wrapper.tokenizer.eos_token_id
            print(f"Set tokenizer pad_token_id to eos_token_id: {model_wrapper.tokenizer.pad_token_id}")
        # Set model's pad token id if needed
        if hasattr(model_wrapper.model.config, "pad_token_id") and model_wrapper.model.config.pad_token_id is None:
             model_wrapper.model.config.pad_token_id = model_wrapper.tokenizer.pad_token_id
             print(f"Set model.config.pad_token_id to: {model_wrapper.model.config.pad_token_id}")


    except Exception as e_load:
         print(f"FATAL: Failed to load model {args.model_id}: {e_load}")
         traceback.print_exc()
         exit(1)


    # --- Prepare Task Dictionary for generate_prompt ---
    minimal_task = {
        "doc_to_text": lambda docs: [ # Lambda processes list of docs
            [doc.get('parsed_image_obj'), doc.get('parsed_text_prompt', '')]
            for doc in docs
        ],
        "multi_modal_data": True,
    }

    # --- Main Processing Loop ---
    results_list = []
    for i, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Calculating CC-SHAP"):
        instance_id = row.get(merge_on, f"index_{i}")
        print(f"\n--- Processing Instance: {instance_id} ---")
        raw_image = row['parsed_image_obj']
        if not isinstance(raw_image, Image.Image):
            print(f"Warning: Invalid image object for instance {instance_id}. Skipping.")
            results_list.append({merge_on: instance_id, 'error': 'Invalid image object'})
            continue

        sample_dict = row.to_dict()
        current_results = {merge_on: instance_id}

        try:
            # 0. Create Prediction Prompt
            print("Creating prediction prompt...")
            if 'parsed_text_prompt' not in sample_dict or not isinstance(sample_dict['parsed_text_prompt'], str):
                 raise ValueError("'parsed_text_prompt' missing or not a string in sample_dict")
            if 'parsed_image_obj' not in sample_dict or not isinstance(sample_dict['parsed_image_obj'], Image.Image):
                 raise ValueError("'parsed_image_obj' missing or not an Image in sample_dict")

            prediction_prompt = create_prediction_prompt(sample_dict, minimal_task)
            original_prediction_text = row['prediction']

            # 1. Explain the Original Prediction
            print("Explaining original prediction...")
            # Tokenize prediction to get target IDs
            # Use add_special_tokens=False as prediction shouldn't have them
            target_tokens = model_wrapper.tokenizer(original_prediction_text, return_tensors='pt', add_special_tokens=False).input_ids

            if target_tokens.shape[1] == 0:
                print(f"Warning: Tokenization of prediction resulted in 0 tokens for '{original_prediction_text}'. Using max_new_tokens_pred.")
                target_tokens = None
                max_pred_tokens_for_shap = args.max_new_tokens_pred
            else:
                max_pred_tokens_for_shap = target_tokens.shape[1] # Use actual length for explanation consistency
                target_tokens = target_tokens.to('cpu') # Move target tokens to CPU for SHAP

            shap_values_pred, mm_score_pred, p_used, n_text_pred, explained_pred_ids = explain_mllm(
                prompt=prediction_prompt,
                raw_image=raw_image,
                model_wrapper=model_wrapper,
                max_new_tokens=max_pred_tokens_for_shap, # Max tokens for *generating* target if needed
                target_output_ids=target_tokens,        # Actual tokens to explain
                p=None, # Calculate p automatically first time
                num_evals=args.shap_num_evals
            )
            current_results['mm_score_prediction'] = mm_score_pred
            current_results['num_patches_p'] = p_used
            current_results['num_text_tokens_pred'] = n_text_pred
            current_results['num_output_tokens_pred'] = explained_pred_ids.shape[1]
            if args.save_shap_values:
                try:
                    current_results['shap_values_prediction'] = json.dumps(serialize_shap(shap_values_pred))
                except Exception as e_json:
                     print(f"Warning: Could not serialize prediction SHAP values: {e_json}")
                     current_results['shap_values_prediction'] = None


            # 2. Explain the Explanation Task
            print("Explaining the explanation task...")
            explanation_prompt_text = create_explanation_prompt(prediction_prompt, original_prediction_text, args.model_family)

            num_image_placeholders = 1  # Use 1 for the new approach

            shap_values_expl, mm_score_expl, _, n_text_expl, explained_expl_ids = explain_mllm(
                prompt=explanation_prompt_text,
                raw_image=raw_image,
                model_wrapper=model_wrapper,
                max_new_tokens=args.max_new_tokens_expl, # Max tokens for *generating* the explanation
                target_output_ids=None,                 # Generate explanation dynamically
                p=p_used, # Reuse p from prediction explanation
                num_evals=args.shap_num_evals
            )
            current_results['mm_score_explanation'] = mm_score_expl
            current_results['num_text_tokens_expl'] = n_text_expl
            current_results['num_output_tokens_expl'] = explained_expl_ids.shape[1]
            if args.save_shap_values:
                 try:
                     current_results['shap_values_explanation'] = json.dumps(serialize_shap(shap_values_expl))
                 except Exception as e_json:
                      print(f"Warning: Could not serialize explanation SHAP values: {e_json}")
                      current_results['shap_values_explanation'] = None


            # 3. Compute CC-SHAP Scores
            print("Computing CC-SHAP scores...")
            marg_pred_str = "" # Assume no marginalization for prediction task output
            # Define marginalization string for explanation task output based on prompt structure
            # This should match the end of the explanation prompt *before* the model starts generating
            # If create_explanation_prompt ends with "...ASSISTANT:", this might be empty ""
            # If it includes the question "Why did you ... step.\nASSISTANT:", marginalize that.
            # Adjust based on actual prompt structure used for explanation SHAP run
            marg_expl_str = " Why did you generate the previous response? Please explain your reasoning step by step.\nASSISTANT:" # Match create_explanation_prompt

            cosine_dist, dist_correl, mse, var_diff, kl_div, js_div, shap_plot_info = compute_cc_shap(
                values_prediction=shap_values_pred,
                values_explanation=shap_values_expl,
                tokenizer=model_wrapper.tokenizer,
                num_patches_p=p_used, # Pass p value used
                num_text_tokens_pred=n_text_pred,
                num_text_tokens_expl=n_text_expl,
                marg_pred_str=marg_pred_str,
                marg_expl_str=marg_expl_str,
                model_family=args.model_family
            )

            # Store CC-SHAP results (lower distance/divergence = better consistency)
            current_results['cc_shap_cosine_distance'] = cosine_dist
            current_results['cc_shap_distance_correlation'] = dist_correl # This is 1-corr, so lower is better
            current_results['cc_shap_mse'] = mse
            current_results['cc_shap_var_diff'] = var_diff
            current_results['cc_shap_kl_divergence'] = kl_div
            current_results['cc_shap_js_distance'] = js_div
            try:
                current_results['shap_plot_info'] = json.dumps(shap_plot_info) # Store aggregated ratios/labels
            except Exception as e_json:
                 print(f"Warning: Could not serialize plot info: {e_json}")
                 current_results['shap_plot_info'] = None


            print(f"Instance {instance_id} Results: MM-Pred={mm_score_pred:.3f}, MM-Expl={mm_score_expl:.3f}, "
                  f"CosineDist={current_results.get('cc_shap_cosine_distance', float('nan')):.3f}")

        except Exception as e:
            print(f"Error processing instance {instance_id}: {e}")
            traceback.print_exc() # Print full traceback for instance error
            current_results['error'] = traceback.format_exc() # Store full error traceback

        results_list.append(current_results)

        # Optional: Save intermediate results
        # if (i + 1) % 5 == 0: # Save every 5 iterations
        #     print(f"Saving intermediate results at iteration {i}...")
        #     temp_df = pd.DataFrame(results_list)
        #     output_basename = f"cc_shap_results_{args.model_id.replace('/', '_')}_subset{args.subset_size}"
        #     temp_df.to_json(f"{args.output_dir}/{output_basename}_intermediate.json", orient="records", indent=2)


    # --- Save Final Results ---
    print("Saving final results...")
    if not results_list:
        print("WARNING: No results were generated.")
        exit()

    results_df = pd.DataFrame(results_list)
    if merge_on not in results_df.columns:
         print(f"ERROR: Merge key '{merge_on}' missing in results_df columns: {results_df.columns}. Cannot merge final results.")
         # Save results_df directly
         results_df_path = f"{args.output_dir}/PARTIAL_cc_shap_results_NO_MERGE.json"
         print(f"Saving partial results to: {results_df_path}")
         results_df.to_json(results_df_path, orient="records", indent=2)
         exit()


    # Merge results back into the original subset DataFrame
    final_df = pd.merge(merged_df, results_df, on=merge_on, how='left')

    os.makedirs(args.output_dir, exist_ok=True)
    output_basename = f"cc_shap_results_{args.model_id.replace('/', '_')}_subset{len(final_df)}"
    json_path = os.path.join(args.output_dir, f"{output_basename}.json")
    csv_path = os.path.join(args.output_dir, f"{output_basename}.csv")
    pkl_path = os.path.join(args.output_dir, f"{output_basename}.pkl")

    # Drop non-serializable columns before saving if necessary
    cols_to_drop_before_save = ['parsed_image_obj', 'example'] # Example columns to drop
    final_df_save = final_df.drop(columns=[col for col in cols_to_drop_before_save if col in final_df.columns])

    print(f"Saving JSON to: {json_path}")
    try:
        final_df_save.to_json(json_path, orient="records", indent=2, force_ascii=False)
    except Exception as e_save:
        print(f"Error saving final JSON: {e_save}. Trying CSV/Pickle.")
        traceback.print_exc()


    print(f"Saving CSV to: {csv_path}")
    try:
        final_df_save.to_csv(csv_path, index=False)
    except Exception as e_save:
        print(f"Error saving final CSV: {e_save}")

    try:
        print(f"Saving Pickle to: {pkl_path}")
        # Use original final_df for pickle if you want to keep complex objects
        with open(pkl_path, "wb") as f:
            pickle.dump(final_df, f)
    except Exception as e_save:
        print(f"Warning: Could not save results as pickle: {e_save}")

    print("Processing finished.")