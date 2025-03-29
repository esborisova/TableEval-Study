import argparse
import copy
import datasets
import math
import numpy as np
import pandas as pd
import random
import shap
import torch

from evaluation.models import HFModel
from evaluator import generate_prompt


def compute_mm_score(text_length, shap_values):
    """ Compute Multimodality Score. (80% textual, 20% visual, possibly: 0% knowledge). """
    text_contrib = np.abs(shap_values.values[0, 0, :text_length]).sum()
    image_contrib = np.abs(shap_values.values[0, 0, text_length:]).sum()
    text_score = text_contrib / (text_contrib + image_contrib)
    # image_score = image_contrib / (text_contrib + image_contrib) # is just 1 - text_score in the two modalities case
    return text_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True,
                        help="The input file with predictions to process.")
    parser.add_argument("--source_data_path", type=str,
                        default="../../data/ComTQA_data/comtqa_pmc_updated_2025-03-07")
    parser.add_argument("--image_path", type=str,
                        default="../../data/ComTQA_data/pubmed/images/png")
    parser.add_argument("--model_id", type=str,
                        default="google/paligemma-3b-mix-224",
                        help="The model to use.")
    parser.add_argument("--output_dir", type=str,
                        default="../../explanations/mm-shap")
    args = parser.parse_args()

    predictions_file = args.input_file
    df = pd.read_json(predictions_file)
    od = datasets.load_from_disk(args.source_data_path)["test"].to_pandas()

    # First, extract instance_id from the example dict into a new column
    df['instance_id'] = df['example'].apply(lambda x: x['instance_id'])
    # Now merge on the instance_id column
    merged_df = pd.merge(df, od, on='instance_id', how='inner')  # or 'left', 'right', 'outer' as needed

    if "ComTQA_data/fintabnet" in args.image_path:
        from evaluation.tasks.ComTQA.fintabnet.image_parser import parse
    elif "ComTQA_data/pubmed" in args.image_path:
        from evaluation.tasks.ComTQA.pubmed.image_parser import parse
    else:
        raise ValueError('Invalid dataset: {}'.format(args.source_data_path))
    merged_df["parsed_image"] = parse(merged_df.to_records(), image_path=args.image_path)
    print(merged_df.head())

    random.seed(1520)

    model_name = args.model_id

    model = HFModel(
        model_name=model_name,
        model_args={},
        multi_modal=True,
        device="mps",
    )

    max_new_tokens = 1024


    def prepare_shap_input(sample):
        """
        A helper function to prepare the input prompt for SHAP analysis.
        Expects sample to be a tuple/list: [raw_image, text_sample].
        Uses default/dummy values for few_shot_samples and task if not available.
        """
        # Provide defaults for few_shot_samples and task
        few_shot_samples = []  # empty list if none available
        # Create a minimal task configuration. Adjust as needed.
        task = {
            "doc_to_text": lambda s: s,  # identity function: returns the text as-is
            "multi_modal_data": True,
        }
        # Use generate_prompt (or generate_string_prompt) to get the prompt in the same format
        # Here we assume that for multi-modal data, generate_prompt returns something like:
        # [raw_image, full_text_prompt]
        prompt_pair = generate_prompt(
            [sample], few_shot_samples, num_fewshot=0, task=task, prompt_template=False)[0]
        return prompt_pair


    for i, row in merged_df.iterrows():
        raw_image, prompt = row["parsed_image"]
        #prompt = row["parsed_image"][1]

        decoded_outputs, logits = model.forward(
            [(raw_image, prompt)],
            do_sample=True
        )

        inputs = model.processor(
            images=[raw_image],
            text=[model.image_token + prompt],
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        # Generate the outputs to extract the generated token ids.
        outputs = model.model.generate(
            **inputs,
            do_sample=True,
            output_scores=True,
            return_dict_in_generate=True
        )
        # Remove the input portion to obtain only the new tokens.
        output_ids = outputs.sequences[:, inputs["input_ids"].shape[1]:].to("cpu")
        # Move the inputs to CPU for further SHAP analysis.
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

        # Determine the number of text tokens and compute the patching parameters.
        nb_text_tokens = inputs["input_ids"].shape[1]
        nb_special_image_prompt_tokens = 0
        p = int(math.ceil(np.sqrt(nb_text_tokens - nb_special_image_prompt_tokens)))
        patch_size = inputs["pixel_values"].shape[-1] // p
        # Create a tensor of negative image token ids to distinguish image patches.
        image_token_ids = torch.tensor(range(-1, -p ** 2 - 1, -1)).unsqueeze(0)
        # Concatenate image token ids with the text input ids.
        X = torch.cat((image_token_ids, inputs["input_ids"]), 1).unsqueeze(1)


        def custom_masker(mask, x):
            """
            Shap relevant function.
            It gets a mask from the shap library with truth values about which image and text tokens to mask (False)
            and which not (True).
            It defines how to mask the text tokens and masks the text tokens. So far, we don't mask the image, but have
            only defined which image tokens to mask. The image tokens masking happens in get_model_prediction().
            """
            masked_X = x.clone()  # x.shape is (num_permutations, img_length+text_length)
            # never mask out <s> and <image> tokens (makes no sense for the model to work without them)
            # find all ids of masked_X where the value is 1 or 32000 since we are not going to mask these special
            # tokens
            if model_name == "llava_vicuna":
                condition = (masked_X == 1) | (masked_X == 32000) | (masked_X == 29871)
            elif "mplug" in model_name:
                condition = torch.isin(masked_X, torch.tensor([151644, 151645]))  # imstart imend
            else:  # bakllava and llava_mistral specific
                condition = (masked_X == 1) | (masked_X == 32000) | (masked_X == 28705)
            indices = torch.nonzero(condition, as_tuple=False)
            mask[indices[:, 1]] = True
            if "mplug" in model_name:
                mask[
                -nb_text_tokens:-nb_text_tokens + nb_special_image_prompt_tokens] = True  # the first 104 tokens are
                # special image prompting tokens, do not touch them. Image will be later masked on the raw parts
                # according to masking of the special image ids

            # set to zero the image tokens we are going to mask
            image_mask = torch.tensor(mask).unsqueeze(0)
            image_mask[:, -nb_text_tokens:] = True  # do not mask text tokens yet
            masked_X[~image_mask] = 0  # ~mask !!! to zero

            # mask the text tokens (delete them)
            text_mask = torch.tensor(mask).unsqueeze(0)
            text_mask[:, :-nb_text_tokens] = True  # do not do anything to image tokens anymore
            if model_name == "llava_vicuna":
                masked_X[~text_mask] = 903
            elif "mplug" in model_name:
                masked_X[~text_mask] = 220
            else:  # bakllava and llava_mistral specific
                masked_X[~text_mask] = 583
            return masked_X  # .unsqueeze(0)


        def get_model_prediction(x):
            """
            Shap relevant function.
            1. Mask the image pixel according to the specified patches to mask from the custom masker.
            2. Predict the model output for all combinations of masked image and tokens. This is then further passed
            to the shap libary.
            """
            with torch.no_grad():
                token_len = inputs["input_ids"].shape[1]
                # split up the input_ids and the image_token_ids from x (containing both appended)
                input_ids = torch.tensor(x[:, -token_len:])  # text ids
                masked_image_token_ids = torch.tensor(x[:, :-token_len])
                # output_ids.shape is (1, output_length); result.shape is (num_permutations, output_length)
                result = np.zeros((input_ids.shape[0], output_ids.shape[1]))

                # call the model for each "new image" generated with masked features
                for i in range(input_ids.shape[0]):
                    raw_image_array = copy.deepcopy(np.array(raw_image))

                    # pathify the image
                    for k in range(masked_image_token_ids[i].shape[0]):
                        if masked_image_token_ids[i][k] == 0:  # should be zero
                            m = k // p
                            n = k % p
                            raw_image_array[m * patch_size:(m + 1) * patch_size,
                                n * patch_size:(n + 1) * patch_size, :] = 0

                    # Use the same processor call as in the main generation,
                    # ensuring that the text includes the special image token.
                    masked_inputs = model.processor(
                        images=[raw_image_array],
                        text=[model.image_token + " " + prompt],
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    ).to(model.device)

                    # Generate outputs using the model.
                    out = model.model.generate(
                        **masked_inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        output_logits=True,
                        output_scores=True,
                        return_dict_in_generate=True
                    )
                    logits = out.logits[0].detach().cpu().numpy()
                    # extract only logits corresponding to target sentence ids
                    result[i] = logits[0, output_ids]
            return result

        explainer = shap.Explainer(get_model_prediction, custom_masker, max_evals=2 * X.shape[2]+1)
        shap_values = explainer(X)[0]

        if len(shap_values.values.shape) == 2:
            shap_values.values = np.expand_dims(shap_values.values, axis=2)
        print("shap_values: ", shap_values)

        mm_score = compute_mm_score(nb_text_tokens, shap_values)
        print("mm_score: ", mm_score)

        merged_df.loc[i, "shap_values"] = shap_values.values.tolist()  # convert numpy array to list
        merged_df.loc[i, "mm_score"] = mm_score

    merged_df.to_json(f"{args.output_dir}/mm-shap_results_{model_name.replace('/', '_')}.json", orient="records")
    merged_df.to_csv(f"{args.output_dir}/mm-shap_results_{model_name.replace('/', '_')}.csv", index=False)

    # If shap_values is complex, you can also use pickle:
    import pickle
    with open(f"{args.output_dir}/mm-shap_results_{model_name.replace('/', '_')}.pkl", "wb") as f:
        pickle.dump(merged_df, f)
