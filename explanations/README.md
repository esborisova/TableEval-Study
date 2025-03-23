## Results from interpretability analyses

### Inseq (Text-only)
Model: [mistralai/Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)
* mistralai/results_dict_comtqa_pmc_Mistral-Nemo-Instruct-2407_2025-03-12_21_15_02.json
* mistralai/results_dict_logicnlg_Mistral-Nemo-Instruct-2407_2025-03-14_21_19_48.json

```bash
python ./explain_llm_predictions.py --input_file results.json --model_id mistralai/Mistral-Nemo-Instruct-2407
```

### MM-SHAP (Vision-language)
Model: [HuggingFaceM4/Idefics3-8B-Llama3](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3)
* idefics/results_image_comtqa_pmc_Idefics3-8B-Llama3_2025-03-12_20_42_17.json
* idefics/results_image_logicnlg_Idefics3-8B-Llama3_2025-03-13_14_32_09.json

```bash
python ./explain_mllm_predictions.py --input_file results.json --model_id HuggingFaceM4/Idefics3-8B-Llama3
```