## Interpretability analyses

### Inseq (Text-only)
Models tested:
* [mistralai/Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)
* [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

Example of generating saliency maps for ComTQA-PMC using Mistral-Nemo and constraining the analysis to two instances (IDs 49 and 101):  
```bash
python ./src/evaluation/explain_llm_predictions.py \
  --input_file comtqa_pmc_results.json \
  --model_id mistralai/Mistral-Nemo-Instruct-2407 \
  --source_data_path data/ComTQA_data/comtqa_pmc_updated_2025-03-07 \
  --instance_ids 49 \
  --instance_ids 101
```
Results are in subdirectories of explanations/inseq/

---

MLLM analysis using CC-SHAP currently not supported.
