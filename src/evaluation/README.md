# Table Understanding Evaluation Pipeline

The Table Understanding Evaluation Pipeline provides a comprehensive framework for evaluating multimodal large language models (LLMs) on various table-understanding benchmarks. The pipeline supports custom evaluation tasks and integrates seamlessly with Huggingface models or local LLM implementations, as well as commercial models via API calls using liteLLM. In the `/task` folder, we provide predefined experiment setups for all published benchmarks. 

## Features
- **Supports Multiple Tasks**: Evaluate your model on a wide variety of predefined table-understanding benchmarks.
- **Flexible Configuration**: Customize model arguments, evaluation tasks, few-shot examples, batch sizes, and device settings.
- **Extensible**: Add your own tasks with a simple YAML configuration.
- **Reproducible**: Set seeds for consistent and reproducible evaluations.
- **Output Logging**: Save model predictions and evaluation scores in JSON format and the logits in HDF5 format.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/esborisova/Table-Understanding-Evaluation-Study.git
   cd Table-Understanding-Evaluation-Study
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
Run the evaluation pipeline using the `src/evaluation/run.py` script with customizable arguments:

### Script Arguments
- **`--model_name`**: Specify the Huggingface model ID or the local path of the LLM to evaluate (e.g., `google/gemma-2-2b-it`, `EleutherAI/pythia-160m`).
- **`--tasks`**: Comma-separated task names as defined in the YAML files (e.g., `task1,task2`, `latex_scigen,html_scigen`).
- **`--model_args`**: String with dict containing arguments for the model (e.g., '{"load_in_8bit": True, "use_cache": True}').
- **`--num_fewshot`**: Number of few-shot examples to use for all tasks (default: task-specific value from the YAML file).
- **`--batch_size`**: Batch size for evaluation.
- **`--device`**: Device to run the evaluation (e.g., `cuda`, `cuda:0`, `cpu`).
- **`--output_path`**: Path to save the outputs (default: `./output`).
- **`--seed`**: Set the seed for reproducibility.
- **`--log_samples`**: If set to `True`, saves model predictions along with scores. Default: `False`
- **`--log_logits`**: If set to `True`, saves logits. Default: `False`
- **`--use_chat_template`**: If set to `True`, LLMs pre-defined chat template is applied to the prompt. Default: `False`
#### Multi-Modal LLM Evaluation:
- **`--multi_modal or -mm`**: If set to `True`, tasks evaluate on multi-modal LLMs. Default: `False`
- **`--image_special_token`**: Changing the special token for the multi_modal LLMs. The default is <image>. 
#### API Model Evaluation:
- **`--api_model`**: If set to `True`, the model will be evaluated using API calls via liteLLM. Default: `False`
- **`--api_key`**: set the API key for the LLM. It is recommended to set the API key manually with `os.environ["GEMINI_API_KEY"] = "your-api-key"`

### Predefined Tasks
- **ComTQA**:
   - **Fintabnet**:

      | **Format**   | **Image**     | **Dictonary**     | **HTML**     | **Latex**     | **XML**     | **Dictonary Spacylayout**     | **HTML Spacylayout**     | **Latex Spacylayout**     | **XML Spacylayout**     |
      |--------------|---------------|-------------------|--------------|---------------|-------------|-------------|---------------|-------------|-------------|
      | **Taks Name**| `image_comtqa_fin`|`dict_comtqa_fin`  | `html_comtqa_fin`   |  `latex_comtqa_fin`     |`xml_comtqa_fin`     |`dict_spacy_comtqa_fin`     |`html_spacy_comtqa_fin`     |`latex_spacy_comtqa_fin`     |`xml_spacy_comtqa_fin`     |
   - **Pubmed**:

     | **Format**   | **Image**     | **Dictonary**     | **HTML**     | **Latex**     | **XML**     |
     |--------------|---------------|-------------------|--------------|---------------|-------------|
     | **Taks Name**| `image_comtqa_pmc`      |      `dict_comtqa_pmc`     | `html_comtqa_pmc`     |  `latex_comtqa_pmc`     |`xml_comtqa_pmc`     |
- **Logic2Text**:
  
  | **Format**   | **Image**     | **Dictonary**     | **HTML**     | **Latex**     | **XML**     |
  |--------------|---------------|-------------------|--------------|---------------|-------------|
  | **Taks Name**| `image_logic2text`      |      `dict_logic2text`     | `html_logic2text`     |  `latex_logic2text`     |`xml_logic2text`     |
- **LogicNLG**:

  | **Format**   | **Image**     | **Dictonary**     | **HTML**     | **Latex**     | **XML**     |
  |--------------|---------------|-------------------|--------------|---------------|-------------|
  | **Taks Name**| `image_logicnlg`      |      `dict_logicnlg`     | `html_logicnlg`     |  `latex_logicnlg`     |`xml_logicnlg`     |
- **SciGen**:

  | **Format**   | **Image**     | **Dictonary**     | **HTML**     | **Latex**     | **XML**     |
  |--------------|---------------|-------------------|--------------|---------------|-------------|
  | **Taks Name**| `image_scigen`      |      `dict_scigen`     | `html_scigen`     |  `latex_scigen`     |`xml_scigen`     |
- **numericNLG**:

  | **Format**   | **Image**     | **Dictonary**     | **HTML**     | **Latex**     | **XML**     |
  |--------------|---------------|-------------------|--------------|---------------|-------------|
  | **Taks Name**| `image_numericnlg`      |      `dict_numericnlg`     | `html_numericnlg`     |  `latex_numericnlg`     |`xml_numericnlg`     |

### Example Command
For LLMs:

```bash
python run.py \
    --model_name EleutherAI/pythia-160m \
    --model_args pretrained=EleutherAI/pythia-160m,dtype=float32 \
    --tasks dict_scigen,dict_numericnlg \
    --num_fewshot 5 \
    --batch_size 16 \
    --device cuda:0 \
    --seed 42 \
    --output_path ./evaluation_results \
    --log_samples True \
    --log_logits True \
```

For MLLMs:

```bash
python run.py \
    --model_name google/paligemma-3b-mix-224 \
    --tasks image_logicnlg,image_numericnlg \
    --num_fewshot 0 \
    --batch_size 1 \
    --device cuda:0 \
    --seed 42 \
    --output_path ./evaluation_results \
    --log_samples True \
    -mm \
    --log_logits True \
```

---

## Task Configuration
You can add new tasks by creating a YAML file in the `tasks` folder. For better organization, you can create subfolders for datasets.

The folder structure is as follows:

```plaintext
Project/
└── tasks/
    └── dataset_name/
        ├── default.yaml
        ├── image_version.yaml
        └── latex_version.yaml
```

### YAML File Structure
```yaml
task_name: <Task Name>  # Unique identifier for the task
path: <Dataset Path>  # Local path or Huggingface dataset ID
test_split: <Test Split Name>  # e.g., 'test', 'validation'
validation_split: <Validation Split Name>  # Optional
train_split: <Train Split Name>  # Optional
num_fewshot: <Default Number of Few-Shots>  # Optional
ignore_columns: Comma-separated column names that should be considered when filtering Nones (e.g., `table_id,paper_id`). If not initialized no columns are considered
save_columns: Comma-separated column names that should be considered when logging the samples (e.g., `table_id,paper_id`). If not initialized all columns are considered
instruction: <Task Instruction>  # e.g., 'Describe the following table' (Optional)
doc_to_text: <Prompt Template>  # e.g., "{{caption}} {{row_headers}} {{column_headers}}" (Jinja2 format)
doc_to_target: <Target Column>  # The column containing the reference answer
multi_modal_data: <True> # Optional, for running MLLMs
metric_list:  # List of metrics for evaluation
  - <Metric1>
  - <Metric2>
```

For MLLMs, a function for parsing images is required. An example is here: [image_parser.py](https://github.com/esborisova/Table-Understanding-Evaluation-Study/blob/evaluation_script/src/evaluation/tasks/ComTQA/fintabnet/image_parser.py)


### Example YAML
For LLMs:

```yaml
task_name: numericnlg_default
path: kasnerz/numericnlg
test_split: test
validation_split: validation
train_split: train
num_fewshot: 0
save_columns: instance_id,table_id,image_name
instruction: ""
doc_to_text: "{{caption}} {{row_headers}} {{column_headers}}"
doc_to_target: description
metric_list:
  - bleu
  - rougeL 
  - meteor
  - PARENT
```

For MLLMs:

```yaml
task_name: image_comtqa_fin
path: ComTQA_data/comtqa_fin
test_split: test
validation_split: validation
train_split: train
num_fewshot: 0
save_columns: instance_id,table_id,image_name
instruction: ""
doc_to_text: !function image_parser.parse
doc_to_target: answer
multi_modal_data: True
metric_list:
  - accuracy
  - f1
  - bleu
  - rougeL
  - meteor
  - bleurt
  - bleu1
  - bleu2
  - bleu3
  - bleu4
  - rougeL
  - rougeLsum
```

---

## Metrics

This repository supports a variety of metrics for evaluating predictions against references. These metrics are defined in the `metrics.py` file and include popular options such as BLEU, ROUGE, METEOR, and more. The framework is extensible, allowing users to add new metrics as needed. To add a custom metric, update the `metrics.py` file with the new metric function and register it using the `@register` decorator. The `@register` decorator takes the metric's name as input, ensuring it is included in the evaluation pipeline. Each metric function must accept two inputs: `predictions` and `references`, and it should return a score representing the evaluation result. This flexible design allows seamless integration of additional metrics tailored to specific use cases.

### List of Current Metrics

| **Name**     | **Official Name**         | **Description**                                                                                                                                       |
|--------------|---------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| `meteor`     | METEOR                    | A metric evaluating translation quality based on precision, recall, and a harmonic mean, accounting for synonyms and stemming.                        |
| `moverS`     | MoverScore                | Measures similarity between texts by capturing semantic alignment using contextual embeddings.                                                         |
| `bleurt`     | BLEURT                    | A learned evaluation metric using pre-trained models to assess fluency, adequacy, and text similarity.                                                |
| `bertS`      | BERTScore                 | Utilizes contextual embeddings from BERT to compare text similarity based on token alignment.                                                         |
| `accuracy`   | Accuracy                  | A basic metric representing the proportion of correctly predicted instances among the total.                                                          |
| `f1`         | F1 Score                  | The harmonic mean of precision and recall, providing a balanced evaluation of both metrics.                                                           |
| `perplexity` | Perplexity                | Measures how well a language model predicts a sample, with lower values indicating better performance.                                                 |
| `rouge`      | ROUGE                     | A set of metrics for evaluating text summarization and translation by comparing overlap in n-grams and sequences.                                      |
| `rougeL`     | ROUGE-L                   | Focuses on the longest common subsequence between reference and candidate texts for evaluation.                                                       |
| `rougeS`     | ROUGE-S                   | Considers skip-bigram co-occurrence, enabling evaluation of distant word relationships.                                                               |
| `rougeM`     | ROUGE-M                   | A variant of ROUGE focusing on specific metrics or configurations within its calculation.                                                             |
| `rouge1`     | ROUGE-1                   | Evaluates the overlap of unigrams (single words) between reference and candidate texts.                                                               |
| `rouge2`     | ROUGE-2                   | Measures the overlap of bigrams (two-word sequences) between reference and candidate texts.                                                           |
| `rouge4`     | ROUGE-4                   | Focuses on the overlap of 4-grams between reference and candidate texts.                                                                              |
| `bleu`       | BLEU                      | Evaluates translation quality by comparing n-gram overlaps between reference and candidate texts.                                                     |
| `bleu1`      | BLEU-1                    | Considers 1-gram overlap (unigrams) in BLEU calculations.                                                                                             |
| `bleu2`      | BLEU-2                    | Evaluates 2-gram overlap (bigrams) in BLEU calculations.                                                                                              |
| `bleu3`      | BLEU-3                    | Measures 3-gram overlap (trigrams) in BLEU calculations.                                                                                              |
| `bleu4`      | BLEU-4                    | Focuses on 4-gram overlap (quadgrams) in BLEU calculations, commonly used in translation tasks.                                                       |
|`sacrebleu`   |SacreBLEU                  | Provides hassle-free computation of shareable, comparable, and reproducible BLEU scores. Expects detokenized outputs, applying its own metric-internal preprocessing, and produces the same values as WMT.|


## Outputs
The results are saved as JSON files in the specified output directory (`--output_path`). If `--log_samples` is enabled, predictions and corresponding scores are logged for each sample.

---

## Contributing
Feel free to contribute by:
1. Adding new tasks with YAML configurations.
2. Adding new metrics in the metrics.py
3. Reporting issues or suggesting improvements.
4. Submitting pull requests.

---
