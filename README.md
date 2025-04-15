# 

# TableEval dataset

TableEval corpus is developed for benchmarking (M)LLMs performance across different table modalities. It contains six data subsets, comprising 3017 tables and 11312 instances in total. Tables are available as PNG images and in four textual formats including HTML, XML, LaTeX, and Dictionary (Dict). All task annotations are taken from the source datasets. 


| Dataset               | Task               | Source             | Image         | Dict          |  LaTeX        | HTML          | XML           |
|-----------------------|--------------------|-------------------|---------------|---------------|---------------|---------------|---------------|
| ComTQA (PubTables-1M) <img src='https://img.shields.io/badge/arXiv-2024-red'> <a href='https://arxiv.org/abs/2406.01326'><img src='https://img.shields.io/badge/PDF-blue'></a> <a href='https://huggingface.co/datasets/ByteDance/ComTQA'><img src='https://img.shields.io/badge/Dataset-gold'> |   VQA              |   PubMed Central                     |       ⬇️        |          ⚙️     |    ⚙️           |    ⚙️           |     📄           |
| numericNLG <img src='https://img.shields.io/badge/ACL-2021-red'> <a href='https://aclanthology.org/2021.acl-long.115.pdf'><img src='https://img.shields.io/badge/PDF-blue'></a> <a href='https://huggingface.co/datasets/kasnerz/numericnlg?row=0'><img src='https://img.shields.io/badge/Dataset-gold'></a>          |   T2T              |   ACL Anthology     |    📄            |       ⬇️         |        ⚙️        |     ⬇️          |       ⚙️         |
| SciGen <img src='https://img.shields.io/badge/arXiv-2021-red'> <a href='https://arxiv.org/abs/2104.08296'><img src='https://img.shields.io/badge/PDF-blue'></a> <a href='https://github.com/UKPLab/SciGen/tree/main'><img src='https://img.shields.io/badge/Dataset-gold'></a>               |   T2T              |   arXiv and ACL Anthology|    📄          |    ⬇️            |       📄        |      ⚙️         |        ⚙️       |
| ComTQA (FinTabNet)  <img src='https://img.shields.io/badge/arXiv-2024-red'> <a href='https://arxiv.org/abs/2406.01326'><img src='https://img.shields.io/badge/PDF-blue'></a> <a href='https://huggingface.co/datasets/ByteDance/ComTQA'><img src='https://img.shields.io/badge/Dataset-gold'>  |   VQA              |   Earnings reports of S&P 500 companies         | 📄              |      ⚙️          |     ⚙️          |       ⚙️        |      ⚙️         |
| LogicNLG <img src='https://img.shields.io/badge/ACL-2020-red'> <a href='https://aclanthology.org/2020.acl-main.708/'><img src='https://img.shields.io/badge/PDF-blue'></a> <a href='https://huggingface.co/datasets/kasnerz/logicnlg'><img src='https://img.shields.io/badge/Dataset-gold'></a>             |   T2T              |   Wikipedia            |  ⚙️              |         ⬇️      |       ⚙️       |      📄           |     ⚙️          |
| Logic2Text  <img src='https://img.shields.io/badge/ACL-2020-red'> <a href='https://aclanthology.org/2020.findings-emnlp.190/'><img src='https://img.shields.io/badge/PDF-blue'></a> <a href='https://huggingface.co/datasets/kasnerz/logic2text'><img src='https://img.shields.io/badge/Dataset-gold'></a>          |   T2T              |   Wikipedia              |       ⚙️         |      ⬇️         |      ⚙️        |       📄          |      ⚙️         |

**Symbol ⬇️ indicates formats already available in the given corpus, while  📄  and ⚙️  denote formats extracted from the table source files (e. g., article PDF, Wikipedia page) and generated from other formats in this study, respectively.


The dataset can be dowloaded from Zenodo: 

# Statistics 
#### Number of tables per format and dataset

| Dataset                  |  Image             | Dict              |  LaTeX        | HTML          | XML           |
|------------------------- |--------------------|-------------------|---------------|---------------|---------------|
|  ComTQA (PubTables-1M)   |   932              |   932             |    932        |    932        |       932     |   
|  numericNLG              |   135              |   135             |    135        |    135        |       135     |             
|  SciGen                  |   1035             |   1035            |    928        |    985        |       961     |
|  ComTQA (FinTabNet)      |   659              |   659             |  659          |    659        |     659       |
|  LogicNLG                |   184              |  184              |     184       |    184        |       184     |
|  Logic2Text              |   72               |    72             |     72        |    72         |       72      |
|  **Total**               |   **3017**         |   **3017**        |   **2910**    |   **2967**    |  **2943**     |


#### Total number of instances per format and dataset

| Dataset                  |  Image             | Dict              |  LaTeX        | HTML          | XML           |
|------------------------- |--------------------|-------------------|---------------|---------------|---------------|
|  ComTQA (PubTables-1M)   |   6232             |    6232           |    6232       |    6232       |      6232     |   
|  numericNLG              |   135              |   135             |    135        |    135        |       135     |             
|  SciGen                  |   1035             |   1035            |    928        |    985        |       961     |
|  ComTQA (FinTabNet)      |   2838             | 2838              |  2838         |   2838        |      2838     |
|  LogicNLG                |   917              |  917              |  917          |  917          |       917     |
|  Logic2Text              |   155              |    155            |     155       |     155       |        155    |
|  **Total**               |   **11312**        |   **11312**        |   **11205**  |   **11262**   |  **11238**    |


# Models

| Model                    |   🤗 HF checkpoint        | Size (B)          | Vision        | 
|------------------------- |---------------------------|-------------------|---------------|
|  Gemini-2.0-Flash        |   --                      |   --              |    ✅         |   
|  LLaVa-NeXT              | llama3-llava-next-8b-hf   |   8               |    ✅         |         
|  Qwen2.5-VL              |  Qwen2.5-VL-3B-Instruct   |  3                |    ✅         |   
|                          |Qwen2.5-VL-7B-Instruct     |  7                |    ✅         |   
|  Idefics3                |   Idefics3-8B-Llama3      |  8                |    ✅         |   
| Llama-3                  |    Llama-3.2-3B-Instruct  |  3                |    ❌         |    
| Qwen2.5                  |  Qwen2.5-3B-Instruct      |    3              |    ❌         |    
|                          |   Qwen2.5-14B-Instruct    |    14             |    ❌         |    
| Mistral-Nemo             |Mistral-Nemo-Instruct-2407 |  12               |    ❌         | 

# Interpretability tools

# Evaluation pipeline

All the instructions on how to run the evaluation are provided in this [README.md](https://github.com/esborisova/Table-Understanding-Evaluation-Study/tree/main/src/evaluation) file.

# Repository structure
```
    ├── src               
    │   ├── application    # data preparation scripts       
    │   ├── evaluation     # evaluation pipeline and code for running intepretability tools
    │   ├── utils          # functions used for data preparation      
    └──  explanations      # intepretability analysis results                         
```
# Citation
TBA
