# Introduction
This repository contains code for the paper *"Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific vs. Non-Scientific Data"*.

We investigate the effectiveness of both *text-based* and *multimodal* LLMs on table understanding tasks through a cross-domain and cross-modality evaluation. Specifically, we compare their performance on tables from *scientific* vs. *non-scientific* contexts and examine their robustness on tables represented as *images* vs. *text*. Additionally, we conduct an interpretability analysis to measure context usage and input relevance. We also introduce the **TableEval** benchmark, comprising **3017** tables from scholarly publications, Wikipedia, and financial reports, where each table is provided in five different formats: **Image**, **Dictionary**, **HTML**, **XML**, and **LaTeX**. For more details, please, refer to the paper.

# TableEval dataset

TableEval corpus is developed for benchmarking (M)LLMs performance across different table modalities. It contains six data subsets, comprising 3017 tables and 11312 instances in total. Tables are available as PNG images and in four textual formats including HTML, XML, LaTeX, and Dictionary (Dict). All task annotations are taken from the source datasets. 

**The dataset can be dowloaded from Hugging Face 🤗:** https://huggingface.co/datasets/katebor/TableEval

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

# Interpretability 

The code, instructions, and examples of silency maps are avaialble [here](https://github.com/esborisova/Table-Understanding-Evaluation-Study/tree/main/explanations).

# Evaluation pipeline

All instructions on how to run the evaluation are provided in this [README.md](https://github.com/esborisova/Table-Understanding-Evaluation-Study/tree/main/src/evaluation) file.

# Repository structure
```
    ├── src               
    │   ├── application    # data preparation scripts       
    │   ├── evaluation     # evaluation pipeline and code for running intepretability tools
    │   ├── utils          # functions used for data preparation      
    └──  explanations      # intepretability analysis results                    
```
# Citation
```bibtex
@inproceedings{borisova-etal-2025-table,
    title = "Table Understanding and (Multimodal) {LLM}s: A Cross-Domain Case Study on Scientific vs. Non-Scientific Data",
    author = {Borisova, Ekaterina  and
      Barth, Fabio  and
      Feldhus, Nils  and
      Abu Ahmad, Raia  and
      Ostendorff, Malte  and
      Ortiz Suarez, Pedro  and
      Rehm, Georg  and
      M{\"o}ller, Sebastian},
    editor = "Chang, Shuaichen  and
      Hulsebos, Madelon  and
      Liu, Qian  and
      Chen, Wenhu  and
      Sun, Huan",
    booktitle = "Proceedings of the 4th Table Representation Learning Workshop",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.trl-1.10/",
    pages = "109--142",
    ISBN = "979-8-89176-268-8",
    abstract = "Tables are among the most widely used tools for representing structured data in research, business, medicine, and education. Although LLMs demonstrate strong performance in downstream tasks, their efficiency in processing tabular data remains underexplored. In this paper, we investigate the effectiveness of both text-based and multimodal LLMs on table understanding tasks through a cross-domain and cross-modality evaluation. Specifically, we compare their performance on tables from scientific vs. non-scientific contexts and examine their robustness on tables represented as images vs. text. Additionally, we conduct an interpretability analysis to measure context usage and input relevance. We also introduce the TableEval benchmark, comprising 3017 tables from scholarly publications, Wikipedia, and financial reports, where each table is provided in five different formats: Image, Dictionary, HTML, XML, and LaTeX. Our findings indicate that while LLMs maintain robustness across table modalities, they face significant challenges when processing scientific tables."
}
```
