# llm-rag-metabolic-qa-eval

This repo contains the code for the paper _Does Domain-Specific Retrieval Augmented Generation Help LLMs Answer Consumer Health Questions?_.

# Requirements
```
python 3.9.0
accelerate==1.3.0
beautifulsoup4==4.13.3
bert_score==0.3.13
chromadb==0.6.3
matplotlib==3.4.3
nltk==3.9.1
numpy==2.2.3
pandas==1.5.3
rank_bm25==0.2.2
Requests==2.32.3
rouge_score==0.1.2
sacrebleu==2.5.1
scikit_learn==1.1.2
scipy==1.15.1
seaborn==0.13.2
sentence_transformers==3.2.0
torch==1.11.0
tqdm==4.65.0
transformers==4.40.2
```


# Methods
## Dataset

1. Download MedQuAD question-answer dataset: [MedQuAD](https://github.com/abachaa/MedQuAD) ([NIDDK](https://github.com/abachaa/MedQuAD/tree/master/5_NIDDK_QA) dataset)
- To extract the MedQuAD document corpus from webpages (n=151 for NIDDK):
First, for all invalid URLs from MedQuAD, get most recent valid URLS from WayBack Machine:
```
python medquad_data_collection/get_all_niddk_urls.py
```
Then with all valid urls, download all documents: 
```
python medquad_data_collection/pull_docs_total.py
```
Result of this process is stored in ```/medquad_data_collection/MEDQUAD_DOCS```.

## Experiments

2. To run vanilla LLM question-answering on all MedQuAD NIDDK questions, run the corresponding script for each model (e.g., for Llama-3-70B-Instruct): 
```
python answer_gneration/vanilla_answer_generation/Meta-Llama-3-70B-Instruct.py
```

3. To prepare and perform RAG on all MedQuAD NIDDK questions: 
- First preprocess, chunk, and vectorize the NIDDK documents (use the randomly sampled set of questions for preprocessing tuning: ```sample12_NEWURLS_MEDQUAD_NIDDK_QA_DATASET.csv```):
```
python answer_generation/RAG_answer_generation/preprocessing/main_rag_prepare.py
```
- Second, perform answer generation with the ChromaDB vector database: 
```
python answer_generation/RAG_answer_generation//corpusaware_exp_medical_rag_qa.py
```

4. To evaluate results of question generation for vanilla LLMs and RAG
- Perform quantitative evaluation for RAG and vanilla LLM answers, respectively: 
```
python answer_generation/metric_evaluation.py
```
- Perform qualitative LLM-as-a-judge evaluation: 
```
python LLM_judge/LLM_judge_RAG.py
```
```
python LLM_judge/LLM_judge_vanilla.py
```
5. Given clinician annotations (n=66) in ```clinician_evaluation/A1_H2H_ANNOTATION_REQUEST.xlsx```, compare LLM-judge vs. clinician's annotations:
```
python LLM_judge/h2h_human_LLM_agreement.py
```


