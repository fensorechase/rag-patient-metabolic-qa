import json
import random
from collections import defaultdict
import pandas as pd
import os
from pathlib import Path

def load_rag_results(file_path):
    """Load and process RAG results JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data.get('results', [])

def load_vanilla_results(file_path):
    """Load vanilla results JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def get_model_name_mapping():
    """Define mapping between file names and standardized model names."""
    return {
        'Meta_Llama_3_1_70B_temp_0.1.json': 'llama_70b',
        'Meta_Llama_3_1_8B_temp_0.1.json': 'llama_8b',
        'Mixtral_8x7B_Instruct_v0_1_temp_0.1.json': 'mixtral',
        'medquad_rag_results_quantized_llama_3.1_70B.json': 'llama_70b', # 
        'medquad_rag_results_Llama-3.1-8B-Instruct.json': 'llama_8b',
        'medquad_rag_results_Mixtral-8x7B-Instruct-v0.1.json': 'mixtral'
    } 

def create_comparison_dataset(rag_dir, vanilla_dir, output_file):
    model_mapping = get_model_name_mapping()
    
    # Create dictionaries to store responses by model and qid
    rag_responses = defaultdict(dict)  # {model: {qid: response_data}}
    vanilla_responses = defaultdict(dict)
    
    # Process RAG results first (since we'll select qids from here)
    questions_by_type = defaultdict(dict)  # {qtype: {qid: full_data}}
    
    # Load RAG results
    for file_name in os.listdir(rag_dir):
        if not file_name.endswith('.json'):
            continue
            
        model_key = model_mapping.get(file_name)
        if not model_key:
            continue
            
        file_path = os.path.join(rag_dir, file_name)
        rag_data = load_rag_results(file_path)
        
        for entry in rag_data:
            qid = entry['qid']
            qtype = entry['qtype']
            
            # Store in rag_responses
            rag_responses[model_key][qid] = entry
            
            # Store for qtype-based selection
            if qid not in questions_by_type[qtype]:
                questions_by_type[qtype][qid] = entry
    
    # Load vanilla results
    for file_name in os.listdir(vanilla_dir):
        if not file_name.endswith('.json'):
            continue
            
        model_key = model_mapping.get(file_name)
        if not model_key:
            continue
            
        file_path = os.path.join(vanilla_dir, file_name)
        vanilla_data = load_vanilla_results(file_path)
        
        for entry in vanilla_data:
            qid = entry['qid']
            vanilla_responses[model_key][qid] = entry
    
    # Randomly select 2 questions from each qtype
    selected_entries = []
    for qtype in questions_by_type:
        qtype_entries = list(questions_by_type[qtype].values())
        num_to_select = min(2, len(qtype_entries))
        selected = random.sample(qtype_entries, num_to_select)
        selected_entries.extend(selected)
    
    # Create comparison data for Excel
    comparison_data = []
    for entry in selected_entries:
        qid = entry['qid']
        
        for model_key in ['llama_70b', 'llama_8b', 'mixtral']:
            rag_entry = rag_responses[model_key].get(qid)
            vanilla_entry = vanilla_responses[model_key].get(qid)
            
            if rag_entry and vanilla_entry:
                comparison_data.append({
                    'QID': qid,
                    'Question Type': entry['qtype'],
                    'Question': entry['question_text'],
                    'Original Answer': entry['original_answer'],
                    'Model': model_key,
                    'Vanilla Answer': vanilla_entry.get('llm_vanilla_answer', ''),
                    'RAG Answer': rag_entry.get('llm_rag_answer', ''),
                    'Vanilla Source': f"vanilla_{model_key}_{qid}",
                    'RAG Source': f"rag_{model_key}_{qid}"
                })
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(comparison_data)
    df.to_excel(output_file, index=False)
    
    print(f"Created comparison file at: {output_file}")
    print(f"Total unique questions processed: {len(selected_entries)}")
    print(f"Total comparisons generated: {len(comparison_data)}")

def main():
    rag_dir = "RAG_final_results"
    vanilla_dir = "final_results_vanilla"
    output_file = "H2H_ANNOTATION_REQUEST.xlsx"
    
    try:
        create_comparison_dataset(rag_dir, vanilla_dir, output_file)
    except Exception as e:
        print(f"Error processing files: {str(e)}")

if __name__ == "__main__":
    main()