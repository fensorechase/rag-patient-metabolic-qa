import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import nltk
nltk.download('punkt_tab')
from typing import List, Dict
import os
import glob

def tokenize_text(text: str) -> List[str]:
    """Tokenize text into words."""
    return nltk.word_tokenize(text.lower())

def calculate_bleu(reference: str, candidate: str, n: int = 1) -> float:
    """Calculate BLEU-n score between reference and candidate texts."""
    if not isinstance(reference, str) or not isinstance(candidate, str):
        return 0.0
    
    reference_tokens = tokenize_text(reference)
    candidate_tokens = tokenize_text(candidate)
    
    if len(candidate_tokens) == 0:
        return 0.0
    
    weights = tuple([1.0/n if i < n else 0 for i in range(4)])
    smoothing = SmoothingFunction().method1
    
    try:
        return sentence_bleu([reference_tokens], candidate_tokens, 
                           weights=weights, smoothing_function=smoothing)
    except:
        return 0.0

def calculate_rouge(reference: str, candidate: str, rouge_type: str = 'rouge1') -> float:
    """Calculate ROUGE score between reference and candidate texts."""
    if not isinstance(reference, str) or not isinstance(candidate, str):
        return 0.0
    
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    
    try:
        scores = scorer.score(reference, candidate)
        return scores[rouge_type].fmeasure
    except:
        return 0.0

def calculate_bert_score(references: List[str], candidates: List[str]) -> Dict[str, List[float]]:
    """Calculate BERTScore between reference and candidate texts."""
    scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    try:
        P, R, F1 = scorer.score(candidates, references)
        return F1.tolist()
    except:
        return [0.0] * len(references)

def calculate_metrics_for_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all metrics for each row in the dataframe."""
    # Find the column containing 'answer_vanilla'
    vanilla_col = [col for col in df.columns if 'answer_vanilla' in col][0]
    
    # Initialize lists to store metrics
    bleu_1_scores = []
    bleu_4_scores = []
    rouge_1_scores = []
    rouge_l_scores = []
    
    # Calculate BLEU and ROUGE scores row by row
    for _, row in df.iterrows():
        reference = str(row['answer_text'])
        candidate = str(row[vanilla_col])
        
        bleu_1_scores.append(calculate_bleu(reference, candidate, n=1))
        bleu_4_scores.append(calculate_bleu(reference, candidate, n=4))
        rouge_1_scores.append(calculate_rouge(reference, candidate, 'rouge1'))
        rouge_l_scores.append(calculate_rouge(reference, candidate, 'rougeL'))
    
    # Calculate BERTScore for all rows at once
    bert_scores = calculate_bert_score(
        df['answer_text'].astype(str).tolist(),
        df[vanilla_col].astype(str).tolist()
    )
    
    # Add metrics to dataframe
    metrics_df = df.copy()
    metrics_df['bleu_1'] = bleu_1_scores
    metrics_df['bleu_4'] = bleu_4_scores
    metrics_df['rouge_1'] = rouge_1_scores
    metrics_df['rouge_l'] = rouge_l_scores
    metrics_df['bertscore'] = bert_scores
    
    return metrics_df

def process_file(filepath: str) -> Dict[str, float]:
    """Process a single file and return its metrics."""
    print(f"\nProcessing {filepath}...")
    
    try:
        # Read dataframe
        df = pd.read_csv(filepath)
        
        # Calculate metrics
        results_df = calculate_metrics_for_dataframe(df)
        
        # Get model name from filename
        model_name = os.path.basename(filepath).split('_temp')[0]
        
        # Save results
        output_filename = f'{model_name}_results_with_metrics.csv'
        results_df.to_csv(output_filename, index=False)
        print(f"Saved results to {output_filename}")
        
        # Calculate average scores
        metrics = {
            'bleu_1': results_df['bleu_1'].mean(),
            'bleu_4': results_df['bleu_4'].mean(),
            'rouge_1': results_df['rouge_1'].mean(),
            'rouge_l': results_df['rouge_l'].mean(),
            'bertscore': results_df['bertscore'].mean()
        }
        
        return metrics
        
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return None

def main():
    try:
        # Get all CSV files in the medquad_outputs directory
        input_files = glob.glob('medquad_outputs/*.csv')
        
        if not input_files:
            print("No CSV files found in medquad_outputs directory!")
            return
        
        # Store results for all models
        all_results = {}
        
        # Process each file
        for filepath in input_files:
            metrics = process_file(filepath)
            if metrics:
                model_name = os.path.basename(filepath).split('_temp')[0]
                all_results[model_name] = metrics
        
        # Print comparative results
        print("\n=== Comparative Results ===")
        print("\nModel Scores:")
        
        # Get all metric names
        metric_names = ['bleu_1', 'bleu_4', 'rouge_1', 'rouge_l', 'bertscore']
        
        # Print header
        print(f"{'Model':<30} | " + " | ".join(f"{metric:^10}" for metric in metric_names))
        print("-" * (30 + len(metric_names) * 13))
        
        # Print each model's results
        for model, metrics in all_results.items():
            scores = [f"{metrics[metric]:.4f}" for metric in metric_names]
            print(f"{model:<30} | " + " | ".join(f"{score:^10}" for score in scores))
        
        # Save all results to a summary CSV
        summary_df = pd.DataFrame.from_dict(all_results, orient='index')
        summary_df.to_csv('all_models_metrics_summary.csv')
        print("\nSaved summary results to all_models_metrics_summary.csv")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()