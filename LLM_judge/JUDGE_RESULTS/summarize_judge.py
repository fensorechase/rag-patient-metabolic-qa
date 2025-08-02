import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import argparse
from logging import getLogger

logger = getLogger(__name__)

"""
How to run:
1. (strict within file) To use only questions with all 8 criteria non-NaN:
python summarize_judge.py --complete-only

2. (most strict) To limit to INTERSECTION of non-NaN judged questions across vanilla, RAG, & all models:
python summarize_judge.py --complete-only --intersection-only

2. (least strict) To use all available data:
python summarize_judge.py
"""

criteria = [
    "1_scientific_consensus", "2_inappropriate_content", 
    "3_missing_content", "4_extent_harm", "5_likelihood_harm",
    "6_bias", "7_empathy", "8_grammaticality"
]

def load_judge_results(base_dir: str) -> dict:
    """Load and process judge results from multiple files with exact filename matching"""
    results = {}
    
    # Define directory paths and exact filenames
    rag_files = {
        "Llama-8B": "medquad_rag_results_Llama-3.1-8B-Instruct_all_evaluators_combined_20250210_045125.json",
        "Llama-70B": "medquad_rag_results_quantized_llama_3.1_70B_all_evaluators_combined_20250210_064236.json",
        "Mixtral-8x7B": "medquad_rag_results_Mixtral-8x7B-Instruct-v0.1_all_evaluators_combined_20250210_083003.json",
        "Meditron-70B": "medquad_rag_results_meditron-70b_all_evaluators_combined_20250316_214006.json"

    }
    
    vanilla_files = {
        "Llama-8B": "dedup_Meta_Llama_3_1_8B_temp_0.1_all_evaluators_combined_20250210_042328.json",
        "Llama-70B": "dedup_Meta_Llama_3_1_70B_temp_0.1_all_evaluators_combined_20250210_065822.json",
        "Mixtral-8x7B": "dedup_Mixtral_8x7B_Instruct_v0_1_temp_0.1_all_evaluators_combined_20250210_093334.json",
        "Meditron-70B": "vanilla_results_meditron_evaluated_by_llama-70b_detailed_20250316_020427.json"
    }
    
    # Load RAG files
    for model, filename in rag_files.items():
        filepath = Path(base_dir) / 'RAG_judged' / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                evaluations = data.get('evaluations', [])
                
                # Keep track of seen question IDs
                seen_question_ids = set()
                unique_evaluations = []
                
                for eval in evaluations:
                    question_id = eval.get('question_text', '')  # or another unique identifier
                    if question_id not in seen_question_ids:
                        seen_question_ids.add(question_id)
                        eval['approach'] = 'RAG'
                        unique_evaluations.append(eval)
                
                results[f"{model}_RAG"] = unique_evaluations
                logger.info(f"Loaded {len(evaluations)} RAG evaluations for {model}")
    
    # Load vanilla files
    for model, filename in vanilla_files.items():
        filepath = Path(base_dir) / 'vanilla_judged' / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)

                evaluations = data.get('evaluations', [])
  
                # Keep track of seen question IDs
                seen_question_ids = set()
                unique_evaluations = []
                
                for eval in evaluations:
                    question_id = eval.get('question_text', '')  # or another unique identifier
                    if question_id not in seen_question_ids:
                        seen_question_ids.add(question_id)
                        eval['approach'] = 'vanilla'
                        unique_evaluations.append(eval)
                
                results[f"{model}_vanilla"] = unique_evaluations
                logger.info(f"Loaded {len(evaluations)} vanilla evaluations for {model}")
    
    return results

def normalize_score(raw_score):
    """Normalize score from [-10, 4] to [0, 10] scale"""
    return ((raw_score + 10) / 14) * 10

def analyze_scores(results: dict, complete_only: bool = False) -> tuple:
    """Analyze scores and count NaNs
    
    Args:
        results: Dictionary of evaluation results
        complete_only: If True, only use questions with all 8 criteria non-NaN
    """
    score_data = []
    nan_counts = {
        'model': [],
        'approach': [],
        'criterion': [],
        'nan_count': []
    }
    
    criteria = [
        "1_scientific_consensus", "2_inappropriate_content", 
        "3_missing_content", "4_extent_harm", "5_likelihood_harm",
        "6_bias", "7_empathy", "8_grammaticality"
    ]
    
    for key, evals in results.items():
        model = key.rsplit('_', 1)[0]  # Split off approach
        
        # Process each evaluation
        for eval in evals:
            scores = eval['criteria_scores']
            
            # Check for NaNs in any criteria
            has_nan = False
            for criterion in criteria:
                if np.isnan(scores[criterion]['score']):
                    has_nan = True
                    nan_counts['model'].append(model)
                    nan_counts['approach'].append(eval['approach'])
                    nan_counts['criterion'].append(criterion)
                    nan_counts['nan_count'].append(1)
            
            # Skip this evaluation if complete_only is True and it has any NaN
            if complete_only and has_nan:
                continue
                
            # Calculate composite score
            raw_total = sum(scores[c]['score'] for c in criteria if not np.isnan(scores[c]['score']))
            norm_score = normalize_score(raw_total)
            
            score_data.append({
                'model': model,
                'approach': eval['approach'],
                'total_score': norm_score
            })
    
    return pd.DataFrame(score_data), pd.DataFrame(nan_counts)


def plot_criteria_scores(results: dict, output_dir: str, complete_only: bool = False):
    """Generate horizontal bar plots with scaled color darkness for ordinal values
    
    Args:
        results: Dictionary of evaluation results
        output_dir: Directory to save output files
        complete_only: If True, only use questions with all 8 criteria non-NaN
    """
    criteria_names = {
        "1_scientific_consensus": "Scientific Consensus",
        "2_inappropriate_content": "Inappropriate Content",
        "3_missing_content": "Missing Content",
        "4_extent_harm": "Extent of Harm",
        "5_likelihood_harm": "Likelihood of Harm",
        "6_bias": "Bias",
        "7_empathy": "Empathy",
        "8_grammaticality": "Grammaticality"
    }
    
    # Define categorical mappings AND their severity order (higher index = lighter shade)
    category_severity = {
        "1_scientific_consensus": {-1: 0, 0: 1, 1: 2},  # Opposed = darkest, Aligned = lightest
        "2_inappropriate_content": {-2: 0, -1: 1, 0: 2},  # Major issues = darkest
        "3_missing_content": {-2: 0, -1: 1, 0: 2},  # Major omissions = darkest
        "4_extent_harm": {-2: 0, -1: 1, 0: 2},  # Severe = darkest
        "5_likelihood_harm": {-2: 0, -1: 1, 0: 2},  # High = darkest
        "6_bias": {-1: 0, 0: 1},  # Yes = darkest
        "7_empathy": {0: 0, 1: 1, 2: 2},  # Low = darkest
        "8_grammaticality": {0: 0, 1: 1}  # Has errors = darkest
    }
    
    category_mappings = {
        "1_scientific_consensus": {-1: "Opposed", 0: "No consensus", 1: "Aligned"},
        "2_inappropriate_content": {-2: "Major issues", -1: "Minor issues", 0: "None"},
        "3_missing_content": {-2: "Major omissions", -1: "Minor omissions", 0: "Complete"},
        "4_extent_harm": {-2: "Severe", -1: "Moderate", 0: "None"},
        "5_likelihood_harm": {-2: "High", -1: "Medium", 0: "Low"},
        "6_bias": {-1: "Yes", 0: "No"},
        "7_empathy": {0: "Low", 1: "Moderate", 2: "High"},
        "8_grammaticality": {0: "Has errors", 1: "No errors"}
    }
    
    # Color scales for each approach (darkest to lightest)
    color_scales = {
        'vanilla': {  # Blue scale
            0: '#08519c',  # Darkest
            1: '#3182bd',
            2: '#6baed6',
            3: '#bdd7e7',  # Lightest for 4-category
            'binary_dark': '#08519c',  # For binary categories - darkest
            'binary_light': '#eef3f8'  # For binary categories - very light
        },
        'RAG': {  # Green scale
            0: '#006d2c',  # Darkest
            1: '#31a354',
            2: '#74c476',
            3: '#bae4b3',  # Lightest for 4-category
            'binary_dark': '#006d2c',  # For binary categories - darkest
            'binary_light': '#eef6ed'  # For binary categories - very light
        }
    }
    
    # Grey scale for legend (darkest to lightest)
    grey_scale = ['#252525', '#636363', '#969696', '#cccccc']
    
    # plt.style.use('seaborn')
    plt.style.use('default')  # Reset to default style
    fig, axes = plt.subplots(4, 2, figsize=(15, 20), facecolor='white')
    axes = axes.flatten()

    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    axes = axes.flatten()
    
    models = ['Meditron-70B', 'Mixtral-8x7B', 'Llama-70B', 'Llama-8B']
    
    for idx, (criterion, name) in enumerate(criteria_names.items()):
        ax = axes[idx]
        categories = category_mappings[criterion]
        severity_map = category_severity[criterion]
        is_binary = len(categories) == 2
        
        # Create category legend with grey scales
        # Sort categories by severity for legend
        sorted_categories = sorted(categories.items(), key=lambda x: severity_map[x[0]])
        category_handles = [plt.Rectangle((0,0), 1, 1, fc=grey_scale[severity_map[score]]) 
                          for score, _ in sorted_categories]
        category_labels = [label for _, label in sorted_categories]
        
        ax.legend(category_handles, category_labels, 
                 loc='upper center', bbox_to_anchor=(0.55, 1.08),
                 ncol=len(categories)) # 0.5, 1.15
        
        y_positions = []
        for i, model in enumerate(models):
            base_pos = i * 3  # Space for two bars plus gap
            y_positions.extend([base_pos, base_pos + 0.8])
            
            for approach in ['vanilla', 'RAG']:
                key = f"{model}_{approach}"
                if key in results:
                    evals = results[key]
                    color_scale = color_scales[approach]
                    
                    # Filter evaluations if complete_only is True
                    if complete_only:
                        criteria_list = list(criteria_names.keys())
                        filtered_evals = []
                        for e in evals:
                            scores_dict = e['criteria_scores']
                            if not any(np.isnan(scores_dict[c]['score']) for c in criteria_list):
                                filtered_evals.append(e)
                        evals = filtered_evals
                    
                    # Get non-NaN scores
                    scores = [e['criteria_scores'][criterion]['score'] for e in evals]
                    valid_scores = [s for s in scores if not np.isnan(s)]
                    
                    if valid_scores:  # Only process if we have valid scores
                        rounded_scores = np.floor(np.array(valid_scores))
                        unique, counts = np.unique(rounded_scores, return_counts=True)
                        
                        # Calculate percentages based on valid scores only
                        percentages = (counts / len(valid_scores)) * 100
                        
                        # Sort by severity
                        severity_order = [severity_map[score] for score in unique]
                        sort_idx = np.argsort(severity_order)
                        unique = unique[sort_idx]
                        percentages = percentages[sort_idx]
                        
                        y_pos = base_pos + (0.8 if approach == 'vanilla' else 0)
                        x_start = 0
                        
                        # Plot each category segment
                        for score, pct in zip(unique, percentages):
                            severity_idx = severity_map[score]
                            if is_binary:
                                color = color_scale['binary_dark'] if severity_idx == 0 else color_scale['binary_light']
                            else:
                                color = color_scale[severity_idx]
                            
                            ax.barh(y_pos, pct, left=x_start, height=0.6,  # Reduced height could be 0.35
                                   color=color)
                            
                            # Add percentage label if significant (> 5%)
                            #if pct > 5:
                            #    ax.text(x_start + pct/2, y_pos, 
                            #           f"{pct:.1f}%", 
                            #           ha='center', va='center',
                            #           color='white' if severity_idx == 0 else 'black')
                            
                            x_start += pct
                        
                        # Fill remainder with lightest shade if not 100%
                        if x_start < 100:
                            remainder_color = color_scale['binary_light'] if is_binary else color_scale[max(severity_map.values())]
                            ax.barh(y_pos, 100-x_start, left=x_start, height=0.35,  # Reduced height
                                   color=remainder_color)
        
        ax.set_title(name, pad=22)
        ax.set_xlabel('Percentage')
        ax.set_yticks([i * 3 + 0.4 for i in range(len(models))])
        ax.set_yticklabels(models)
        ax.set_xlim(0, 100)  # Force x-axis to exactly 100%
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_axisbelow(True)
    
    # Add main RAG vs Vanilla legend to the figure
    vanilla_patch = plt.Rectangle((0, 0), 1, 1, fc='#3182bd', label='Vanilla')
    rag_patch = plt.Rectangle((0, 0), 1, 1, fc='#31a354', label='RAG')
    fig.legend(handles=[vanilla_patch, rag_patch],
              loc='upper right',
              bbox_to_anchor=(1.0, 1.0)) # (0.98, 0.98))
    
    plt.tight_layout()
    
    # Save with appropriate filename
    if complete_only:
        plt.savefig(Path(output_dir) / 'criteria_scores_complete_only.pdf', bbox_inches='tight', dpi=300)
    else:
        plt.savefig(Path(output_dir) / 'criteria_scores.pdf', bbox_inches='tight', dpi=300)


def generate_latex_table(score_df: pd.DataFrame, output_dir: str, complete_only: bool = False) -> None:
    """Generate LaTeX table of composite scores"""
    summary = score_df.groupby(['model', 'approach'])['total_score'].agg(['mean', 'std', 'count']).round(2)
    
    latex = """\\begin{table}[h]
\\centering
\\begin{tabular}{llccc}
\\toprule
Model & Approach & Mean Score & Std Dev & Count \\\\
\\midrule
"""
    
    for idx in summary.index:
        model, approach = idx
        mean, std, count = summary.loc[idx]
        latex += f"{model} & {approach} & {mean:.2f} & {std:.2f} & {count:.0f} \\\\\n"
    
    # Add caption based on mode
    if complete_only:
        latex += """\\bottomrule
\\end{tabular}
\\caption{Composite Score Summary (0-10 scale) - Only complete evaluations}
\\label{tab:scores_complete}
\\end{table}"""
    else:
        latex += """\\bottomrule
\\end{tabular}
\\caption{Composite Score Summary (0-10 scale)}
\\label{tab:scores}
\\end{table}"""
    
    # Save with appropriate filename
    if complete_only:
        with open(Path(output_dir) / 'score_summary_complete_only.tex', 'w') as f:
            f.write(latex)
    else:
        with open(Path(output_dir) / 'score_summary.tex', 'w') as f:
            f.write(latex)


def generate_criteria_statistics(results: dict, output_dir: str, complete_only: bool = False):
    """Generate statistics for each criterion by model and approach
    
    Args:
        results: Dictionary of evaluation results
        output_dir: Directory to save output files
        complete_only: If True, only use questions with all 8 criteria non-NaN
    """
    criteria = [
        "1_scientific_consensus", "2_inappropriate_content", 
        "3_missing_content", "4_extent_harm", "5_likelihood_harm",
        "6_bias", "7_empathy", "8_grammaticality"
    ]
    
    criteria_names = {
        "1_scientific_consensus": "Scientific Consensus",
        "2_inappropriate_content": "Inappropriate Content",
        "3_missing_content": "Missing Content",
        "4_extent_harm": "Extent of Harm",
        "5_likelihood_harm": "Likelihood of Harm",
        "6_bias": "Bias",
        "7_empathy": "Empathy",
        "8_grammaticality": "Grammaticality"
    }
    
    # Initialize data collection
    stats_data = []
    
    for key, evals in results.items():
        model = key.rsplit('_', 1)[0]
        approach = key.rsplit('_', 1)[1]
        
        # Filter evaluations if complete_only is True
        if complete_only:
            filtered_evals = []
            for e in evals:
                scores_dict = e['criteria_scores']
                if not any(np.isnan(scores_dict[c]['score']) for c in criteria):
                    filtered_evals.append(e)
            evals = filtered_evals
        
        for criterion in criteria:
            # Get non-NaN scores for this criterion
            scores = [e['criteria_scores'][criterion]['score'] for e in evals 
                     if not np.isnan(e['criteria_scores'][criterion]['score'])]
            
            if scores:
                stats_data.append({
                    'Model': model,
                    'Approach': approach,
                    'Criterion': criteria_names[criterion],
                    'Count': len(scores),
                    'Mean': np.mean(scores),
                    'Std': np.std(scores),
                    'Min': np.min(scores),
                    'Max': np.max(scores)
                })
    
    # Create dataframe and save as CSV
    stats_df = pd.DataFrame(stats_data)
    stats_df = stats_df.round(2)
    
    # Save with appropriate filename
    if complete_only:
        stats_df.to_csv(Path(output_dir) / 'criteria_statistics_complete_only.csv', index=False)
    else:
        stats_df.to_csv(Path(output_dir) / 'criteria_statistics.csv', index=False)
    
    # Generate LaTeX table for each criterion
    for criterion in criteria_names.values():
        criterion_df = stats_df[stats_df['Criterion'] == criterion]
        
        latex = f"""\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{llccc}}
\\toprule
Model & Approach & Mean & Std Dev & Count \\\\
\\midrule
"""
        
        for _, row in criterion_df.iterrows():
            latex += f"{row['Model']} & {row['Approach']} & {row['Mean']:.2f} & {row['Std']:.2f} & {row['Count']:.0f} \\\\\n"
        
        # Add caption based on mode
        if complete_only:
            latex += f"""\\bottomrule
\\end{{tabular}}
\\caption{{{criterion} Scores - Only complete evaluations}}
\\label{{tab:{criterion.lower().replace(' ', '_')}_complete}}
\\end{{table}}"""
        else:
            latex += f"""\\bottomrule
\\end{{tabular}}
\\caption{{{criterion} Scores}}
\\label{{tab:{criterion.lower().replace(' ', '_')}}}
\\end{{table}}"""
        
        # Save with appropriate filename
        file_name = criterion.lower().replace(' ', '_')
        if complete_only:
            with open(Path(output_dir) / f'{file_name}_complete_only.tex', 'w') as f:
                f.write(latex)
        else:
            with open(Path(output_dir) / f'{file_name}.tex', 'w') as f:
                f.write(latex)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze judge results")
    parser.add_argument("--complete-only", action="store_true", 
                        help="Only use questions where all 8 criteria are non-NaN")
    parser.add_argument("--intersection-only", action="store_true", 
                    help="Only use questions with complete data across all models and approaches")
    
    args = parser.parse_args()

    # For tracking question NaNs:
    all_question_texts = set()
    question_text_data = {}  # Map question_text -> {model_approach -> has_complete_data}
    
    base_dir = "."  # Current directory
    output_dir = "./criteria_summary"  # Output directory for results
    
    # Adjust output directory if using complete-only mode
    if args.complete_only:
        output_dir += "_complete_only"

    # Adjust output directory if using --intersection-only mode.
    if args.complete_only:
        output_dir += "_intersection_only"
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)


    # Load results using exact filenames
    results = load_judge_results(base_dir)


    # After loading all files
    for key, evals in results.items():
        for eval_item in evals:
            text = eval_item.get('question_text', '')
            if text:
                all_question_texts.add(text)
                
                # Check if this item has complete data
                scores = eval_item['criteria_scores']
                has_complete = not any(np.isnan(scores[c]['score']) for c in criteria)
                
                if text not in question_text_data:
                    question_text_data[text] = {}
                question_text_data[text][key] = has_complete
    
    
    if args.intersection_only:
        # Find question_texts present in all models with complete data
        complete_questions = set()
        all_keys = results.keys()
        
        for text, data in question_text_data.items():
            if len(data) == len(all_keys) and all(data.values()):
                complete_questions.add(text)
        
        # Filter results
        for key in results:
            results[key] = [e for e in results[key] if e.get('question_text', '') in complete_questions]

    ######################################
    # Perform analysis below:
    ######################################
    # Analyze scores
    score_df, nan_df = analyze_scores(results, args.complete_only)
    
    # Generate plots
    plot_criteria_scores(results, output_dir, args.complete_only)
    
    # Generate LaTeX table
    generate_latex_table(score_df, output_dir, args.complete_only)
    
    # Generate criteria-specific statistics
    generate_criteria_statistics(results, output_dir, args.complete_only)
    
    # Print NaN summary
    print("\nNaN Counts by Model, Approach, and Criterion:")
    nan_summary = nan_df.groupby(['model', 'approach', 'criterion'])['nan_count'].sum()
    print(nan_summary)
    
    # Print mode information
    if args.complete_only:
        print(f"\nRunning in complete-only mode: only using questions with all 8 criteria non-NaN")
    else:
        print(f"\nRunning in standard mode: using all available data")

if __name__ == "__main__":
    main()