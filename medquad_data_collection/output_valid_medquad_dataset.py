import pandas as pd
import numpy as np

def merge_medical_qa_dataset(original_dataset_path, wayback_urls_path, output_path):
    """
    Merge the original medical QA dataset with WayBack Machine URLs
    
    Args:
        original_dataset_path (str): Path to the original medquad_NIDDK_qa_dataset.csv
        wayback_urls_path (str): Path to NIDDK_doc_list_with_wayback_urls.csv
        output_path (str): Path to save the final merged dataset
    """
    # Read the original dataset and the WayBack URLs dataset
    original_df = pd.read_csv(original_dataset_path)
    wayback_df = pd.read_csv(wayback_urls_path)
    
    # Print initial dataset information
    print(f"Original dataset shape: {original_df.shape}")
    print(f"WayBack URLs dataset shape: {wayback_df.shape}")
    
    # Create a mapping dictionary from document_url to new_doc_url
    # Prioritize non-empty new_doc_url entries
    url_mapping = wayback_df[wayback_df['new_doc_url'].notna()].set_index('document_url')['new_doc_url'].to_dict()
    
    # Create a new column 'valid_doc_url' in the original dataset
    original_df['valid_doc_url'] = original_df['document_url'].map(url_mapping).fillna(original_df['document_url'])
    
    # Print summary statistics
    total_urls = len(original_df)
    updated_urls = (original_df['valid_doc_url'] != original_df['document_url']).sum()
    
    print("\nURL Mapping Summary:")
    print(f"Total URLs: {total_urls}")
    print(f"Updated URLs (using WayBack Machine): {updated_urls}")
    print(f"Replacement percentage: {(updated_urls/total_urls)*100:.2f}%")
    
    # Save the updated dataset
    original_df.to_csv(output_path, index=False)
    print(f"\nUpdated dataset saved to {output_path}")

def main():
    original_dataset_path = "medquad_NIDDK_qa_dataset.csv"
    wayback_urls_path = "NIDDK_doc_list_with_wayback_urls.csv"
    output_path = "NEWURLS_MEDQUAD_NIDDK_QA_DATASET.csv" # Input file to download docs with: NEWURLS_MEDQUAD_NIDDK_QA_DATASET.csv
    
    merge_medical_qa_dataset(original_dataset_path, wayback_urls_path, output_path)

if __name__ == "__main__":
    main()