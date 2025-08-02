# Preprocess documents for RAG:
# De-duplicate rows in medquad_NIDDK_qa_dataset.csv based on "document_id" column.
# For each resulting unique input row, do the following and then print the values in the output file NIDDK_doc_list.csv:
# 1. Parse the webpage given by "document_url", and if it's expired with a 404 error, make a flag 1 in column "expired_document_url" else 0.
# 2. Also, preserve all other columns as they are: qid	qtype	unique_question_id	question_text	answer_text	document_id	document_source	document_url	document_focus


import pandas as pd
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import time
from tqdm import tqdm

def check_url_status(url):
    """
    Check if a URL is accessible or returns a 404 error
    Returns 1 if expired (404), 0 if accessible
    """
    try:
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 404:
            return 1
        return 0
    except requests.RequestException:
        # If there's any error in accessing the URL, mark it as expired
        return 1

def preprocess_medical_documents(input_file, output_file):
    """
    Preprocess medical documents from CSV file
    """
    try:
        # Read the input CSV file
        df = pd.read_csv(input_file)
        print(f"Original dataset shape: {df.shape}")
        
        # Remove duplicates based on document_id
        df_unique = df.drop_duplicates(subset=['document_id'])
        print(f"Dataset shape after removing duplicates: {df_unique.shape}")
        
        # Add new column for expired URL flag
        df_unique['expired_document_url'] = 0
        
        # Check each URL and update the expired flag
        print("Checking URLs status...")
        for idx in tqdm(df_unique.index):
            url = df_unique.loc[idx, 'document_url']
            if pd.notna(url):  # Check only if URL is not NaN
                df_unique.loc[idx, 'expired_document_url'] = check_url_status(url)
            time.sleep(1)  # Add delay to avoid overwhelming servers
        
        # Save the processed data
        columns_order = [
            'qid', 'qtype', 'unique_question_id', 'question_text', 
            'answer_text', 'document_id', 'document_source', 
            'document_url', 'document_focus', 'expired_document_url'
        ]
        
        df_unique[columns_order].to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")
        
        # Print summary statistics
        total_docs = len(df_unique)
        expired_docs = df_unique['expired_document_url'].sum()
        print(f"\nSummary:")
        print(f"Total unique documents: {total_docs}")
        print(f"Expired URLs: {expired_docs} ({(expired_docs/total_docs)*100:.2f}%)")
        print(f"Active URLs: {total_docs - expired_docs} ({((total_docs-expired_docs)/total_docs)*100:.2f}%)")
        
    except Exception as e:
        print(f"Error processing documents: {str(e)}")

if __name__ == "__main__":
    input_file = "medquad_NIDDK_qa_dataset.csv"
    output_file = "NIDDK_doc_list.csv"
    preprocess_medical_documents(input_file, output_file)