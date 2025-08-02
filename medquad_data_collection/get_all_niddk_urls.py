import pandas as pd
import requests
import time
from tqdm import tqdm
import json
import urllib.parse

def get_wayback_url(original_url):
    """
    Retrieve the most recent WayBack Machine archive URL for a given expired URL
    
    Args:
        original_url (str): The original URL that returned a 404
    
    Returns:
        str or None: The WayBack Machine URL if found, None otherwise
    """
    try:
        # Encode the original URL for the Wayback Machine API
        encoded_url = urllib.parse.quote(original_url)
        
        # Wayback Machine API endpoint for finding the most recent capture
        wayback_api_url = f"https://archive.org/wayback/available?url={encoded_url}"
        
        # Send request to Wayback Machine API
        response = requests.get(wayback_api_url, timeout=10)
        
        # Check if request was successful
        if response.status_code == 200:
            data = response.json()
            
            # Check if any snapshots are available
            if data.get('archived_snapshots', {}).get('closest', {}):
                wayback_url = data['archived_snapshots']['closest'].get('url')
                
                # Validate the retrieved URL
                if wayback_url:
                    # Additional check to ensure the URL is a valid Wayback Machine URL
                    if requests.head(wayback_url, timeout=10).status_code == 200:
                        return wayback_url
        
        return None
    
    except Exception as e:
        print(f"Error retrieving WayBack URL for {original_url}: {e}")
        return None

def process_expired_documents(input_file, output_file):
    """
    Process CSV file to find WayBack Machine URLs for expired documents
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
    """
    # Read the input CSV file
    df = pd.read_csv(input_file)
    
    # Filter rows with expired documents
    expired_docs = df[df['expired_document_url'] == 1].copy()
    print(f"Total expired documents to process: {len(expired_docs)}")
    
    # Add new column for WayBack Machine URLs
    df['new_doc_url'] = ''
    
    # Process each expired document
    for idx in tqdm(expired_docs.index):
        original_url = expired_docs.loc[idx, 'document_url']
        
        # Retrieve WayBack Machine URL
        wayback_url = get_wayback_url(original_url)
        
        # Update the DataFrame
        if wayback_url:
            df.loc[idx, 'new_doc_url'] = wayback_url
        
        # Implement rate limiting to avoid overwhelming the Wayback Machine API
        time.sleep(1)
    
    # Save the updated DataFrame
    df.to_csv(output_file, index=False)
    
    # Print summary statistics
    wayback_urls_found = df['new_doc_url'].notna().sum()
    print(f"\nSummary:")
    print(f"Total expired documents: {len(expired_docs)}")
    print(f"WayBack Machine URLs found: {wayback_urls_found}")
    print(f"Success rate: {(wayback_urls_found/len(expired_docs))*100:.2f}%")

def main():
    input_file = "NIDDK_doc_list.csv"
    output_file = "NIDDK_doc_list_with_wayback_urls.csv"
    process_expired_documents(input_file, output_file)

if __name__ == "__main__":
    main()