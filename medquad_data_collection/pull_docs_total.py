# pull_docs_total.py
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import hashlib
import time
from tqdm import tqdm
import re
from urllib.parse import urlparse
import logging
from pathlib import Path

def setup_logging():
    """Configure logging for the script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('webpage_download.log'),
            logging.StreamHandler()
        ]
    )

def create_safe_filename(url):
    """Create a safe filename from URL"""
    # Create MD5 hash of the URL
    url_hash = hashlib.md5(url.encode()).hexdigest()
    # Get the domain name
    domain = urlparse(url).netloc
    # Create filename using domain and hash
    return f"{domain}_{url_hash}.txt"

def extract_webpage_content(response):
    """Extract and clean webpage content"""
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Remove script and style elements
    for element in soup(['script', 'style', 'meta', 'link']):
        element.decompose()
    
    # Get main content
    content = {
        'title': soup.title.string if soup.title else 'No title',
        'h1': soup.h1.get_text().strip() if soup.h1 else 'No H1',
        'body': soup.get_text(separator=' ', strip=True),
        'raw_html': str(soup)
    }
    
    return content

def download_webpage(url, output_dir, headers, max_retries=3, wait_time=120):
    """Download webpage content with retries"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=wait_time)
            response.raise_for_status()
            return extract_webpage_content(response)
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                logging.error(f"Failed to download {url}: {str(e)}")
                return None
            time.sleep(min(2 ** attempt, 30))  # Exponential backoff
    return None

def download_and_save_webpages(input_file, output_dir):
    """Main function to download and save webpage contents"""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging()
    
    # Read input CSV and get unique URLs
    df = pd.read_csv(input_file)
    unique_urls = df['valid_doc_url'].unique()
    logging.info(f"Found {len(unique_urls)} unique URLs to process")
    
    # Setup headers for requests
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5'
    }
    
    # Track successful and failed downloads
    successful = 0
    failed = 0
    
    # Process each unique URL
    for url in tqdm(unique_urls, desc="Downloading webpages"):
        # Create safe filename
        filename = create_safe_filename(url)
        filepath = os.path.join(output_dir, filename)
        
        # Skip if already downloaded
        if os.path.exists(filepath):
            logging.info(f"Skipping already downloaded URL: {url}")
            successful += 1
            continue
            
        # Download webpage
        content = download_webpage(url, output_dir, headers)
        
        if content:
            # Save content
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    # Save metadata and content
                    f.write(f"URL: {url}\n")
                    f.write(f"Title: {content['title']}\n")
                    f.write(f"H1: {content['h1']}\n")
                    f.write("="*80 + "\n")
                    f.write("CLEANED_CONTENT:\n")
                    f.write(content['body'])
                    f.write("\n" + "="*80 + "\n")
                    f.write("RAW_HTML:\n")
                    f.write(content['raw_html'])
                
                # Show progress with H1 preview
                h1_preview = ' '.join(content['h1'].split()[:20])
                logging.info(f"Downloaded: {url}\nH1 Preview: {h1_preview}")
                successful += 1
                
            except Exception as e:
                logging.error(f"Error saving content for {url}: {str(e)}")
                failed += 1
        else:
            failed += 1
        
        # Add delay between requests
        time.sleep(2)
    
    # Print final summary
    logging.info("\nDownload Summary:")
    logging.info(f"Total URLs processed: {len(unique_urls)}")
    logging.info(f"Successfully downloaded: {successful}")
    logging.info(f"Failed: {failed}")

def main():
    input_file = "NEWURLS_MEDQUAD_NIDDK_QA_DATASET.csv"
    output_dir = "MEDQUAD_DOCS"
    download_and_save_webpages(input_file, output_dir)

if __name__ == "__main__":
    main()