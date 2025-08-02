import os
from bs4 import BeautifulSoup
import re
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import logging
from dataclasses import dataclass
from collections import Counter
import hashlib
import nltk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessedDocument:
    url: str
    title: str
    cleaned_content: str
    structured_content: str
    stats: Dict[str, int]

@dataclass
class ChunkConfig:
    chunk_size: int = 200
    chunk_overlap: int = 0
    min_chunk_size: int = 50
    max_chunk_size: int = 300
    respect_sentence_boundaries: bool = True
    clean_whitespace: bool = True
    remove_empty_lines: bool = True

class DocumentChunker:
    def __init__(self, config: ChunkConfig = None):
        self.config = config or ChunkConfig()

    def count_words(self, text: str) -> int:
        return len(text.split())

    def clean_text(self, text: str) -> str:
        if self.config.clean_whitespace:
            text = re.sub(r'\s+', ' ', text)
        if self.config.remove_empty_lines:
            lines = [line.strip() for line in text.splitlines()]
            text = '\n'.join(line for line in lines if line)
        return text.strip()

    def create_chunk_hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def split_into_sentences(self, text: str) -> List[str]:
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        return nltk.sent_tokenize(text)

    def create_chunk(self, text: str, metadata: Dict) -> Dict:
        return {
            'content': text,
            'hash': self.create_chunk_hash(text),
            'word_count': self.count_words(text)
        }

    def split_into_semantic_chunks(self, text: str, metadata: Dict) -> List[Dict]:
        text = self.clean_text(text)
        chunks = []

        if self.config.respect_sentence_boundaries:
            sentences = self.split_into_sentences(text)
            current_chunk = []
            current_word_count = 0

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                sentence_words = self.count_words(sentence)

                if current_word_count + sentence_words > self.config.chunk_size:
                    if current_chunk and current_word_count >= self.config.min_chunk_size:
                        chunk_text = ' '.join(current_chunk)
                        chunks.append(self.create_chunk(chunk_text, metadata))
                    current_chunk = []
                    current_word_count = 0

                current_chunk.append(sentence)
                current_word_count += sentence_words

            if current_chunk and current_word_count >= self.config.min_chunk_size:
                chunk_text = ' '.join(current_chunk)
                chunks.append(self.create_chunk(chunk_text, metadata))
        return chunks

class MedicalDocumentProcessor:
    def __init__(self, docs_dir: str, output_dir: str, boilerplate_threshold_percentage: float = 0.1):
        self.docs_dir = Path(docs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.boilerplate_threshold_percentage = boilerplate_threshold_percentage

    def extract_url_and_title(self, content: str) -> tuple[str, str]:
        url_match = re.search(r'URL:\s*(http[^\n]+)', content)
        title_match = re.search(r'Title:\s*([^\n]+)', content)

        url = url_match.group(1).strip() if url_match else ""
        title = title_match.group(1).strip() if title_match else ""

        return url, title

    def extract_cleaned_content(self, content: str) -> str:
        """Extract and clean CLEANED_CONTENT section with minimal processing."""
        # Find the CLEANED_CONTENT section
        cleaned_match = re.search(r'CLEANED_CONTENT:(.*?)(?:={20,}|\Z)', content, re.DOTALL)
        if not cleaned_match:
            return ""

        text = cleaned_match.group(1).strip()

        # Remove large boilerplate blocks
        boilerplate_blocks = [
            r"View the web archive through the Wayback Machine.*?End of Term Web Archive",
            r"Collection: End of Term.*?For more information",
            r"U\.S\. Department of Health and Human Services.*?Contact Us \| Jobs at NIDDK \| Get Email Updates",
            r"Research and Funding for Scientists.*?Current Funding Opportunities",
            r"Health Information.*?La información de la salud en español",
            r"This content is provided as a service.*?NIH\.\.\.Turning Discovery Into Health",
            r"Contact the NIDDK Health Information Center.*?Clinical Trials"
        ]

        for block in boilerplate_blocks:
            text = re.sub(block, '', text, flags=re.DOTALL | re.IGNORECASE)

        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)

        # Basic text cleanup
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Remove excessive newlines

        return text.strip()

    def detect_boilerplate_chunks(self, processed_documents: List[ProcessedDocument]) -> List[str]:
        chunk_counts = Counter()
        temp_chunker = DocumentChunker(config=ChunkConfig())

        for doc in processed_documents:
            if not doc.cleaned_content:
                continue
            metadata = {"source_file": doc.title}
            chunks = temp_chunker.split_into_semantic_chunks(doc.cleaned_content, metadata)
            for chunk_data in chunks:
                chunk_counts[chunk_data['hash']] += 1

        boilerplate_chunk_hashes = []
        threshold = len(processed_documents) * self.boilerplate_threshold_percentage
        logger.info(f"Boilerplate threshold: {threshold} documents (percentage: {self.boilerplate_threshold_percentage*100}%)")

        for chunk_hash, count in chunk_counts.items():
            if count > threshold:
                boilerplate_chunk_hashes.append(chunk_hash)
        logger.info(f"Detected {len(boilerplate_chunk_hashes)} boilerplate chunks.")
        return boilerplate_chunk_hashes

    def process_document(self, file_path: Path, boilerplate_hashes: List[str]) -> Optional[ProcessedDocument]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            url, title = self.extract_url_and_title(content)
            cleaned_content = self.extract_cleaned_content(content)
            
            if not cleaned_content:
                logger.warning(f"No cleaned content found in {file_path}")
                return None

            # Create structured content
            sections = []
            current_section = []
            lines = cleaned_content.split('\n')

            for line in lines:
                if re.match(r'^[A-Z][^.?!]+[.?!]', line.strip()):
                    if current_section:
                        sections.append('\n'.join(current_section))
                        current_section = []
                current_section.append(line.strip())

            if current_section:
                sections.append('\n'.join(current_section))
            structured_content = '\n\n'.join(sections)

            return ProcessedDocument(
                url=url,
                title=title,
                cleaned_content=cleaned_content,
                structured_content=structured_content,
                stats={'total_blocks': len(sections), 'valid_blocks': len(sections)}
            )

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return None

    def process_documents(self):
        stats = {
            'files_processed': 0,
            'successful_files': 0,
            'total_blocks': 0,
            'valid_blocks': 0,
            'failed_files': []
        }
        processed_docs_list = []

        for file_path in self.docs_dir.glob('*.txt'):
            try:
                stats['files_processed'] += 1
                processed_doc = self.process_document(file_path, [])

                if processed_doc and (processed_doc.cleaned_content or processed_doc.structured_content):
                    stats['successful_files'] += 1
                    stats['total_blocks'] += processed_doc.stats['total_blocks']
                    stats['valid_blocks'] += processed_doc.stats['valid_blocks']
                    processed_docs_list.append(processed_doc)

                    output_file = self.output_dir / f"processed_{file_path.name}"
                    output_data = {
                        "url": processed_doc.url,
                        "title": processed_doc.title,
                        "cleaned_content": processed_doc.cleaned_content,
                        "structured_content": processed_doc.structured_content,
                        "stats": processed_doc.stats
                    }

                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(output_data, f, indent=2, ensure_ascii=False)

                    logger.info(f"Successfully processed {file_path.name}")
                else:
                    stats['failed_files'].append(str(file_path))

            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                stats['failed_files'].append(str(file_path))
                continue

        boilerplate_hashes = self.detect_boilerplate_chunks(processed_docs_list)
        boilerplate_output_file = self.output_dir / 'boilerplate_chunk_hashes.json'
        with open(boilerplate_output_file, 'w') as f:
            json.dump(boilerplate_hashes, f, indent=2)
        logger.info(f"Saved {len(boilerplate_hashes)} boilerplate chunk hashes to {boilerplate_output_file}")

        stats_file = self.output_dir / 'processing_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Processing complete. Success rate: {stats['successful_files']}/{stats['files_processed']}")
        if stats['failed_files']:
            logger.warning(f"Failed files: {stats['failed_files']}")

        return stats

if __name__ == "__main__":
    processor = MedicalDocumentProcessor("MEDQUAD_DOCS", "processed_docs", boilerplate_threshold_percentage=0.2)
    processor.process_documents()