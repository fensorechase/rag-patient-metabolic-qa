# chunker.py (Enhanced Version with main_rag_prepare.py compatibility)
from typing import List, Dict
import json
from pathlib import Path
from dataclasses import dataclass
import hashlib
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
import re

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChunkConfig:
    # Keep the original parameter names and defaults for compatibility
    chunk_size: int = 300
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    max_chunk_size: int = 500
    respect_sentence_boundaries: bool = True
    clean_whitespace: bool = True
    remove_empty_lines: bool = True
    include_section_headers: bool = True  # New parameter, but won't break existing code

class DocumentChunker:
    def __init__(self, input_dir: str, output_dir: str, boilerplate_hashes_file: str = None, config: ChunkConfig = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or ChunkConfig()
        self.boilerplate_hashes = set()
        if boilerplate_hashes_file:
            self.load_boilerplate_hashes(boilerplate_hashes_file)

    def load_boilerplate_hashes(self, boilerplate_hashes_file: str):
        """Load boilerplate chunk hashes from JSON file."""
        try:
            with open(boilerplate_hashes_file, 'r') as f:
                self.boilerplate_hashes = set(json.load(f))
            logger.info(f"Loaded {len(self.boilerplate_hashes)} boilerplate chunk hashes from {boilerplate_hashes_file}")
        except FileNotFoundError:
            logger.warning(f"Boilerplate hash file not found: {boilerplate_hashes_file}. Boilerplate filtering will be skipped.")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding boilerplate hash file {boilerplate_hashes_file}: {e}. Boilerplate filtering will be skipped.")

    def count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())

    def clean_text(self, text: str) -> str:
        """Clean text while preserving section headers and structure."""
        if self.config.clean_whitespace:
            text = re.sub(r'\s+', ' ', text)
        
        if self.config.remove_empty_lines:
            lines = [line.strip() for line in text.splitlines()]
            text = '\n'.join(line for line in lines if line)

        # Preserve section headers while cleaning
        if self.config.include_section_headers:
            text = re.sub(r'(?<=[.!?])\s+(?=[A-Z])', '\n\n', text)

        # Remove standard boilerplate
        text = re.sub(r'View or Print All Sections|Clinical Trials.*?digestive diseases\.', '', text, flags=re.DOTALL)
        return text.strip()

    def create_chunk_hash(self, text: str) -> str:
        """Create hash of chunk content for deduplication."""
        return hashlib.md5(text.encode()).hexdigest()

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK."""
        return nltk.sent_tokenize(text)

    def create_chunk(self, text: str, metadata: Dict) -> Dict:
        """Create a chunk with enhanced metadata and content structure."""
        # Include document focus and title in content for better context
        title = metadata.get('title', '')
        focus = metadata.get('document_focus', '')
        section_title = metadata.get('section_title', '')
        
        enhanced_text = text
        if self.config.include_section_headers:
            prefix_parts = []
            if title:
                prefix_parts.append(f"Title: {title}")
            if focus:
                prefix_parts.append(f"Topic: {focus}")
            if section_title:
                prefix_parts.append(f"Section: {section_title}")
            
            if prefix_parts:
                enhanced_text = f"{' | '.join(prefix_parts)}\n\n{enhanced_text}"

        return {
            'content': enhanced_text,
            'metadata': {
                'url': metadata.get('url', ''),
                'title': title,
                'source_file': metadata.get('source_file', ''),
                'chunk_hash': self.create_chunk_hash(enhanced_text),
                'word_count': self.count_words(enhanced_text),
                'section_title': section_title,
                'document_focus': focus,
                'original_document': metadata.get('original_document', {}),
                'structured_content': metadata.get('structured_content', '')
            },
            'hash': self.create_chunk_hash(enhanced_text),
            'word_count': self.count_words(enhanced_text)
        }

    def split_into_semantic_chunks(self, text: str, metadata: Dict) -> List[Dict]:
        """Split text into semantic chunks with section handling."""
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
                        chunk = self.create_chunk(chunk_text, metadata)
                        if chunk['hash'] not in self.boilerplate_hashes:
                            chunks.append(chunk)

                    # Handle overlap
                    if self.config.chunk_overlap > 0 and current_chunk:
                        overlap_sentences = current_chunk[-self.config.chunk_overlap:]
                        current_chunk = overlap_sentences
                        current_word_count = self.count_words(" ".join(overlap_sentences))
                    else:
                        current_chunk = []
                        current_word_count = 0

                current_chunk.append(sentence)
                current_word_count += sentence_words

            # Handle remaining text
            if current_chunk and current_word_count >= self.config.min_chunk_size:
                chunk_text = ' '.join(current_chunk)
                chunk = self.create_chunk(chunk_text, metadata)
                if chunk['hash'] not in self.boilerplate_hashes:
                    chunks.append(chunk)

        else:
            # Word-based chunking (fallback)
            words = text.split()
            current_chunk = []
            current_word_count = 0

            for word in words:
                if current_word_count + 1 > self.config.chunk_size:
                    if current_word_count >= self.config.min_chunk_size:
                        chunk_text = ' '.join(current_chunk)
                        chunk = self.create_chunk(chunk_text, metadata)
                        if chunk['hash'] not in self.boilerplate_hashes:
                            chunks.append(chunk)
                    current_chunk = []
                    current_word_count = 0

                current_chunk.append(word)
                current_word_count += 1

            if current_chunk and current_word_count >= self.config.min_chunk_size:
                chunk_text = ' '.join(current_chunk)
                chunk = self.create_chunk(chunk_text, metadata)
                if chunk['hash'] not in self.boilerplate_hashes:
                    chunks.append(chunk)

        return chunks

    def process_chunks(self):
        """Process all documents into chunks with enhanced metadata, structure, and cross-document similarity detection."""
        
        all_chunks = []
        unique_chunks = {}
        processed_files = list(self.input_dir.glob('processed_*.txt'))
        
        if not processed_files:
            logger.error("No processed files found in input directory")
            return []
        
        logger.info(f"Found {len(processed_files)} processed files to chunk")
        
        # Store chunks by document for similarity comparison
        chunks_by_doc = {}
        
        # First pass: Generate chunks for each document
        for file_path in processed_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc_data = json.load(f)
                
                base_metadata = {
                    'url': doc_data.get('url', ''),
                    'title': doc_data.get('title', ''),
                    'source_file': str(file_path),
                    'document_focus': doc_data.get('document_focus', ''),
                    'original_document': doc_data,
                    'structured_content': doc_data.get('structured_content', '')
                }
                
                content = doc_data.get('cleaned_content', '') or '\n\n'.join(doc_data.get('sections', []))
                
                if not content.strip():
                    logger.warning(f"No valid content found in {file_path}")
                    continue
                
                doc_data['stats'] = doc_data.get('stats', {})
                doc_data['stats']['valid_blocks'] = max(1, doc_data['stats'].get('valid_blocks', 0))
                
                chunks = self.split_into_semantic_chunks(content, base_metadata)
                chunks_by_doc[str(file_path)] = chunks
                
            except Exception as e:
                logger.error(f"Error processing chunks for {file_path.name}: {str(e)}")
                continue
        
        # Second pass: Detect and filter similar chunks across documents
        all_chunk_texts = []
        chunk_metadata = []
        
        for doc_path, doc_chunks in chunks_by_doc.items():
            for chunk in doc_chunks:
                all_chunk_texts.append(chunk['content'])
                chunk_metadata.append({
                    'doc_path': doc_path,
                    'chunk': chunk
                })
        
        if not all_chunk_texts:
            logger.warning("No valid chunks found in any document")
            return []
        
        # Calculate TF-IDF vectors and similarity matrix
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(all_chunk_texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Track which chunks to keep
            chunks_to_keep = set()
            similarity_threshold = 0.85  # Adjust this threshold as needed
            
            for i in range(len(all_chunk_texts)):
                current_doc = chunk_metadata[i]['doc_path']
                should_keep = True
                
                # Compare with all previous chunks
                for j in range(i):
                    if i != j and chunk_metadata[j]['doc_path'] != current_doc:
                        if similarity_matrix[i, j] > similarity_threshold:
                            # If chunks are very similar and from different documents,
                            # mark the current chunk for exclusion
                            should_keep = False
                            break
                
                if should_keep:
                    chunks_to_keep.add(i)
            
            # Create final unique chunks list
            for i in chunks_to_keep:
                chunk = chunk_metadata[i]['chunk']
                chunk_hash = chunk['hash']
                if chunk_hash not in unique_chunks:
                    unique_chunks[chunk_hash] = chunk
            
            logger.info(f"Filtered out {len(all_chunk_texts) - len(chunks_to_keep)} similar chunks across documents")
            
        except Exception as e:
            logger.error(f"Error during similarity detection: {str(e)}")
            # Fallback: use all chunks without similarity filtering
            for meta in chunk_metadata:
                chunk = meta['chunk']
                chunk_hash = chunk['hash']
                if chunk_hash not in unique_chunks:
                    unique_chunks[chunk_hash] = chunk
        
        # Save unique chunks
        if unique_chunks:
            unique_chunks_list = list(unique_chunks.values())
            output_file = self.output_dir / "unique_chunks.json"
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(unique_chunks_list, f, indent=2)
                logger.info(f"Saved {len(unique_chunks_list)} unique chunks to {output_file}")
            except Exception as e:
                logger.error(f"Error saving chunks to file: {str(e)}")
        
        return list(unique_chunks.values())

if __name__ == "__main__":
    chunker = DocumentChunker(
        "processed_docs", 
        "chunked_docs", 
        boilerplate_hashes_file="processed_docs/boilerplate_chunk_hashes.json"
    )
    chunker.process_chunks()