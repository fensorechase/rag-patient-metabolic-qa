# Updated main_rag_prepare.py
import logging
import json
from pathlib import Path
from doc_preprocessor import MedicalDocumentProcessor
from chunker import DocumentChunker, ChunkConfig
import embedder
import argparse # Import argparse

logging.basicConfig(level=logging.DEBUG)

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_processed_docs_exist(processed_dir: str) -> bool:
    """Check if processed documents exist and are valid."""
    processed_path = Path(processed_dir)

    if not processed_path.exists():
        return False

    processed_files = list(processed_path.glob('processed_*.txt'))
    if not processed_files:
        return False

    # Check at least one file has content
    for file_path in processed_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if data.get('cleaned_content') or data.get('structured_content'):
                    return True
        except Exception:
            continue

    return False

def load_processing_stats(processed_dir: str) -> dict:
    """Load and validate processing statistics."""
    stats_file = Path(processed_dir) / 'processing_stats.json'
    try:
        with open(stats_file, 'r') as f:
            stats = json.load(f)

        # Validate required fields
        required_fields = {'files_processed', 'successful_files', 'total_blocks', 'valid_blocks'}
        if not all(field in stats for field in required_fields):
            logger.warning("Stats file missing required fields")
            return {}

        return stats
    except Exception as e:
        logger.error(f"Error loading processing stats: {str(e)}")
        return {}

def validate_chunks(chunks: list) -> bool:
    """Validate chunk format and content."""
    if not chunks:
        return False

    required_fields = {'content', 'metadata', 'hash'}
    valid_chunks = 0

    for chunk in chunks:
        if not all(field in chunk for field in required_fields):
            logger.error(f"Chunk missing required fields. Found: {chunk.keys()}")
            continue

        # Validate content is not empty
        if not chunk['content'].strip():
            logger.error("Empty chunk content found")
            continue

        valid_chunks += 1

    # Require at least one valid chunk
    return valid_chunks > 0

def main():
    processed_dir = "processed_docs"
    chunked_dir = "chunked_docs"

    # Step 1: Process documents
    processor = MedicalDocumentProcessor("MEDQUAD_DOCS", processed_dir)

    if not check_processed_docs_exist(processed_dir):
        logger.info("Starting document processing...")
        try:
            stats = processor.process_documents()
            if stats['successful_files'] == 0:
                raise ValueError("No documents were successfully processed")

            logger.info(f"Successfully processed {stats['successful_files']} documents")

        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise
    else:
        logger.info("Found existing processed documents")

    # Log processing statistics
    stats = load_processing_stats(processed_dir)
    if stats:
        logger.info(f"Processing summary:")
        logger.info(f"- Files processed: {stats['files_processed']}")
        logger.info(f"- Successful files: {stats['successful_files']}")
        logger.info(f"- Total blocks: {stats['total_blocks']}")
        logger.info(f"- Valid blocks: {stats['valid_blocks']}")

        if stats.get('failed_files'):
            logger.warning(f"Failed files: {len(stats['failed_files'])}")

    # Step 2: Create chunks
    logger.info("Starting document chunking...")
    try:
        chunk_config = ChunkConfig( # Create ChunkConfig object here
            chunk_size=150, # Old: 200
            chunk_overlap=0, # Old: 100
            min_chunk_size=50,
            max_chunk_size=400, # Old: 800
            respect_sentence_boundaries=True,
            clean_whitespace=True,
            remove_empty_lines=True
        )
        chunker = DocumentChunker(
            processed_dir,
            chunked_dir,
            boilerplate_hashes_file="processed_docs/boilerplate_chunk_hashes.json", # Correct argument order, pass boilerplate_hashes_file path
            config=chunk_config # Pass ChunkConfig object as config argument
        )

        chunks = chunker.process_chunks()

        if not chunks:
            logger.error("No chunks were generated. Checking processed docs...")
            processed_files = list(Path(processed_dir).glob('processed_*.txt'))
            if not processed_files:
                raise ValueError("No valid processed documents found")
            else:
                # Check content of processed files
                for file_path in processed_files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            doc_data = json.load(f)
                            if not doc_data.get('cleaned_content') and not doc_data.get('structured_content'):
                                logger.warning(f"No content found in {file_path.name}")
                            else:
                                logger.info(f"Found valid content in {file_path.name}")
                    except Exception as e:
                        logger.error(f"Error reading {file_path.name}: {str(e)}")

        if not validate_chunks(chunks):
            raise ValueError("Generated chunks are invalid")

        logger.info(f"Successfully created {len(chunks)} chunks")

    except Exception as e:
        logger.error(f"Chunking failed: {str(e)}")
        raise

    # Step 3: Generate embeddings
    logger.info("Starting embedding generation...")
    try:
        embedder.main()
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()