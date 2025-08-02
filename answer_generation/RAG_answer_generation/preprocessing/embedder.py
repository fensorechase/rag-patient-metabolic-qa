# embedder.py

import torch
from transformers import AutoTokenizer, AutoModel
import chromadb
from typing import List, Dict
import pandas as pd
import json
from pathlib import Path
import logging
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Initialize Chroma with persistent storage
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        
        # Enable mixed precision
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Optimize model for inference
        self.model.eval()
        if hasattr(self.model, "half"):
            self.model = self.model.half()  # Use FP16
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Generate embeddings with mixed precision"""
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Use mixed precision
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
                    embeddings.extend(batch_embeddings)
            
            # Optional: Clear cache every few batches
            if i % 100 == 0:
                torch.cuda.empty_cache()
        
        return np.array(embeddings)
    
    def create_chroma_collection(self, chunks: List[Dict], collection_name: str = "medical_qa"):
        """Create and populate Chroma collection"""
        logger.info(f"Processing {len(chunks)} chunks")
        
        # Log sample chunk structure
        if chunks:
            logger.info(f"Sample chunk structure: {json.dumps(chunks[0], indent=2)}")
        else:
            logger.error("No chunks found!")
            return None
            
        # Check if collection exists using new API
        existing_collections = self.chroma_client.list_collections()
        
        # Delete if exists
        if collection_name in existing_collections:
            self.chroma_client.delete_collection(name=collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
            
        # Create new collection
        collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Extract sections from chunks
        sections = []
        metadata_list = []
        ids = []
        
        for chunk in chunks:
            # Handle both possible chunk structures
            if 'sections' in chunk:
                # If chunks contain multiple sections
                for section in chunk['sections']:
                    sections.append(section)
                    metadata_list.append({
                        'url': chunk.get('metadata', {}).get('url', ''),
                        'title': chunk.get('metadata', {}).get('title', ''),
                        'h1': chunk.get('metadata', {}).get('h1', '')
                    })
                    ids.append(chunk.get('hash', str(len(ids))))
            else:
                # If chunks are already individual sections
                sections.append(chunk.get('content', ''))
                metadata_list.append({
                    'url': chunk.get('metadata', {}).get('url', ''),
                    'title': chunk.get('metadata', {}).get('title', ''),
                    'h1': chunk.get('metadata', {}).get('h1', '')
                })
                ids.append(chunk.get('hash', str(len(ids))))
        
        if not sections:
            logger.error("No valid sections found in chunks!")
            return None
            
        logger.info(f"Extracted {len(sections)} sections for processing")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.generate_embeddings(sections)
        
        # Add to collection in batches
        batch_size = 100
        total_batches = (len(sections) + batch_size - 1) // batch_size
        logger.info(f"Adding {len(sections)} documents in {total_batches} batches...")
        
        for i in range(0, len(sections), batch_size):
            end_idx = min(i + batch_size, len(sections))
            collection.add(
                embeddings=embeddings[i:end_idx].tolist(),
                documents=sections[i:end_idx],
                metadatas=metadata_list[i:end_idx],
                ids=ids[i:end_idx]
            )
            logger.info(f"Added batch {i//batch_size + 1} of {total_batches}")
        
        # Verify collection exists using new API
        try:
            self.chroma_client.get_collection(name=collection_name)
            logger.info(f"Successfully created and populated collection '{collection_name}' with {len(sections)} sections")
        except ValueError as e:
            raise RuntimeError(f"Failed to create collection '{collection_name}': {str(e)}")
        
        return embeddings
        
    def tune_parameters(self, qa_file: str, n_samples: int = 10):
        """Tune embedding parameters using QA pairs"""
        # Load QA data
        df = pd.read_csv(qa_file)
        df = df.sample(n=min(n_samples, len(df)))
        
        # Generate embeddings for questions and answers
        logger.info("Generating embeddings for parameter tuning...")
        question_embeddings = self.generate_embeddings(df['question_text'].tolist())
        answer_embeddings = self.generate_embeddings(df['answer_text'].tolist())
        
        # Calculate cosine similarities
        similarities = np.sum(question_embeddings * answer_embeddings, axis=1) / (
            np.linalg.norm(question_embeddings, axis=1) * np.linalg.norm(answer_embeddings, axis=1)
        )
        
        # Log results
        logger.info(f"Average Q-A similarity: {np.mean(similarities):.4f}")
        logger.info(f"Std Q-A similarity: {np.std(similarities):.4f}")
        
        return {
            'mean_similarity': float(np.mean(similarities)),
            'std_similarity': float(np.std(similarities))
        }

def main():
    try:
        # Load chunks
        chunk_file = "chunked_docs/unique_chunks.json"
        logger.info(f"Loading chunks from {chunk_file}")
        
        if not Path(chunk_file).exists():
            raise FileNotFoundError(f"Chunk file not found: {chunk_file}")
            
        with open(chunk_file, 'r') as f:
            chunks = json.load(f)
            
        logger.info(f"Loaded chunks from file. Number of chunks: {len(chunks)}")
        
        # Validate chunk structure
        if not chunks:
            raise ValueError("No chunks loaded from file")
            
        # Log sample chunk for debugging
        logger.info(f"Sample chunk structure: {json.dumps(chunks[0], indent=2)}")
        
        embedder = DocumentEmbedder()
        
        # Tune parameters
        logger.info("Starting parameter tuning...")
        tune_results = embedder.tune_parameters("NEWURLS_MEDQUAD_NIDDK_QA_DATASET.csv")
        
        # Save tune results
        with open("tune_results.json", 'w') as f:
            json.dump(tune_results, f, indent=2)
        logger.info(f"Saved tuning results: {tune_results}")
        
        # Create embeddings and store in Chroma
        collection_name = 'medical_qa'
        logger.info(f"Creating and populating collection: {collection_name}")
        embedder.create_chroma_collection(chunks, collection_name)
        
    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error in embedder main: {str(e)}")
        raise

if __name__ == "__main__":
    main()