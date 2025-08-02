# corpusaware_exp_medical_rog.py
import pandas as pd
import chromadb
import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    AutoConfig, 
    AutoModel
)
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.util import ngrams
import logging
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from collections import defaultdict, Counter
import time
import pickle
import os
import glob
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import math

# Previous imports and configurations remain the same...
import argparse
from torch.cuda.amp import autocast
import datetime

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {
    "llama-70b": "/local/scratch/cfensor/huggingface_cache/quantized_llama_3.1_70B",  # Updated to local path
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
}

@dataclass
class RAGConfig:
    num_docs: int = 5
    max_tokens: int = 512
    temperature: float = 0.1
    model_name: str = "meta-llama/Llama-3.1-70B-Instruct"
    chroma_db_path: str = "chroma_db"
    hf_auth: str = ""
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    mmr_lambda: float = 0.5
    hybrid_search_weight: float = 0.5
    batch_size: int = 8 # Reduced from 16 for memory optimization
    num_gpus: int = 1 # Can set to 2 if needed.
    torch_dtype: str = "bfloat16"
    use_flash_attention: bool = True
    max_memory: Optional[Dict] = None
    tensor_parallel: bool = True
    initial_k: int = 10  # Retrieve more docs initially for reranking



@dataclass
class CorpusStats:
    """Statistics about the document corpus"""
    term_cooccurrence: Dict[str, Dict[str, float]]
    term_idf: Dict[str, float]
    bigram_scores: Dict[Tuple[str, str], float]
    global_term_weights: Dict[str, float]
    
    @classmethod
    def load(cls, cache_file: str) -> 'CorpusStats':
        """Load corpus statistics from cache"""
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    def save(self, cache_file: str):
        """Save corpus statistics to cache"""
        with open(cache_file, 'wb') as f:
            pickle.dump(self, f)

class CorpusAnalyzer:
    """Analyzes document corpus to extract term relationships"""
    
    def __init__(self, docs_dir: str, cache_dir: str = "cache"):
        self.docs_dir = Path(docs_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "corpus_stats.pkl"
        
    def get_corpus_stats(self, force_rebuild: bool = False) -> CorpusStats:
        """Get corpus statistics, using cache if available"""
        if not force_rebuild and self.cache_file.exists():
            return CorpusStats.load(self.cache_file)
        
        # Process all documents
        documents = []
        for doc_file in tqdm(glob.glob(str(self.docs_dir / "*.txt")), desc="Loading documents"):
            with open(doc_file, 'r') as f:
                doc_data = json.load(f)
                if 'cleaned_content' in doc_data:
                    documents.append(doc_data['cleaned_content'])
        
        # Build term statistics
        term_cooccurrence = self._build_term_cooccurrence(documents)
        term_idf = self._calculate_idf(documents)
        bigram_scores = self._extract_significant_bigrams(documents)
        global_weights = self._calculate_global_weights(documents)
        
        # Create and cache corpus stats
        stats = CorpusStats(
            term_cooccurrence=term_cooccurrence,
            term_idf=term_idf,
            bigram_scores=bigram_scores,
            global_term_weights=global_weights
        )
        stats.save(self.cache_file)
        return stats
    
    def _build_term_cooccurrence(self, documents: List[str]) -> Dict[str, Dict[str, float]]:
        """Build term co-occurrence matrix with PMI scores"""
        # Initialize counters
        term_counts = Counter()
        cooccurrence = defaultdict(Counter)
        total_windows = 0
        window_size = 5
        
        # Count terms and co-occurrences
        for doc in documents:
            tokens = word_tokenize(doc.lower())
            term_counts.update(tokens)
            
            # Count co-occurrences within windows
            for i in range(len(tokens)):
                window = tokens[max(0, i - window_size):min(len(tokens), i + window_size + 1)]
                for j, term1 in enumerate(window):
                    for term2 in window[j+1:]:
                        cooccurrence[term1][term2] += 1
                        cooccurrence[term2][term1] += 1
                total_windows += 1
        
        # Calculate PMI scores
        pmi_scores = defaultdict(dict)
        total_terms = sum(term_counts.values())
        
        for term1, cooccur in cooccurrence.items():
            p_term1 = term_counts[term1] / total_terms
            
            for term2, count in cooccur.items():
                p_term2 = term_counts[term2] / total_terms
                p_together = count / (total_windows * 2)  # * 2 because we counted both directions
                
                if p_together > 0:
                    pmi = math.log2(p_together / (p_term1 * p_term2))
                    if pmi > 0:  # Only keep positive PMI scores
                        pmi_scores[term1][term2] = pmi
        
        return dict(pmi_scores)
    
    def _calculate_idf(self, documents: List[str]) -> Dict[str, float]:
        """Calculate IDF scores for terms"""
        vectorizer = TfidfVectorizer(use_idf=True)
        vectorizer.fit_transform(documents)
        return dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
    
    def _extract_significant_bigrams(self, documents: List[str]) -> Dict[Tuple[str, str], float]:
        """Extract significant bigrams using likelihood ratios"""
        bigram_measures = BigramAssocMeasures()
        all_bigrams = defaultdict(int)
        
        for doc in documents:
            tokens = word_tokenize(doc.lower())
            finder = BigramCollocationFinder.from_words(tokens)
            finder.apply_freq_filter(3)  # Minimum frequency threshold
            scored = finder.score_ngrams(bigram_measures.likelihood_ratio)
            for bigram, score in scored:
                all_bigrams[bigram] += score
        
        # Normalize scores
        max_score = max(all_bigrams.values()) if all_bigrams else 1
        return {bigram: score/max_score for bigram, score in all_bigrams.items()}
    
    def _calculate_global_weights(self, documents: List[str]) -> Dict[str, float]:
        """Calculate global term weights using TF-IDF variance"""
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Calculate variance of TF-IDF scores for each term
        feature_names = vectorizer.get_feature_names_out()
        variances = np.asarray(tfidf_matrix.power(2).mean(axis=0) - 
                             np.power(tfidf_matrix.mean(axis=0), 2)).flatten()
        
        return dict(zip(feature_names, variances))

class MedicalRAGPipeline:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.setup_gpu_environment()
        self.setup_embedding_model()
        self.setup_cross_encoder()
        self.setup_llm()
        self.setup_chroma()
        self.stop_words = set(stopwords.words('english'))
        
        # Add embedding cache
        self.embedding_cache = {}
        
        # Initialize corpus analyzer and load statistics
        self.corpus_analyzer = CorpusAnalyzer("processed_docs")
        self.corpus_stats = self.corpus_analyzer.get_corpus_stats()

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding with caching"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        inputs = self.embedding_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.embedding_device)

        with torch.no_grad(), torch.amp.autocast('cuda'):
            try:
                outputs = self.embedding_model(**inputs)
                embedding = outputs.last_hidden_state[:, 0].cpu().numpy()
                self.embedding_cache[text] = embedding[0]
                
                # Clear cache if it gets too large
                if len(self.embedding_cache) > 10000:
                    self.embedding_cache.clear()
                    
                return embedding[0]
            except Exception as e:
                logger.error(f"Error during embedding generation: {str(e)}")
                raise
    
    def setup_gpu_environment(self):
        """Configure GPU environment for optimal performance"""
        if torch.cuda.is_available():
            # Get actual number of available GPUs
            available_gpus = torch.cuda.device_count()
            self.config.num_gpus = min(self.config.num_gpus, available_gpus)
            
            logger.info(f"Using {self.config.num_gpus} GPU(s) out of {available_gpus} available")
            
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            self.compute_dtype = getattr(torch, self.config.torch_dtype)

            if self.config.num_gpus > 1 and self.config.tensor_parallel:
                os.environ["TOKENIZERS_PARALLELISM"] = "true"

            # Simplified max_memory calculation with proper unit formatting
            if self.config.max_memory:
                try:
                    self.max_memory = eval(self.config.max_memory)
                except:
                    logger.warning("Error evaluating max_memory string, using default calculation.")
                    total_mem = torch.cuda.get_device_properties(0).total_memory
                    mem_per_gpu = total_mem * 0.8 / self.config.num_gpus
                    self.max_memory = {i: f"{int(mem_per_gpu / 1024 / 1024 / 1024)}GB" 
                                    for i in range(self.config.num_gpus)}
            else:
                total_mem = torch.cuda.get_device_properties(0).total_memory
                mem_per_gpu = total_mem * 0.8 / self.config.num_gpus
                self.max_memory = {i: f"{int(mem_per_gpu / 1024 / 1024 / 1024)}GB" 
                                for i in range(self.config.num_gpus)}

            # Ensure max_memory values are properly formatted strings
            for key in self.max_memory:
                if not isinstance(self.max_memory[key], str):
                    # Convert to GB and ensure proper formatting
                    mem_gb = int(float(self.max_memory[key]) / (1024 * 1024 * 1024))
                    self.max_memory[key] = f"{mem_gb}GB"

            logger.debug(f"Max memory configuration: {self.max_memory}")

    def setup_embedding_model(self):
        """Initialize the embedding model with optimizations"""
        logger.info(f"Loading embedding model: {self.config.embedding_model}")

        # Force embedding model to GPU 0
        self.embedding_device = torch.device("cuda:0")
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(self.config.embedding_model)

        # Load model and force it to cuda:0
        with torch.cuda.amp.autocast():
            self.embedding_model = AutoModel.from_pretrained(
                self.config.embedding_model,
                torch_dtype=self.compute_dtype,
                device_map={"": 0}  # Force model to GPU 0
            )

        self.embedding_model.eval()

    def setup_cross_encoder(self):
        """Initialize cross-encoder for reranking"""
        self.cross_encoder = CrossEncoder(
            self.config.cross_encoder_model,
            device='cuda:0',
            max_length=512
        )


    def expand_question(self, question: str) -> str:
        """Expand query using corpus statistics"""
        tokens = word_tokenize(question.lower())
        expanded_terms = set()
        
        # Add original terms
        expanded_terms.update(tokens)
        
        # Add terms with high co-occurrence scores
        for token in tokens:
            if token in self.corpus_stats.term_cooccurrence:
                # Get top co-occurring terms
                cooccurring = sorted(
                    self.corpus_stats.term_cooccurrence[token].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]  # Top 3 co-occurring terms
                
                expanded_terms.update(term for term, score in cooccurring)
        
        # Add significant bigrams
        token_pairs = list(ngrams(tokens, 2))
        for pair in token_pairs:
            if pair in self.corpus_stats.bigram_scores:
                score = self.corpus_stats.bigram_scores[pair]
                if score > 0.5:  # Significant bigram threshold
                    expanded_terms.add(' '.join(pair))
        
        # Weight terms by global importance
        weighted_terms = []
        for term in expanded_terms:
            weight = self.corpus_stats.global_term_weights.get(term, 1.0)
            # Repeat terms based on weight quantiles
            repeats = 1 + int(weight * 3)  # Max 4 repeats for highest weighted terms
            weighted_terms.extend([term] * repeats)
        
        # Combine original question with weighted expansions
        expanded_query = f"{question} {question} " + " ".join(weighted_terms)
        return expanded_query

    # ... [rest of the MedicalRAGPipeline class remains the same]



    def compute_bm25_scores(self, query: str, documents: List[str]) -> np.ndarray:
        """Compute BM25 scores for documents"""
        # Tokenize documents
        tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
        
        # Create BM25 model
        bm25 = BM25Okapi(tokenized_docs)
        
        # Get scores
        query_tokens = word_tokenize(query.lower())
        scores = np.array(bm25.get_scores(query_tokens))
        
        # Normalize scores
        if len(scores) > 0:
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        return scores



    def _extract_key_terms(self, question: str) -> List[str]:
        """Extract key medical terms from question without using predefined lists"""
        # Remove common question words
        question_words = {'what', 'is', 'are', 'how', 'many', 'people', 'affected', 'by'}
        
        # Tokenize and clean
        tokens = word_tokenize(question.lower())
        
        # Extract potential key terms (words not in question_words or stop_words)
        key_terms = [
            token for token in tokens 
            if token not in question_words 
            and token not in self.stop_words
            and len(token) > 2  # Avoid very short words
        ]
        
        # Also check for multi-word terms using corpus statistics
        bigrams = list(ngrams(tokens, 2))
        for bigram in bigrams:
            if bigram in self.corpus_stats.bigram_scores:
                if self.corpus_stats.bigram_scores[bigram] > 0.5:  # Significant bigram
                    key_terms.append(' '.join(bigram))
        
        return key_terms


    def add_chunk_context(self, documents: List[str], metadatas: List[Dict]) -> List[str]:
        """Add surrounding chunk context to improve relevance scoring"""
        chunks_with_context = []
        
        for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
            context = []
            
            # Find chunks from same document
            doc_id = metadata.get('document_id', '')
            section = metadata.get('section_title', '')
            
            for j, other_meta in enumerate(metadatas):
                if j != i and other_meta.get('document_id') == doc_id:
                    # Prioritize chunks from same section
                    if other_meta.get('section_title') == section:
                        context.append(documents[j])
            
            # Combine with original chunk
            if context:
                context_str = " ... ".join(context[:2])  # Limit context size
                enhanced_doc = f"{doc}\nRelated Context: {context_str}"
            else:
                enhanced_doc = doc
                
            chunks_with_context.append(enhanced_doc)
        
        return chunks_with_context
    

    def compute_chunk_similarity(self, documents: List[str]) -> np.ndarray:
        """Compute similarity between chunks to identify related content"""
        n_docs = len(documents)
        similarity_scores = np.zeros(n_docs)
        
        # Generate embeddings for all chunks
        chunk_embeddings = []
        for doc in documents:
            with torch.cuda.device(0):
                emb = self.generate_embedding(doc)
                chunk_embeddings.append(emb)
        
        chunk_embeddings = np.array(chunk_embeddings)
        
        # Compute pairwise similarities
        for i in range(n_docs):
            # Cosine similarity with other chunks
            similarities = np.dot(chunk_embeddings[i], chunk_embeddings.T)
            similarities = similarities / (
                np.linalg.norm(chunk_embeddings[i]) * 
                np.linalg.norm(chunk_embeddings, axis=1)
            )
            
            # Remove self-similarity
            similarities[i] = 0
            
            # Take weighted average of top similarities
            top_k = min(3, len(similarities) - 1)
            top_sims = np.sort(similarities)[-top_k:]
            
            # Exponentially weighted sum
            weights = np.exp(np.arange(top_k))
            similarity_scores[i] = np.sum(top_sims * weights) / np.sum(weights)
        
        # Normalize scores
        if len(similarity_scores) > 0:
            similarity_scores = (similarity_scores - similarity_scores.min()) / (
                similarity_scores.max() - similarity_scores.min() + 1e-8
            )
        
        return similarity_scores
    


    def get_relevant_documents(self, question: str) -> Tuple[List[Dict], List[float]]:
        """Enhanced document retrieval with key term boosting and guaranteed minimum documents"""
        try:
            # Extract key medical terms from question
            key_terms = self._extract_key_terms(question)
            expanded_question = self.expand_question(question)
            
            with torch.cuda.device(0):
                # Generate embeddings
                question_embedding = self.generate_embedding(expanded_question)
                
                # Get larger initial set for filtering
                initial_k = min(30, self.config.initial_k * 3)  # Increased for better filtering
                
                initial_results = self.collection.query(
                    query_embeddings=[question_embedding.tolist()],
                    n_results=initial_k,
                    include=['documents', 'metadatas', 'distances']
                )
                
                if not initial_results['documents'] or len(initial_results['documents'][0]) == 0:
                    logger.warning("No documents retrieved from initial query")
                    return [], []
                
                documents = initial_results['documents'][0]
                metadatas = initial_results['metadatas'][0]
                distances = initial_results['distances'][0]

                # Compute base scores
                emb_scores = np.exp(-np.array(distances))
                bm25_scores = self.compute_bm25_scores(question, documents)
                
                # Compute key term presence scores
                key_term_scores = np.zeros(len(documents))
                for i, doc in enumerate(documents):
                    doc_lower = doc.lower()
                    # Count how many key terms are in the document
                    term_matches = sum(term.lower() in doc_lower for term in key_terms)
                    if term_matches > 0:
                        # Exponential boost based on number of matching terms
                        key_term_scores[i] = np.exp(term_matches - 1)
                    else:
                        # Reduced penalty for documents without key terms
                        key_term_scores[i] = 0.3  # Changed from 0.1 to be less punitive
                
                # Enhanced hybrid scoring with key term boosting
                hybrid_scores = (
                    0.4 * emb_scores +
                    0.3 * bm25_scores +
                    0.3 * key_term_scores
                )
                
                # Cross-encoder scoring
                chunks_with_context = self.add_chunk_context(documents, metadatas)
                pairs = [(question, doc) for doc in chunks_with_context]
                cross_scores = self.cross_encoder.predict(pairs)
                cross_scores = 1 / (1 + np.exp(-cross_scores))
                
                # Compute chunk similarity
                chunk_similarity = self.compute_chunk_similarity(documents)
                
                # Final scoring with adjusted weighting
                final_scores = np.zeros(len(documents))
                for i in range(len(documents)):
                    if key_term_scores[i] > 0.3:  # Modified threshold
                        final_scores[i] = (
                            0.35 * hybrid_scores[i] +
                            0.35 * cross_scores[i] +
                            0.30 * chunk_similarity[i]
                        )
                    else:
                        final_scores[i] = 0.3 * (  # Increased from 0.1 to 0.3
                            0.35 * hybrid_scores[i] +
                            0.35 * cross_scores[i] +
                            0.30 * chunk_similarity[i]
                        )
                
                # Sort by final scores
                indices = np.argsort(final_scores)[::-1]
                
                # Always take top k documents regardless of threshold
                top_k = min(self.config.num_docs, len(indices))
                selected_indices = indices[:top_k]
                
                # Prepare results
                docs = []
                scores = []
                
                for idx in selected_indices:
                    doc = {
                        'document': documents[idx],
                        'metadata': metadatas[idx],
                        'scores': {
                            'embedding': float(emb_scores[idx]),
                            'bm25': float(bm25_scores[idx]),
                            'key_term': float(key_term_scores[idx]),
                            'cross_encoder': float(cross_scores[idx]),
                            'chunk_similarity': float(chunk_similarity[idx]),
                            'final': float(final_scores[idx])
                        }
                    }
                    docs.append(doc)
                    scores.append(final_scores[idx])
                
                if docs:
                    logger.info(f"Retrieved {len(docs)} documents with scores: {[d['scores']['final'] for d in docs]}")
                else:
                    logger.warning("No documents retrieved after processing")
                
                return docs, scores
                
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise

    
    ########################################################################

    def setup_llm(self):
        """Initialize the LLM with optimizations for local quantized model"""
        logger.info(f"Loading model: {self.config.model_name}")

        # Set PyTorch memory allocator settings
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=self.compute_dtype
        )

        model_config = AutoConfig.from_pretrained(
            self.config.model_name,
            use_flash_attention=self.config.use_flash_attention
        )

        # Load model with optimized memory settings
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map={"": 0},  # Force everything to GPU 0
            max_memory={0: "65GiB"},  # Reduced from 70GiB to leave more headroom
            torch_dtype=self.compute_dtype,
            local_files_only=True
        )

        # Load tokenizer from same local directory
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            local_files_only=True,
            padding_side="left"
        )

        # Handle pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Updated pipeline configuration with smaller batch size
        self.generate_text = transformers.pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task='text-generation',
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_tokens,
            do_sample=True,
            batch_size=8,  # Reduced from config batch size for generation
            device_map={"": 0},
            pad_token_id=self.tokenizer.pad_token_id,
            return_full_text=True
        )

    def setup_chroma(self):
        """Initialize ChromaDB client"""
        self.chroma_client = chromadb.PersistentClient(path=self.config.chroma_db_path)
        collections = self.chroma_client.list_collections()
        logger.info(f"Available collections: {collections}")
        
        try:
            if "medical_qa" in collections:
                self.collection = self.chroma_client.get_collection(name="medical_qa")
                logger.info(f"Successfully connected to existing collection 'medical_qa'")
            else:
                logger.error("Collection 'medical_qa' not found in available collections.")
                raise RuntimeError("Collection 'medical_qa' not found. Run embedder.py first.")
        except Exception as e:
            logger.error(f"Error accessing collection: {e}")
            raise RuntimeError("Error accessing collection. Run embedder.py first.") from e

    def create_medical_prompt(self, question: str, question_type: str, context_docs: List[Dict]) -> str:
        """Create a structured medical QA prompt with enhanced context formatting"""
        formatted_contexts = []
        for i, doc in enumerate(context_docs, 1):
            formatted_doc = f"Document {i}:\n"
            metadata = doc['metadata']
            
            # Include more metadata fields in context
            if metadata.get('title'):
                formatted_doc += f"Title: {metadata['title']}\n"
            if metadata.get('section_title'):
                formatted_doc += f"Section: {metadata['section_title']}\n"
            if metadata.get('url'):
                formatted_doc += f"Source: {metadata['url']}\n"
                
            formatted_doc += f"Content: {doc['document']}\n"
            formatted_doc += f"Relevance Score: {doc.get('similarity_score', 1.0):.2f}\n"
            formatted_contexts.append(formatted_doc)
        
        context_str = "\n".join(formatted_contexts)
        
        prompt = f"""You are a medical professional answering patient questions. Answer the following {question_type} question using ONLY the provided medical reference documents.

    Reference Documents:
    {context_str}

    Question: {question}

    Instructions:
    1. Use ONLY information from the provided reference documents
    2. Be clear, concise, and accurate
    3. If the documents don't contain enough information to answer the question completely, acknowledge this
    4. Structure complex answers with appropriate formatting
    5. Include relevant medical terminology when present in the source documents

    Answer:"""
        
        return prompt

    def process_dataset(self, input_file: str, output_file: str):
        """Process dataset with optimized batching, memory management, and progress tracking"""
        df = pd.read_csv(input_file)
        model_short_name = self.config.model_name.split('/')[-1]
        
        # Load progress tracking
        progress_file = Path(output_file).parent / f"{Path(output_file).stem}_{model_short_name}_progress.json"
        processed_qids = set()
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    processed_qids = set(json.load(f))
            except Exception as e:
                logger.error(f"Error loading progress file: {str(e)}")
        
        # Filter unprocessed questions
        df = df[~df['qid'].astype(str).isin(processed_qids)]
        if len(df) == 0:
            logger.info("All questions have been processed!")
            return
        
        logger.info(f"Processing {len(df)} remaining questions")
        
        # Smaller retrieval batch size
        retrieval_batch_size = self.config.batch_size * 2
        results = []
        
        for batch_start in tqdm(range(0, len(df), retrieval_batch_size)):
            batch_end = min(batch_start + retrieval_batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]
            
            try:
                # Get documents for entire retrieval batch
                all_docs = []
                all_scores = []
                questions = batch_df['question_text'].tolist()
                
                for question in questions:
                    try:
                        docs, scores = self.get_relevant_documents(question)
                        all_docs.append(docs)
                        all_scores.append(scores)
                        torch.cuda.empty_cache()
                    except Exception as e:
                        logger.error(f"Error retrieving documents: {str(e)}")
                        all_docs.append([])
                        all_scores.append([])
                
                # Process generation in smaller batches
                for i in range(0, len(batch_df), 8):
                    sub_batch_df = batch_df.iloc[i:i + 8]
                    sub_batch_docs = all_docs[i:i + 8]
                    
                    try:
                        batch_results = self._process_batch(sub_batch_df)
                        results.extend(batch_results)
                        
                        # Update processed QIDs
                        new_qids = set(str(result['qid']) for result in batch_results)
                        processed_qids.update(new_qids)
                        
                        # Save results and progress more frequently
                        if len(results) >= 50:
                            self._save_results(results, output_file, model_short_name)
                            results = []
                            
                            # Save progress
                            with open(progress_file, 'w') as f:
                                json.dump(list(processed_qids), f)
                            
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        logger.error(f"Error processing batch: {str(e)}")
                        torch.cuda.empty_cache()
                        continue
                    
                    torch.cuda.empty_cache()
                
                # Force garbage collection and cache clearing
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error in batch processing: {str(e)}")
                continue
        
        # Save any remaining results
        if results:
            self._save_results(results, output_file, model_short_name)
            
            # Final progress update
            with open(progress_file, 'w') as f:
                json.dump(list(processed_qids), f)


    def format_response_with_metadata(self, response: str, source_chunks: List[Dict]) -> Dict:
        """Format response with comprehensive source metadata."""
        sources = []
        for chunk in source_chunks:
            source_metadata = chunk.get('metadata', {})
            sources.append({
                'url': source_metadata.get('url', ''),
                'title': source_metadata.get('title', ''),
                'source_file': source_metadata.get('source_file', ''),
                'section_title': source_metadata.get('section_title', ''),
                'chunk_content': chunk.get('content', ''),
                'chunk_hash': source_metadata.get('chunk_hash', '')
            })
        
        return {
            'response': response,
            'sources': sources,
            'metadata': {
                'timestamp': datetime.datetime.now().isoformat(),
                'model_name': self.model_name,
                'total_source_chunks': len(source_chunks)
            }
        }

    def _process_batch(self, batch_df: pd.DataFrame) -> List[Dict]:
        """Process a batch of questions with memory optimizations"""
        batch_results = []
        
        try:
            # Get relevant documents for all questions in batch
            questions = batch_df['question_text'].tolist()
            all_docs = []
            all_scores = []
            
            for question in questions:
                try:
                    docs, scores = self.get_relevant_documents(question)
                    all_docs.append(docs)
                    all_scores.append(scores)
                except Exception as e:
                    logger.error(f"Error retrieving documents for question: {str(e)}")
                    all_docs.append([])
                    all_scores.append([])
            
            # Generate prompts for all questions
            prompts = []
            for idx, (q, docs) in enumerate(zip(questions, all_docs)):
                try:
                    qt = batch_df.iloc[idx].get('qtype', 'general')
                    prompts.append(self.create_medical_prompt(q, qt, docs))
                except Exception as e:
                    logger.error(f"Error creating prompt for question {idx}: {str(e)}")
                    prompts.append("")
            
            # Generate answers in smaller sub-batches
            for sub_batch_start in range(0, len(prompts), 4):  # Process 4 prompts at a time
                sub_batch_end = min(sub_batch_start + 4, len(prompts))
                sub_batch_prompts = prompts[sub_batch_start:sub_batch_end]
                
                try:
                    with torch.amp.autocast('cuda'):
                        responses = self.generate_text(sub_batch_prompts)
                    
                    for idx, response in enumerate(responses):
                        global_idx = sub_batch_start + idx
                        try:
                            # Get question metadata
                            question_metadata = {
                                'qid': str(batch_df.iloc[global_idx].get('qid', '')),
                                'qtype': batch_df.iloc[global_idx].get('qtype', 'general'),
                                'unique_question_id': f"Q{str(batch_df.iloc[global_idx].name).zfill(4)}",
                                'document_id': batch_df.iloc[global_idx].get('document_id', None),
                                'document_source': batch_df.iloc[global_idx].get('document_source', 'NIDDK'),
                                'document_url': batch_df.iloc[global_idx].get('document_url', ''),
                                'document_focus': batch_df.iloc[global_idx].get('document_focus', ''),
                                'valid_doc_url': batch_df.iloc[global_idx].get('valid_doc_url', '')
                            }
                            
                            # Format documents
                            formatted_docs = []
                            if global_idx < len(all_docs) and all_docs[global_idx]:
                                for doc in all_docs[global_idx]:
                                    formatted_doc = {
                                        "content": doc.get('document', ''),
                                        "metadata": doc.get('metadata', {}),
                                        "similarity_score": doc.get('scores', {}).get('final', 0.0)
                                    }
                                    formatted_docs.append(formatted_doc)
                            
                            # Extract answer
                            raw_response = response["generated_text"].strip() if isinstance(response, dict) else response[0]["generated_text"].strip()
                            answer_split = raw_response.split("Answer:", 1)
                            llm_rag_answer = answer_split[1].strip() if len(answer_split) > 1 else raw_response
                            
                            result = {
                                **question_metadata,
                                'question_text': batch_df.iloc[global_idx]['question_text'],
                                'original_answer': batch_df.iloc[global_idx].get('answer_text', ''),
                                'llm_rag_answer': llm_rag_answer,
                                'full_prompt': prompts[global_idx],
                                'model_info': {
                                    'model_name': self.config.model_name,
                                    'temperature': self.config.temperature,
                                    'max_tokens': self.config.max_tokens
                                },
                                'retrieved_documents': formatted_docs,
                                'timestamp': str(pd.Timestamp.now()),
                                'num_source_documents': len(formatted_docs)
                            }
                            
                            batch_results.append(result)
                            
                        except Exception as e:
                            logger.error(f"Error processing result {global_idx}: {str(e)}")
                            continue
                    
                    torch.cuda.empty_cache()  # Clear cache after each sub-batch
                    
                except Exception as e:
                    logger.error(f"Error generating answers for sub-batch: {str(e)}")
                    torch.cuda.empty_cache()
                    continue
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            torch.cuda.empty_cache()
        
        return batch_results

    def _save_results(self, results: List[Dict], output_file: str, model_short_name: str):
        """Save results with enhanced metadata structure and append functionality"""
        output_path = Path(output_file)
        final_output_file = output_path.parent / f"{output_path.stem}_{model_short_name}.json"
        
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (pd.Series, pd.DataFrame)):
                return obj.to_dict()
            try:
                if isinstance(obj, (float, int, str, bool, dict, list)):
                    return obj
                elif obj is None or (isinstance(obj, float) and np.isnan(obj)):
                    return None
                else:
                    return str(obj)
            except (TypeError, ValueError):
                return str(obj)
        
        # Process results to ensure they're serializable
        processed_results = []
        for result in results:
            processed_result = {}
            for key, value in result.items():
                if isinstance(value, dict):
                    processed_result[key] = {k: convert_to_serializable(v) for k, v in value.items()}
                elif isinstance(value, list):
                    processed_result[key] = [convert_to_serializable(item) for item in value]
                else:
                    processed_result[key] = convert_to_serializable(value)
            processed_results.append(processed_result)
        
        # Load existing data if file exists
        existing_data = {'metadata': {}, 'results': []}
        if final_output_file.exists():
            try:
                with open(final_output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except Exception as e:
                logger.error(f"Error loading existing results: {str(e)}")
        
        # Update metadata
        existing_data['metadata'] = {
            'model': {
                'name': self.config.model_name,
                'temperature': float(self.config.temperature),
                'max_tokens': int(self.config.max_tokens),
                'num_docs': int(self.config.num_docs)
            },
            'embedding_model': self.config.embedding_model,
            'last_update_timestamp': str(pd.Timestamp.now()),
            'total_questions_processed': len(existing_data['results']) + len(processed_results)
        }
        
        # Append new results
        existing_data['results'].extend(processed_results)
        
        # Save to file with backup mechanism
        backup_file = str(final_output_file) + '.bak'
        try:
            # First save to backup file
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            
            # Then replace original file
            os.replace(backup_file, final_output_file)
            logger.info(f"Saved {len(processed_results)} new results, total: {len(existing_data['results'])} in {final_output_file}")
        except Exception as e:
            logger.error(f"Error saving results to file: {str(e)}")
            if os.path.exists(backup_file):
                try:
                    # Attempt to restore from backup
                    os.replace(backup_file, final_output_file)
                except:
                    logger.error("Failed to restore from backup")



def parse_args():
    parser = argparse.ArgumentParser(description='Medical RAG QA Pipeline')
    parser.add_argument('--model', choices=list(SUPPORTED_MODELS.keys()), default='llama-70b')
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)  # Increased from 4 to 16
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--torch_dtype', choices=['float16', 'bfloat16'], default='bfloat16')
    parser.add_argument('--use_flash_attention', type=bool, default=True)
    parser.add_argument('--max_memory', type=str, default=None)
    parser.add_argument('--tensor_parallel', type=bool, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = RAGConfig(
        num_docs=5,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        model_name=SUPPORTED_MODELS[args.model],
        batch_size=args.batch_size,
        num_gpus=1,
        torch_dtype=args.torch_dtype,
        use_flash_attention=args.use_flash_attention,
        max_memory=args.max_memory,
        tensor_parallel=args.tensor_parallel,
        initial_k=10
    )
    
    pipeline = MedicalRAGPipeline(config)
    input_file = "deduplicated_NEWURLS_MEDQUAD_NIDDK_QA_DATASET.csv" # OLD: (had 1192 duplicated) NEWURLS_MEDQUAD_NIDDK_QA_DATASET.csv
    test_input_file = "sample10_NEWURLS_MEDQUAD_NIDDK_QA_DATASET.csv"
    output_file = "medquad_rag_results.json"
    pipeline.process_dataset(input_file, output_file)

if __name__ == "__main__":
    main()