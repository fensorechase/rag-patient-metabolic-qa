import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import chromadb
import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChunkAnalyzer:
    def __init__(self, chunked_dir: str, processed_dir: str, chroma_dir: str):
        self.chunked_dir = Path(chunked_dir)
        self.processed_dir = Path(processed_dir)
        self.chroma_dir = Path(chroma_dir)
        
    def load_chunks(self) -> List[Dict]:
        """Load chunks from the unique_chunks.json file"""
        chunks_file = self.chunked_dir / "unique_chunks.json"
        if chunks_file.exists():
            with open(chunks_file, 'r') as f:
                return json.load(f)
        return []
        
    def analyze_chunk_lengths(self, chunks: List[Dict]) -> Dict:
        """Analyze chunk length distribution"""
        lengths = [len(chunk['content'].split()) for chunk in chunks]
        return {
            'mean': np.mean(lengths),
            'median': np.median(lengths),
            'min': np.min(lengths),
            'max': np.max(lengths),
            'std': np.std(lengths),
            'lengths': lengths
        }
    
    def plot_chunk_distribution(self, stats: Dict, save_path: str):
        """Create and save distribution plot"""
        plt.figure(figsize=(12, 6))
        
        # Create main distribution plot
        sns.histplot(data=stats['lengths'], bins=30, kde=True)
        
        # Add vertical lines for key statistics
        plt.axvline(stats['mean'], color='r', linestyle='--', label=f'Mean: {stats["mean"]:.0f}')
        plt.axvline(stats['median'], color='g', linestyle='--', label=f'Median: {stats["median"]:.0f}')
        
        # Add labels and title
        plt.xlabel('Chunk Length (words)')
        plt.ylabel('Count')
        plt.title('Distribution of Chunk Lengths')
        plt.legend()
        
        # Add text box with statistics
        stats_text = f"""Statistics:
Mean: {stats['mean']:.0f}
Median: {stats['median']:.0f}
Min: {stats['min']:.0f}
Max: {stats['max']:.0f}
Std: {stats['std']:.0f}"""
        
        plt.text(0.95, 0.95, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save plot
        plt.tight_layout()
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plot saved to {save_path}")
    
    def analyze_chroma_collection(self):
        """Analyze Chroma collection statistics"""
        try:
            client = chromadb.PersistentClient(path=str(self.chroma_dir))
            collections = client.list_collections()
            
            collection_stats = []
            for collection in collections:
                count = collection.count()
                collection_stats.append({
                    'name': collection.name,
                    'count': count
                })
            
            return collection_stats
        except Exception as e:
            logger.error(f"Error analyzing Chroma collection: {e}")
            return []
    
    def get_rag_compatibility_report(self, stats: Dict) -> str:
        """Generate compatibility report for different LLM contexts"""
        llm_contexts = {
            'Llama-3.1-70B': 128000, # No longer 4096
            'Deepseek-R1-Distill-Llama-70B': 128000,
            'Mistral-8x7b': 32000,
            'Mistral-7B': 8192
        }
        
        report = "RAG Compatibility Report:\n" + "="*50 + "\n"
        
        # Assuming average token length is ~1.3 words
        avg_tokens_per_chunk = stats['mean'] * 1.3
        
        for llm, context in llm_contexts.items():
            chunks_per_context = int(context / avg_tokens_per_chunk)
            report += f"\n{llm}:\n"
            report += f"- Context window: {context} tokens\n"
            report += f"- Can fit approximately {chunks_per_context} chunks per context\n"
            report += f"- Chunk compatibility: {'✓ Good' if avg_tokens_per_chunk < context/4 else '⚠ May need adjustment'}\n"
        
        return report

def main():
    analyzer = ChunkAnalyzer(
        chunked_dir="chunked_docs",
        processed_dir="processed_docs",
        chroma_dir="chroma_db"
    )
    
    # Load and analyze chunks
    chunks = analyzer.load_chunks()
    stats = analyzer.analyze_chunk_lengths(chunks)
    
    # Create and save distribution plot
    analyzer.plot_chunk_distribution(stats, "chunk_length_distribution.pdf")
    
    # Analyze Chroma collection
    collection_stats = analyzer.analyze_chroma_collection()
    
    # Generate compatibility report
    rag_report = analyzer.get_rag_compatibility_report(stats)
    
    # Save reports
    with open("chunk_analysis_report.txt", "w") as f:
        f.write("Chunk Analysis Report\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total unique chunks: {len(chunks)}\n\n")
        f.write("Chunk Length Statistics:\n")
        f.write(f"- Mean: {stats['mean']:.2f} words\n")
        f.write(f"- Median: {stats['median']:.2f} words\n")
        f.write(f"- Min: {stats['min']} words\n")
        f.write(f"- Max: {stats['max']} words\n")
        f.write(f"- Std: {stats['std']:.2f} words\n\n")
        
        f.write("Chroma Collection Statistics:\n")
        for stat in collection_stats:
            f.write(f"- Collection '{stat['name']}': {stat['count']} embeddings\n")
        
        f.write("\n" + rag_report)

if __name__ == "__main__":
    main()