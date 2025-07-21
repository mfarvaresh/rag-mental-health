"""Generate embeddings for text chunks using sentence-transformers"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import logging
from tqdm import tqdm
import pickle

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_DIR = DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Model configuration
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # Fast and good for semantic search
BATCH_SIZE = 32

def load_embedding_model(model_name: str = DEFAULT_MODEL):
    """Load sentence transformer model"""
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Show model info
    logger.info(f"Model embedding dimension: {model.get_sentence_embedding_dimension()}")
    logger.info(f"Max sequence length: {model.max_seq_length}")
    
    return model

def load_chunks(source: str = "all") -> List[Dict]:
    """Load chunks from JSON files"""
    if source == "all":
        chunk_file = CHUNKS_DIR / "all_chunks.json"
    else:
        chunk_file = CHUNKS_DIR / f"{source}_chunks.json"
    
    if not chunk_file.exists():
        logger.error(f"Chunk file not found: {chunk_file}")
        return []
    
    with open(chunk_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    logger.info(f"Loaded {len(chunks)} chunks from {chunk_file.name}")
    return chunks

def generate_embeddings(texts: List[str], model) -> np.ndarray:
    """Generate embeddings for a list of texts"""
    # Process in batches for efficiency
    embeddings = []
    
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Generating embeddings"):
        batch = texts[i:i + BATCH_SIZE]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

def create_embedding_index(chunks: List[Dict], model_name: str = DEFAULT_MODEL) -> Dict:
    """Create embeddings for all chunks"""
    model = load_embedding_model(model_name)
    
    # Extract texts
    texts = [chunk['text'] for chunk in chunks]
    
    # Generate embeddings
    logger.info(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = generate_embeddings(texts, model)
    
    # Create index structure
    index = {
        'model_name': model_name,
        'embedding_dim': embeddings.shape[1],
        'num_chunks': len(chunks),
        'chunks': chunks,
        'embeddings': embeddings
    }
    
    return index

def save_embeddings(index: Dict, filename: str):
    """Save embeddings and metadata"""
    output_path = EMBEDDINGS_DIR / filename
    
    # Save as pickle for numpy arrays
    with open(output_path, 'wb') as f:
        pickle.dump(index, f)
    
    logger.info(f"Saved embeddings to {output_path}")
    
    # Also save metadata as JSON for inspection
    metadata = {
        'model_name': index['model_name'],
        'embedding_dim': index['embedding_dim'],
        'num_chunks': index['num_chunks'],
        'sources': {}
    }
    
    # Count chunks by source
    for chunk in index['chunks']:
        source = chunk['source']
        metadata['sources'][source] = metadata['sources'].get(source, 0) + 1
    
    metadata_path = output_path.with_suffix('.meta.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def load_embeddings(filename: str) -> Dict:
    """Load embeddings from file"""
    input_path = EMBEDDINGS_DIR / filename
    
    with open(input_path, 'rb') as f:
        index = pickle.load(f)
    
    logger.info(f"Loaded embeddings from {input_path}")
    return index

def compute_similarity(query_embedding: np.ndarray, 
                      corpus_embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and corpus"""
    # Normalize embeddings
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    corpus_norm = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    
    # Compute cosine similarity
    similarities = np.dot(corpus_norm, query_norm)
    
    return similarities

def search_embeddings(query: str, 
                     index: Dict, 
                     top_k: int = 5) -> List[Tuple[Dict, float]]:
    """Search for similar chunks using embeddings"""
    model = load_embedding_model(index['model_name'])
    
    # Encode query
    query_embedding = model.encode([query])[0]
    
    # Compute similarities
    similarities = compute_similarity(query_embedding, index['embeddings'])
    
    # Get top k results
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        chunk = index['chunks'][idx]
        score = float(similarities[idx])
        results.append((chunk, score))
    
    return results

def embed_all_sources():
    """Generate embeddings for all data sources"""
    logger.info("Starting embedding generation...")
    
    # Load all chunks
    chunks = load_chunks("all")
    
    if not chunks:
        logger.error("No chunks found. Run chunk.py first!")
        return
    
    # Create embeddings
    index = create_embedding_index(chunks)
    
    # Save embeddings
    save_embeddings(index, "all_embeddings.pkl")
    
    # Print summary
    print("\n=== Embedding Summary ===")
    print(f"Model: {index['model_name']}")
    print(f"Embedding dimension: {index['embedding_dim']}")
    print(f"Total chunks embedded: {index['num_chunks']}")
    print(f"\nEmbeddings saved to: {EMBEDDINGS_DIR.absolute()}")
    
    # Test search
    print("\n=== Testing Search ===")
    test_query = "how to deal with anxiety"
    results = search_embeddings(test_query, index, top_k=3)
    
    print(f"\nQuery: '{test_query}'")
    print("\nTop 3 results:")
    for i, (chunk, score) in enumerate(results):
        print(f"\n{i+1}. Score: {score:.3f}")
        print(f"   Source: {chunk['source']}")
        print(f"   Text: {chunk['text'][:150]}...")

def embed_by_source(source: str):
    """Generate embeddings for a specific source"""
    chunks = load_chunks(source)
    
    if not chunks:
        logger.error(f"No chunks found for {source}")
        return
    
    index = create_embedding_index(chunks)
    save_embeddings(index, f"{source}_embeddings.pkl")
    
    logger.info(f"Created embeddings for {source}: {len(chunks)} chunks")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ["counselchat", "reddit", "mind"]:
            embed_by_source(sys.argv[1])
        elif sys.argv[1] == "test":
            # Test search functionality
            index = load_embeddings("all_embeddings.pkl")
            query = input("Enter search query: ")
            results = search_embeddings(query, index)
            
            print(f"\nResults for: '{query}'")
            for i, (chunk, score) in enumerate(results):
                print(f"\n{i+1}. Score: {score:.3f}")
                print(f"   Source: {chunk['source']}")
                print(f"   Text: {chunk['text'][:200]}...")
    else:
        embed_all_sources()