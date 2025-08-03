"""Vector database using ChromaDB for efficient similarity search"""
import json
import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from sentence_transformers import SentenceTransformer
import pickle

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_DIR = DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
VECTORDB_DIR = DATA_DIR / "vectordb"
VECTORDB_DIR.mkdir(parents=True, exist_ok=True)

# ChromaDB settings
COLLECTION_NAME = "mental_health_rag"
DEFAULT_MODEL = "all-MiniLM-L6-v2"

def init_chromadb(persist_dir: Optional[Path] = None):
    """Initialize ChromaDB client"""
    if persist_dir is None:
        persist_dir = VECTORDB_DIR
    
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False)
    )
    
    logger.info(f"Initialized ChromaDB at {persist_dir}")
    return client

def create_collection(client, 
                     collection_name: str = COLLECTION_NAME,
                     embedding_dim: int = 384):
    """Create or get a collection"""
    try:
        # Try to get existing collection
        collection = client.get_collection(name=collection_name)
        logger.info(f"Using existing collection: {collection_name}")
    except:
        # Create new collection
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        logger.info(f"Created new collection: {collection_name}")
    
    return collection

def load_and_index_chunks(collection, 
                         source: str = "all",
                         batch_size: int = 100):
    """Load chunks and add to ChromaDB"""
    # Load chunks
    if source == "all":
        chunk_file = CHUNKS_DIR / "all_chunks.json"
    else:
        chunk_file = CHUNKS_DIR / f"{source}_chunks.json"
    
    with open(chunk_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Load pre-computed embeddings if available
    embedding_file = EMBEDDINGS_DIR / f"{source}_embeddings.pkl"
    embeddings = None
    
    if embedding_file.exists():
        logger.info(f"Loading pre-computed embeddings from {embedding_file}")
        with open(embedding_file, 'rb') as f:
            index = pickle.load(f)
            embeddings = index['embeddings']
    else:
        logger.info("No pre-computed embeddings found. Will generate on-the-fly.")
        model = SentenceTransformer(DEFAULT_MODEL)
    
    # Process in batches
    total_added = 0
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        
        # Prepare data for ChromaDB
        ids = []
        texts = []
        metadatas = []
        batch_embeddings = []
        
        for j, chunk in enumerate(batch_chunks):
            # Create unique ID
            chunk_id = f"{chunk['source']}_{i+j}"
            ids.append(chunk_id)
            texts.append(chunk['text'])
            
            # Flatten metadata for ChromaDB
            metadata = {
                'source': chunk['source'],
                'chunk_index': chunk['metadata'].get('chunk_index', 0)
            }
            
            # Add source-specific metadata
            if chunk['source'] == 'counselchat':
                metadata['topic'] = chunk['metadata'].get('topic', 'general')
                metadata['upvotes'] = chunk['metadata'].get('upvotes', 0)
            elif chunk['source'] == 'reddit':
                metadata['subreddit'] = chunk['metadata'].get('subreddit', '')
                metadata['score'] = chunk['metadata'].get('score', 0)
            elif chunk['source'] == 'mind.org.uk':
                metadata['topic'] = chunk['metadata'].get('topic', '')
                metadata['url'] = chunk['metadata'].get('url', '')
            elif chunk['source'] == 'pubmed':
                metadata['title'] = chunk['metadata'].get('title', '')
                metadata['pmid'] = chunk['metadata'].get('pmid', '')
                metadata['query'] = chunk['metadata'].get('query', '')
            elif chunk['source'] == 'who':
                metadata['topic'] = chunk['metadata'].get('topic', '')
                metadata['url'] = chunk['metadata'].get('url', '')
            
            metadatas.append(metadata)
            
            # Get embedding
            if embeddings is not None:
                batch_embeddings.append(embeddings[i+j].tolist())
            else:
                # Generate embedding on-the-fly
                embedding = model.encode(chunk['text'])
                batch_embeddings.append(embedding.tolist())
        
        # Add to collection
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=batch_embeddings
        )
        
        total_added += len(batch_chunks)
        logger.info(f"Added {total_added}/{len(chunks)} chunks to vector DB")
    
    return total_added

def search_vectordb(query: str,
                   collection,
                   n_results: int = 5,
                   filter_dict: Optional[Dict] = None,
                   rerank: bool = True,
                   top_k: int = 5) -> List[Dict]:
    """Search the vector database with optional metadata filtering and reranking."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=filter_dict
    )
    formatted_results = []
    for i in range(len(results['ids'][0])):
        result = {
            'id': results['ids'][0][i],
            'text': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i] if 'distances' in results else None
        }
        formatted_results.append(result)
    # Rerank if enabled
    if rerank and formatted_results:
        from .vectordb import rerank_results
        formatted_results = rerank_results(formatted_results, query, top_k=top_k)
    return formatted_results

def hybrid_search_vectordb(query: str,
                         collection,
                         n_results: int = 5,
                         filter_dict: Optional[Dict] = None,
                         rerank: bool = True,
                         top_k: int = 5) -> List[Dict]:
    """Hybrid search: combine vector search with keyword filtering and reranking."""
    # Step 1: Vector search
    vector_results = collection.query(
        query_texts=[query],
        n_results=n_results * 2,  # get more for hybrid
        where=filter_dict
    )
    formatted_results = []
    for i in range(len(vector_results['ids'][0])):
        result = {
            'id': vector_results['ids'][0][i],
            'text': vector_results['documents'][0][i],
            'metadata': vector_results['metadatas'][0][i],
            'distance': vector_results['distances'][0][i] if 'distances' in vector_results else None
        }
        formatted_results.append(result)
    # Step 2: Keyword filtering (simple)
    keyword_results = [r for r in formatted_results if query.lower() in r['text'].lower()]
    # Step 3: Merge and deduplicate
    all_results = keyword_results + [r for r in formatted_results if r not in keyword_results]
    # Step 4: Rerank
    if rerank and all_results:
        from .vectordb import rerank_results
        all_results = rerank_results(all_results, query, top_k=top_k)
    return all_results[:top_k]

def rerank_results(results: List[Dict], query: str, top_k: int = 5) -> List[Dict]:
    """Rerank retrieved results using a cross-encoder for better relevance."""
    from sentence_transformers import CrossEncoder
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [(query, r['text']) for r in results]
    scores = model.predict(pairs)
    reranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return [r for r, s in reranked[:top_k]]

def delete_collection(client, collection_name: str = COLLECTION_NAME):
    """Delete a collection"""
    try:
        client.delete_collection(name=collection_name)
        logger.info(f"Deleted collection: {collection_name}")
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")

def get_collection_stats(collection) -> Dict:
    """Get statistics about the collection"""
    count = collection.count()
    
    # Get sample of metadata to analyze
    sample = collection.get(limit=min(1000, count))
    
    stats = {
        'total_documents': count,
        'sources': {}
    }
    
    # Count by source
    for metadata in sample['metadatas']:
        source = metadata.get('source', 'unknown')
        stats['sources'][source] = stats['sources'].get(source, 0) + 1
    
    return stats

def build_vectordb(source: str = "all", rebuild: bool = False):
    """Build the vector database"""
    logger.info("Building vector database...")
    
    # Initialize ChromaDB
    client = init_chromadb()
    
    if rebuild:
        delete_collection(client, COLLECTION_NAME)
    
    # Create collection
    collection = create_collection(client)
    
    # Check if already populated
    if collection.count() > 0 and not rebuild:
        logger.info(f"Collection already has {collection.count()} documents")
        return collection
    
    # Load and index chunks
    total_added = load_and_index_chunks(collection, source)
    
    # Print stats
    stats = get_collection_stats(collection)
    print("\n=== Vector Database Stats ===")
    print(f"Total documents: {stats['total_documents']}")
    print("Documents by source:")
    for source, count in stats['sources'].items():
        print(f"  - {source}: {count}")
    
    return collection

def test_search(query: Optional[str] = None):
    """Test the vector database search"""
    client = init_chromadb()
    collection = client.get_collection(COLLECTION_NAME)
    
    if query is None:
        query = "how to cope with anxiety attacks"
    
    print(f"\n=== Searching for: '{query}' ===\n")
    
    # Basic search
    results = search_vectordb(query, collection, n_results=3)
    
    for i, result in enumerate(results):
        print(f"{i+1}. Source: {result['metadata']['source']}")
        print(f"   Text: {result['text'][:200]}...")
        print(f"   Metadata: {result['metadata']}")
        print()
    
    # Rerank results
    reranked_results = rerank_results(results, query, top_k=3)
    print(f"=== Reranked Results ===\n")
    for i, result in enumerate(reranked_results):
        print(f"{i+1}. Source: {result['metadata']['source']}")
        print(f"   Text: {result['text'][:200]}...")
        print(f"   Metadata: {result['metadata']}")
        print()
    
    # Filtered search example
    print("\n=== Filtered search (Reddit only) ===\n")
    reddit_results = search_vectordb(
        query, 
        collection, 
        n_results=2,
        filter_dict={"source": "reddit"}
    )
    
    for i, result in enumerate(reddit_results):
        print(f"{i+1}. Subreddit: r/{result['metadata'].get('subreddit', 'unknown')}")
        print(f"   Score: {result['metadata'].get('score', 0)}")
        print(f"   Text: {result['text'][:200]}...")
        print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "build":
            # Build vector database
            source = sys.argv[2] if len(sys.argv) > 2 else "all"
            build_vectordb(source)
        elif sys.argv[1] == "rebuild":
            # Rebuild from scratch
            build_vectordb(rebuild=True)
        elif sys.argv[1] == "test":
            # Test search
            query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None
            test_search(query)
        elif sys.argv[1] == "stats":
            # Show statistics
            client = init_chromadb()
            collection = client.get_collection(COLLECTION_NAME)
            stats = get_collection_stats(collection)
            print(json.dumps(stats, indent=2))
    else:
        # Default: build if needed
        build_vectordb()