"""RAG pipeline combining retrieval and generation with ollama"""
import chromadb
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import json
from datetime import datetime

# Import our modules
from .llm import generate_mental_health_response, verify_ollama_connection
from .vectordb import init_chromadb, search_vectordb

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = DATA_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# RAG Configuration
DEFAULT_TOP_K = 5  # Number of chunks to retrieve
MIN_RELEVANCE_SCORE = 0.3  # Minimum similarity score

def retrieve_context(query: str, 
                    collection,
                    top_k: int = DEFAULT_TOP_K,
                    source_filter: Optional[str] = None) -> List[Dict]:
    """Retrieve relevant context from vector database"""
    
    # Build filter if source specified
    filter_dict = {"source": source_filter} if source_filter else None
    
    # Search vector database
    results = search_vectordb(
        query=query,
        collection=collection,
        n_results=top_k,
        filter_dict=filter_dict
    )
    
    # Filter by relevance score if available
    if results and results[0].get('distance') is not None:
        # ChromaDB returns distance (lower is better)
        # Convert to similarity score (higher is better)
        filtered_results = []
        for result in results:
            similarity = 1 - result['distance']
            if similarity >= MIN_RELEVANCE_SCORE:
                result['similarity'] = similarity
                filtered_results.append(result)
        results = filtered_results
    
    return results

def format_context_for_llm(results: List[Dict]) -> List[str]:
    """Format retrieved chunks for LLM context"""
    contexts = []
    
    for result in results:
        # Include source information
        source = result['metadata']['source']
        
        # Format based on source type
        if source == 'counselchat':
            context = f"[Professional Therapist Response]\n{result['text']}"
        elif source == 'reddit':
            subreddit = result['metadata'].get('subreddit', 'mentalhealth')
            score = result['metadata'].get('score', 0)
            context = f"[Community Discussion from r/{subreddit} (Score: {score})]\n{result['text']}"
        elif source == 'mind.org.uk':
            topic = result['metadata'].get('topic', 'mental health')
            context = f"[Mind.org.uk - {topic}]\n{result['text']}"
        else:
            context = f"[{source}]\n{result['text']}"
        
        contexts.append(context)
        print(context)
    
    return contexts

def rag_query(query: str,
             collection,
             top_k: int = DEFAULT_TOP_K,
             source_filter: Optional[str] = None,
             verbose: bool = False) -> Dict:
    """Complete RAG pipeline: retrieve context and generate response"""
    
    # Record start time
    start_time = datetime.now()
    
    # Step 1: Retrieve relevant context
    if verbose:
        print(f"\n🔍 Searching for relevant information...")
    
    retrieved_results = retrieve_context(
        query=query,
        collection=collection,
        top_k=top_k,
        source_filter=source_filter
    )
    
    if not retrieved_results:
        logger.warning("No relevant context found")
        contexts = []
    else:
        contexts = format_context_for_llm(retrieved_results)
        
        if verbose:
            print(f"✓ Found {len(contexts)} relevant sources")
    
    # Step 2: Generate response with ollama
    if verbose:
        print(f"\n🤖 Generating response with phi3:mini...")
    
    response = generate_mental_health_response(
        user_query=query,
        retrieved_contexts=contexts
    )
    
    # Record end time
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Prepare result
    result = {
        'query': query,
        'response': response,
        'contexts': contexts,
        'sources': [r['metadata']['source'] for r in retrieved_results],
        'num_contexts': len(contexts),
        'duration_seconds': duration,
        'timestamp': start_time.isoformat()
    }
    
    if verbose:
        print(f"✓ Generated response in {duration:.2f} seconds")
        print_formatted_response(result)
    
    return result

def print_formatted_response(result: Dict):
    """Pretty print the RAG response"""
    print("\n" + "="*60)
    print("📝 QUERY:", result['query'])
    print("="*60)
    
    print("\n📚 SOURCES USED:")
    for i, source in enumerate(result['sources']):
        print(f"  {i+1}. {source}")
    
    print("\n💬 RESPONSE:")
    print("-"*60)
    print(result['response'])
    print("-"*60)
    
    print(f"\n⏱️  Response time: {result['duration_seconds']:.2f} seconds")

def save_interaction(result: Dict, log_file: Optional[str] = None):
    """Save interaction to log file"""
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOGS_DIR / f"rag_interaction_{timestamp}.json"
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved interaction to {log_file}")

def run_rag_pipeline(query: str,
                    source_filter: Optional[str] = None,
                    save_log: bool = True,
                    verbose: bool = True):
    """Run the complete RAG pipeline"""
    
    # Verify ollama connection
    if not verify_ollama_connection():
        print("❌ Cannot connect to Ollama. Please ensure it's running.")
        return None
    
    # Initialize vector database
    client = init_chromadb()
    
    try:
        collection = client.get_collection("mental_health_rag")
    except:
        print("❌ Vector database not found. Please run vectordb.py first.")
        return None
    
    # Check if collection has data
    if collection.count() == 0:
        print("❌ Vector database is empty. Please index your data first.")
        return None
    
    # Run RAG query
    result = rag_query(
        query=query,
        collection=collection,
        source_filter=source_filter,
        verbose=verbose
    )
    
    # Save interaction log
    if save_log:
        save_interaction(result)
    
    return result

# Comparison functions for evaluation
def compare_with_vanilla_llm(query: str, verbose: bool = True):
    """Compare RAG response with vanilla LLM response"""
    from .llm import generate_response
    
    print("\n" + "="*60)
    print("🔬 COMPARISON: RAG vs Vanilla LLM")
    print("="*60)
    
    # Get RAG response
    print("\n1️⃣  RAG Response (with context):")
    rag_result = run_rag_pipeline(query, verbose=False)
    
    if rag_result:
        print("-"*60)
        print(rag_result['response'])
        print(f"\nSources: {', '.join(rag_result['sources'])}")
        print(f"Time: {rag_result['duration_seconds']:.2f}s")
    
    # Get vanilla LLM response
    print("\n2️⃣  Vanilla LLM Response (no context):")
    start_time = datetime.now()
    
    vanilla_response = generate_response(
        prompt=query,
        system_prompt="You are a helpful mental health assistant."
    )
    
    duration = (datetime.now() - start_time).total_seconds()
    
    print("-"*60)
    print(vanilla_response)
    print(f"\nTime: {duration:.2f}s")
    
    # Save comparison
    if rag_result:
        comparison = {
            'query': query,
            'rag_response': rag_result['response'],
            'rag_sources': rag_result['sources'],
            'rag_time': rag_result['duration_seconds'],
            'vanilla_response': vanilla_response,
            'vanilla_time': duration,
            'timestamp': datetime.now().isoformat()
        }
        
        comparison_file = LOGS_DIR / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Comparison saved to: {comparison_file}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "compare":
            # Compare RAG vs vanilla
            query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "How can I manage panic attacks?"
            compare_with_vanilla_llm(query)
        else:
            # Run RAG query
            query = " ".join(sys.argv[1:])
            run_rag_pipeline(query)
    else:
        # Interactive mode
        print("\n🧠 Mental Health RAG System")
        print("Type 'quit' to exit, 'compare' for comparison mode\n")
        
        while True:
            query = input("\n❓ Your question: ").strip()
            
            if query.lower() == 'quit':
                break
            elif query.lower() == 'compare':
                test_query = input("Enter query for comparison: ").strip()
                compare_with_vanilla_llm(test_query)
            elif query:
                run_rag_pipeline(query)