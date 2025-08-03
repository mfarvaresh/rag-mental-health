"""RAG pipeline combining retrieval and generation with ollama"""
import chromadb
from pathlib import Path
from typing import List, Dict, Optional
import logging
import json
from datetime import datetime

# ──────────────────────────────────────────────────────────────
#  Internal imports
# ──────────────────────────────────────────────────────────────
from .llm import generate_mental_health_response, verify_ollama_connection
from .vectordb import init_chromadb, search_vectordb
from .quick_rag_fixes import (
    rerank_contexts,          # new
    enhance_rag_response,     # already used
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
#  Project paths
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = DATA_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
#  RAG configuration
# ──────────────────────────────────────────────────────────────
DEFAULT_TOP_K = 5            # chunks returned *after* rerank
PRE_RERANK_MULTIPLIER = 2    # how many candidates to fetch before rerank

# ──────────────────────────────────────────────────────────────
#  Retrieval
# ──────────────────────────────────────────────────────────────
def retrieve_context(
    query: str,
    collection,
    top_k: int = DEFAULT_TOP_K,
    source_filter: Optional[str] = None
) -> List[Dict]:
    """Vector search ➜ cross-encoder rerank ➜ top-k results."""

    # 1. initial wide search
    filter_dict = {"source": source_filter} if source_filter else None
    initial = search_vectordb(
        query=query,
        collection=collection,
        n_results=top_k * PRE_RERANK_MULTIPLIER,
        filter_dict=filter_dict,
    )

    if not initial:
        return []

    # 2. rerank with cross-encoder
    reranked = rerank_contexts(query, initial, top_k=top_k)
    return reranked

# ──────────────────────────────────────────────────────────────
#  Formatting helpers
# ──────────────────────────────────────────────────────────────
def format_context_for_llm(results: List[Dict]) -> List[str]:
    """Convert DB rows → plain-text snippets tagged by source."""
    snippets = []
    for res in results:
        src = res["metadata"]["source"]
        text = res["text"]
        if src == "counselchat":
            snippet = f"[Professional Therapist Response]\n{text}"
        elif src == "reddit":
            subreddit = res["metadata"].get("subreddit", "mentalhealth")
            score = res["metadata"].get("score", 0)
            snippet = f"[Community Discussion from r/{subreddit} (Score: {score})]\n{text}"
        elif src == "mind.org.uk":
            topic = res["metadata"].get("topic", "mental health")
            snippet = f"[Mind.org.uk – {topic}]\n{text}"
        else:
            snippet = f"[{src}]\n{text}"
        snippets.append(snippet)
    return snippets

# ──────────────────────────────────────────────────────────────
#  Main RAG routine
# ──────────────────────────────────────────────────────────────
def rag_query(
    query: str,
    collection,
    top_k: int = DEFAULT_TOP_K,
    source_filter: Optional[str] = None,
    verbose: bool = False,
    llm: str = "ollama",
) -> Dict:
    """Retrieve, generate, polish, and return a complete RAG answer."""
    start_time = datetime.now()

    # Retrieve
    if verbose:
        print("\n🔍 Retrieving context …")
    retrieved = retrieve_context(query, collection, top_k, source_filter)
    contexts  = format_context_for_llm(retrieved)

    # Generate
    if verbose:
        print("🤖 Generating response …")
    raw_answer = generate_mental_health_response(
        user_query=query,
        retrieved_contexts=contexts,
        llm=llm,
    )

    # Post-process
    answer, reranked_ctx = enhance_rag_response(
        query=query,
        contexts=retrieved,
        original_response=raw_answer,
    )

    # Assemble result
    duration = (datetime.now() - start_time).total_seconds()
    result = {
        "query": query,
        "response": answer,
        "contexts": [c["text"] for c in reranked_ctx],
        "sources": [c["metadata"]["source"] for c in reranked_ctx],
        "num_contexts": len(reranked_ctx),
        "duration_seconds": duration,
        "timestamp": start_time.isoformat(),
    }

    if verbose:
        print_formatted_response(result)
    return result

# ──────────────────────────────────────────────────────────────
#  Convenience wrappers
# ──────────────────────────────────────────────────────────────
def print_formatted_response(result: Dict):
    print("\n" + "=" * 60)
    print("📝 QUERY:", result["query"])
    print("=" * 60)

    print("\n📚 SOURCES USED:")
    for i, s in enumerate(result["sources"], 1):
        print(f"  {i}. {s}")

    print("\n💬 RESPONSE:")
    print("-" * 60)
    print(result["response"])
    print("-" * 60)
    print(f"\n⏱️  Response time: {result['duration_seconds']:.2f} s")

def save_interaction(result: Dict, log_file: Optional[str] = None):
    if log_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOGS_DIR / f"rag_interaction_{ts}.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved interaction to {log_file}")

def run_rag_pipeline(
    query: str,
    source_filter: Optional[str] = None,
    save_log: bool = True,
    verbose: bool = True,
    llm: str = "ollama",
):
    if not verify_ollama_connection():
        print("❌ Cannot connect to Ollama; please start it first.")
        return None

    client = init_chromadb()
    try:
        collection = client.get_collection("mental_health_rag")
    except KeyError:
        print("❌ Collection not found. Run vectordb.py first.")
        return None
    if collection.count() == 0:
        print("❌ Vector DB is empty. Index your data first.")
        return None

    result = rag_query(query, collection, source_filter=source_filter, verbose=verbose, llm=llm)
    if save_log:
        save_interaction(result)
    return result
