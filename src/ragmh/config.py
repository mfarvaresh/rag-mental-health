# RAG Configuration
RAG_CONFIG = {
    "retrieval": {
        "initial_top_k": 10,
        "rerank_top_k": 5,
        "min_relevance_score": 0.4,
        "use_reranking": True
    },
    "generation": {
        "temperature": 0.7,
        "max_tokens": 300,
        "fallback_temperature": 0.9  # If first attempt fails quality check
    },
    "quality": {
        "min_response_words": 50,
        "max_response_words": 300,
        "require_empathy": True,
        "require_actionable_advice": True
    },
    "models": {
        "embedding": "all-MiniLM-L6-v2",
        "embedding_reason": "all-mpnet-base-v2",  # Enhanced reasoning model (alternative to Reason-ModernColBERT)
        "reranking": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "llm": "ollama"  # or "gpt-3.5-turbo-0125"
    }
}