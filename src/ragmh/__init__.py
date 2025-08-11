"""RAG Mental Health package"""

from .ingest import ingest_all_data
from .chunk import chunk_all_data
from .embed import embed_all_sources
from .vectordb import build_vectordb, init_chromadb, get_collection_stats
from .chains import run_rag_pipeline
from .llm import verify_ollama_connection

__version__ = "0.1.0"
__all__ = [
    "ingest_all_data",
    "chunk_all_data",
    "embed_all_sources",
    "build_vectordb",
    "init_chromadb",
    "get_collection_stats",
    "run_rag_pipeline",
    "verify_ollama_connection",
]