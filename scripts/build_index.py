#!/usr/bin/env python
"""Quick script to build the entire RAG index"""
import sys
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragmh import (
    ingest_all_data,
    chunk_all_data,
    embed_all_sources,
    build_vectordb,
    verify_ollama_connection
)

def main():
    print("🚀 Building RAG Mental Health Index")
    print("="*50)
    
    # Check ollama
    print("\n✓ Checking Ollama connection...")
    if not verify_ollama_connection():
        print("❌ Ollama not running! Please start it with: ollama serve")
        return
    
    # Run pipeline
    print("\n1️⃣  Ingesting data...")
    ingest_all_data()
    
    print("\n2️⃣  Chunking documents (including PubMed and WHO)...")
    chunk_all_data()
    
    print("\n3️⃣  Generating embeddings (including PubMed and WHO)...")
    embed_all_sources()
    
    print("\n4️⃣  Building vector database...")
    build_vectordb()
    
    print("\n✅ Index build complete!")
    print("\nYou can now:")
    print("- Run queries: python -m ragmh query 'your question'")
    print("- Start chat: python -m ragmh chat")
    print("- Compare: python -m ragmh compare 'your question'")
    print("- Run queries: python -m ragmh pubmed 'your query' --max 30")
    print("- Run queries: python -m ragmh who mental-health")


if __name__ == "__main__":
    main()