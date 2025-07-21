#!/usr/bin/env python
"""Quick script to build the entire RAG index"""
import sys
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
    
    print("\n2️⃣  Chunking documents...")
    chunk_all_data()
    
    print("\n3️⃣  Generating embeddings...")
    embed_all_sources()
    
    print("\n4️⃣  Building vector database...")
    build_vectordb()
    
    print("\n✅ Index build complete!")
    print("\nYou can now:")
    print("- Run queries: python -m ragmh query 'your question'")
    print("- Start chat: python -m ragmh chat")
    print("- Compare: python -m ragmh compare 'your question'")

if __name__ == "__main__":
    main()