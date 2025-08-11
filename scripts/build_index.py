#!/usr/bin/env python
import sys
import nltk
nltk.download('punkt')
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragmh import (
    ingest_all_data,
    chunk_all_data,
    embed_all_sources,
    build_vectordb,
    verify_ollama_connection,
)

def main():
    print("🚀 Building RAG Mental Health Index")
    print("="*50)
    print("\n✓ Checking Ollama connection...")
    if not verify_ollama_connection():
        print("❌ Ollama not running! Start with: ollama serve")
        return
    print("\n1️⃣  Ingesting data...")
    ingest_all_data()
    print("\n2️⃣  Chunking documents...")
    chunk_all_data()
    print("\n3️⃣  Generating embeddings...")
    embed_all_sources()
    print("\n4️⃣  Building vector database...")
    build_vectordb()
    print("\n✅ Index build complete!")

if __name__ == "__main__":
    main()