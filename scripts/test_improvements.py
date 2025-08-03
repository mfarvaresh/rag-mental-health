#!/usr/bin/env python
"""Test improved RAG system"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ragmh import run_rag_pipeline

test_queries = [
    "I'm feeling anxious about my job interview tomorrow",
    "How can I deal with panic attacks?",
    "I've been feeling really depressed lately",
    "What are some coping strategies for stress?"
]

print("🧪 Testing Improved RAG System\n")

for query in test_queries:
    print(f"❓ Query: {query}")
    result = run_rag_pipeline(query, verbose=False)
    print(f"💬 Response: {result['response'][:200]}...")
    print(f"⏱️  Time: {result['duration_seconds']:.2f}s")
    print("-" * 60 + "\n")