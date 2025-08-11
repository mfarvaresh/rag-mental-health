"""
Vanilla LLM module for comparison with RAG responses
Place this in: src/ragmh/vanilla_llm.py
"""

import logging
from typing import Dict, Optional
from datetime import datetime

from .llm import generate_response, verify_ollama_connection
from .quick_rag_fixes import enhance_response

logger = logging.getLogger(__name__)

VANILLA_SYSTEM_PROMPT = (
    "You are a helpful mental health support assistant. "
    "Provide compassionate, evidence-based guidance. "
    "Be empathetic, practical, and encourage professional help when appropriate. "
    "Keep responses concise (100-200 words) and avoid diagnosing or prescribing medication."
)

def generate_vanilla_response(
    query: str,
    llm: str = "ollama",
    temperature: float = 0.7,
    max_tokens: int = 300,
    enhance: bool = True
) -> str:
    """Generate a response without RAG context."""
    raw_response = generate_response(
        prompt=query,
        context=[],
        system_prompt=VANILLA_SYSTEM_PROMPT,
        llm=llm,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return enhance_response(raw_response, query) if enhance else raw_response

def compare_rag_vs_vanilla(
    query: str,
    rag_result: Dict,
    llm: str = "ollama",
    verbose: bool = True
) -> Dict:
    """Compare RAG vs vanilla responses and return a summary dict."""
    start_time = datetime.now()
    vanilla_response = generate_vanilla_response(query, llm=llm)
    vanilla_duration = (datetime.now() - start_time).total_seconds()
    comparison = {
        "query": query,
        "rag_response": rag_result["response"],
        "vanilla_response": vanilla_response,
        "rag_duration": rag_result["duration_seconds"],
        "vanilla_duration": vanilla_duration,
        "rag_sources": rag_result["sources"],
        "rag_num_contexts": rag_result["num_contexts"],
        "llm": llm,
        "timestamp": datetime.now().isoformat(),
    }
    if verbose:
        print_comparison(comparison)
    return comparison

def print_comparison(comparison: Dict):
    """Pretty-print comparison results."""
    print("\n" + "="*80)
    print("🔍 QUERY:", comparison["query"])
    print("="*80)
    print("\n📚 RAG RESPONSE (with context):")
    print("-"*80)
    print(comparison["rag_response"])
    print(f"\nSources used: {', '.join(comparison['rag_sources'])}")
    print(f"Response time: {comparison['rag_duration']:.2f}s")
    print("\n\n🤖 VANILLA LLM RESPONSE (no context):")
    print("-"*80)
    print(comparison["vanilla_response"])
    print(f"Response time: {comparison['vanilla_duration']:.2f}s")
    print("\n\n📊 COMPARISON METRICS:")
    print(f"- RAG used {comparison['rag_num_contexts']} context chunks")
    print(f"- RAG was {comparison['rag_duration']/comparison['vanilla_duration']:.1f}x slower")
    print(f"- Model used: {comparison['llm']}")
    print("="*80)

def evaluate_response_quality(
    response: str,
    evaluation_criteria: Optional[Dict] = None
) -> Dict:
    """Return a structured template for manual quality evaluation."""
    if evaluation_criteria is None:
        evaluation_criteria = {
            "empathy": "Shows understanding and compassion",
            "specificity": "Provides concrete, actionable advice",
            "safety": "Avoids harmful suggestions, includes disclaimers",
            "professionalism": "Suggests professional help when appropriate",
            "clarity": "Clear, easy to understand language",
        }
    return {
        "response": response,
        "criteria": evaluation_criteria,
        "instructions": "Rate each criterion from 1-5 (1=poor, 5=excellent)",
    }

def compare_responses_cli(query: str, llm: str = "ollama"):
    """CLI helper to run RAG and vanilla comparison."""
    if llm == "ollama" and not verify_ollama_connection():
        print("❌ Ollama not available")
        return None
    from .chains import run_rag_pipeline
    print("\n🔄 Generating RAG response...")
    rag_result = run_rag_pipeline(query=query, save_log=False, verbose=False, llm=llm)
    if not rag_result:
        print("❌ RAG pipeline failed")
        return None
    print("🔄 Generating vanilla response...")
    comparison = compare_rag_vs_vanilla(query=query, rag_result=rag_result, llm=llm, verbose=True)
    return comparison

if __name__ == "__main__":
    test_query = "How can I deal with anxiety before a job interview?"
    compare_responses_cli(test_query)