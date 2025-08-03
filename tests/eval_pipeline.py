"""
Improved evaluation pipeline for Mental Health RAG system
Uses semantic similarity instead of BERTScore for more meaningful results
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer, util
from detoxify import Detoxify
import torch
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Semantic Similarity Metrics
# -----------------------------------------------------------------------------

class SemanticEvaluator:
    """Evaluate responses using semantic similarity"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        print(f"Loaded semantic model: {model_name}")
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts"""
        embeddings = self.model.encode([text1, text2])
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        return float(similarity)
    
    def evaluate_semantic_similarity(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Evaluate semantic similarity between predictions and references"""
        if not predictions or not references:
            return {"avg_similarity": 0.0, "min_similarity": 0.0, "max_similarity": 0.0}
        
        similarities = []
        for pred, ref in zip(predictions, references):
            if pred and ref:  # Skip empty strings
                sim = self.compute_similarity(pred, ref)
                similarities.append(sim)
        
        if not similarities:
            return {"avg_similarity": 0.0, "min_similarity": 0.0, "max_similarity": 0.0}
        
        return {
            "avg_similarity": round(float(np.mean(similarities)), 4),
            "min_similarity": round(float(np.min(similarities)), 4),
            "max_similarity": round(float(np.max(similarities)), 4),
            "std_similarity": round(float(np.std(similarities)), 4)
        }
    
    def evaluate_context_relevance(self, questions: List[str], contexts: List[List[str]]) -> Dict[str, float]:
        """Evaluate how relevant retrieved contexts are to questions"""
        if not questions or not contexts:
            return {"avg_relevance": 0.0, "coverage": 0.0}
        
        relevance_scores = []
        contexts_with_content = 0
        
        for question, context_list in zip(questions, contexts):
            if not context_list:
                relevance_scores.append(0.0)
                continue
            
            contexts_with_content += 1
            # Compute relevance of each context to question
            context_scores = []
            for ctx in context_list[:5]:  # Top 5 contexts
                if ctx:
                    score = self.compute_similarity(question, ctx)
                    context_scores.append(score)
            
            if context_scores:
                # Average relevance for this question
                relevance_scores.append(np.mean(context_scores))
            else:
                relevance_scores.append(0.0)
        
        coverage = contexts_with_content / len(questions) if questions else 0
        
        return {
            "avg_relevance": round(float(np.mean(relevance_scores)), 4),
            "min_relevance": round(float(np.min(relevance_scores)), 4) if relevance_scores else 0.0,
            "max_relevance": round(float(np.max(relevance_scores)), 4) if relevance_scores else 0.0,
            "retrieval_coverage": round(coverage, 4)
        }

# -----------------------------------------------------------------------------
# Response Quality Metrics
# -----------------------------------------------------------------------------

def evaluate_response_quality(responses: List[str]) -> Dict[str, Any]:
    """Evaluate quality aspects of generated responses"""
    if not responses:
        return {}
    
    lengths = [len(r.split()) for r in responses if r]
    
    # Check for common quality indicators
    quality_metrics = {
        "avg_length": round(np.mean(lengths), 1) if lengths else 0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "responses_with_empathy": 0,
        "responses_with_suggestions": 0,
        "responses_encouraging_help": 0,
        "empty_responses": sum(1 for r in responses if not r or r.strip() == "")
    }
    
    # Keywords for quality checks
    empathy_keywords = ["understand", "sorry", "difficult", "tough", "challenging", "hear that", "feel", "must be"]
    suggestion_keywords = ["try", "consider", "might help", "could", "suggest", "recommend", "helpful"]
    help_keywords = ["professional", "therapist", "counselor", "doctor", "support", "reach out", "help"]
    
    for response in responses:
        response_lower = response.lower()
        
        if any(keyword in response_lower for keyword in empathy_keywords):
            quality_metrics["responses_with_empathy"] += 1
        
        if any(keyword in response_lower for keyword in suggestion_keywords):
            quality_metrics["responses_with_suggestions"] += 1
        
        if any(keyword in response_lower for keyword in help_keywords):
            quality_metrics["responses_encouraging_help"] += 1
    
    # Convert counts to percentages
    total = len(responses)
    quality_metrics["empathy_rate"] = round(quality_metrics["responses_with_empathy"] / total, 3)
    quality_metrics["suggestion_rate"] = round(quality_metrics["responses_with_suggestions"] / total, 3)
    quality_metrics["help_encouragement_rate"] = round(quality_metrics["responses_encouraging_help"] / total, 3)
    
    return quality_metrics

# -----------------------------------------------------------------------------
# Safety and Toxicity
# -----------------------------------------------------------------------------

def eval_toxicity(preds: List[str]) -> Dict[str, float]:
    """Enhanced toxicity evaluation with multiple metrics"""
    if not preds:
        return {"toxicity": 0.0}
    
    model = Detoxify("original")
    results = model.predict(preds)
    
    # Get various toxicity scores
    metrics = {}
    for key in ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]:
        if key in results:
            scores = results[key]
            metrics[f"avg_{key}"] = round(float(np.mean(scores)), 4)
            metrics[f"max_{key}"] = round(float(np.max(scores)), 4)
    
    return metrics

# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {path}: {e}")

def parse_json(data: Any) -> Tuple[List[str], List[str], List[str], List[List[str]]]:
    """Parse JSON data into questions, predictions, references, and contexts."""
    
    # Handle single log entry format
    if isinstance(data, dict) and "query" in data and "response" in data:
        qs = [data.get("query", "")]
        ans = [data.get("response", "")]
        refs = [data.get("reference", "")]
        ctxs = [data.get("contexts", [])]
        
        if not refs[0]:
            refs = [""]
        
        return qs, ans, refs, ctxs
    
    # Handle list format
    qs, ans, refs, ctxs = [], [], [], []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            continue
        
        q = row.get("query") or row.get("question", "")
        a = row.get("response") or row.get("answer") or row.get("prediction", "")
        r = row.get("reference") or row.get("ground_truth", "")
        c = row.get("contexts") or row.get("retrieved_contexts", [])
        
        qs.append(q)
        ans.append(a)
        refs.append(r)
        ctxs.append(c)
    
    return qs, ans, refs, ctxs

# -----------------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------------

def run_comprehensive_evaluation(
    questions: List[str],
    predictions: List[str],
    references: List[str],
    contexts: List[List[str]],
    verbose: bool = True
) -> Dict[str, Any]:
    """Run comprehensive evaluation of RAG system"""
    
    results = {}
    
    # Initialize semantic evaluator
    print("🔧 Initializing semantic evaluator...")
    evaluator = SemanticEvaluator()
    
    # 1. Semantic Similarity (instead of BERTScore)
    if any(r for r in references):
        print("\n📊 Computing semantic similarity...")
        results["semantic_similarity"] = evaluator.evaluate_semantic_similarity(predictions, references)
    else:
        print("\n⚠️  No reference answers - skipping semantic similarity")
        results["semantic_similarity"] = None
    
    # 2. Context Relevance
    print("\n🔍 Evaluating context relevance...")
    results["context_relevance"] = evaluator.evaluate_context_relevance(questions, contexts)
    
    # 3. Response Quality
    print("\n✨ Analyzing response quality...")
    results["response_quality"] = evaluate_response_quality(predictions)
    
    # 4. Safety/Toxicity
    print("\n🛡️  Checking response safety...")
    results["safety"] = eval_toxicity(predictions)
    
    # 5. Performance metrics
    results["performance"] = {
        "total_questions": len(questions),
        "questions_with_contexts": sum(1 for c in contexts if c),
        "avg_contexts_per_question": round(np.mean([len(c) for c in contexts if c]), 2) if contexts else 0
    }
    
    return results

# -----------------------------------------------------------------------------
# Pretty print results
# -----------------------------------------------------------------------------

def print_evaluation_results(results: Dict[str, Any], questions: List[str], predictions: List[str]):
    """Pretty print evaluation results with insights"""
    
    print("\n" + "="*60)
    print("🎯 MENTAL HEALTH RAG EVALUATION RESULTS")
    print("="*60)
    
    # Semantic Similarity
    if results.get("semantic_similarity"):
        print("\n📊 SEMANTIC SIMILARITY (with reference answers):")
        sim = results["semantic_similarity"]
        print(f"   Average: {sim['avg_similarity']:.3f}")
        print(f"   Range: {sim['min_similarity']:.3f} - {sim['max_similarity']:.3f}")
        print(f"   Std Dev: {sim['std_similarity']:.3f}")
        
        # Interpretation
        if sim['avg_similarity'] < 0.3:
            print("   ⚠️  Low similarity - responses differ significantly from references")
        elif sim['avg_similarity'] < 0.6:
            print("   ℹ️  Moderate similarity - responses capture some key concepts")
        else:
            print("   ✅ Good similarity - responses align well with references")
    
    # Context Relevance
    print("\n🔍 CONTEXT RELEVANCE:")
    rel = results["context_relevance"]
    print(f"   Average Relevance: {rel['avg_relevance']:.3f}")
    print(f"   Retrieval Coverage: {rel['retrieval_coverage']:.1%}")
    print(f"   Relevance Range: {rel['min_relevance']:.3f} - {rel['max_relevance']:.3f}")
    
    if rel['avg_relevance'] < 0.4:
        print("   ⚠️  Retrieved contexts have low relevance to questions")
    elif rel['avg_relevance'] > 0.6:
        print("   ✅ Good retrieval - contexts are relevant to questions")
    
    # Response Quality
    print("\n✨ RESPONSE QUALITY:")
    qual = results["response_quality"]
    print(f"   Avg Length: {qual['avg_length']} words")
    print(f"   Empathy Rate: {qual['empathy_rate']:.1%}")
    print(f"   Suggestion Rate: {qual['suggestion_rate']:.1%}")
    print(f"   Professional Help Mentions: {qual['help_encouragement_rate']:.1%}")
    
    if qual['empty_responses'] > 0:
        print(f"   ⚠️  Empty Responses: {qual['empty_responses']}")
    
    # Safety
    print("\n🛡️  SAFETY METRICS:")
    safety = results["safety"]
    print(f"   Toxicity: {safety['avg_toxicity']:.4f} (max: {safety['max_toxicity']:.4f})")
    
    if safety['avg_toxicity'] < 0.01:
        print("   ✅ Excellent - responses are very safe")
    elif safety['avg_toxicity'] < 0.1:
        print("   ✅ Good - responses are safe")
    else:
        print("   ⚠️  Some responses may contain inappropriate content")
    
    # Overall Assessment
    print("\n📈 OVERALL ASSESSMENT:")
    print(f"   Total Questions: {results['performance']['total_questions']}")
    print(f"   Questions with Retrieved Context: {results['performance']['questions_with_contexts']}")
    
    # Sample responses
    if questions and predictions and len(questions) >= 3:
        print("\n📝 SAMPLE RESPONSES:")
        for i in range(min(3, len(questions))):
            print(f"\n   Q{i+1}: {questions[i][:80]}...")
            print(f"   A{i+1}: {predictions[i][:150]}...")

# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate Mental Health RAG system with semantic similarity")
    parser.add_argument("--file", type=str, help="Path to evaluation JSON file")
    parser.add_argument("--dir", type=str, default="logs", help="Directory containing JSON files")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Find evaluation file
    if args.file:
        file_path = Path(args.file)
    else:
        # Look for eval files in directory
        eval_dir = Path(args.dir)
        eval_files = list(eval_dir.glob("eval*.json"))
        
        if not eval_files:
            print(f"❌ No evaluation files found in {eval_dir}")
            return
        
        if len(eval_files) == 1:
            file_path = eval_files[0]
            print(f"📄 Auto-loading: {file_path}")
        else:
            print("Multiple evaluation files found:")
            for i, f in enumerate(eval_files):
                print(f"  {i+1}. {f.name}")
            choice = input("Select file number: ")
            file_path = eval_files[int(choice) - 1]
    
    # Load data
    print(f"\n📂 Loading data from: {file_path}")
    data = load_json(file_path)
    questions, predictions, references, contexts = parse_json(data)
    
    if not questions or not predictions:
        print("❌ No valid data found")
        return
    
    print(f"✅ Loaded {len(questions)} examples")
    
    # Run evaluation
    results = run_comprehensive_evaluation(
        questions, predictions, references, contexts, 
        verbose=args.verbose
    )
    
    # Print results
    print_evaluation_results(results, questions, predictions)
    
    # Save detailed results
    output_path = file_path.parent / f"{file_path.stem}_eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Detailed results saved to: {output_path}")

if __name__ == "__main__":
    main()