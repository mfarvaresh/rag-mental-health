#!/usr/bin/env python

from sentence_transformers import SentenceTransformer, CrossEncoder, util
import numpy as np
from typing import List, Dict, Optional, Tuple

semantic_model = None
reranker_model = None

def get_semantic_model():
    """Load semantic model if not already loaded."""
    global semantic_model
    if semantic_model is None:
        semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    return semantic_model

def get_reranker_model():
    """Load reranker model if not already loaded."""
    global reranker_model
    if reranker_model is None:
        reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return reranker_model

ENHANCED_SYSTEM_PROMPT = """You are a compassionate mental health support assistant with expertise in providing evidence-based guidance.

Your approach:
1. Validate feelings first
2. Use provided context
3. Be specific and practical
4. Encourage professional help
5. Stay hopeful

Important:
- Never diagnose or prescribe medication
- For any mention of self-harm or suicide, immediately provide crisis resources
- Keep responses between 100-200 words
- Use warm, conversational language
- Avoid clinical jargon unless explaining it
"""

def build_better_prompt(query: str, contexts: List[str]) -> str:
    """Build a prompt for the LLM, prioritizing safety and context."""
    crisis_keywords = ["suicide", "kill myself", "end my life", "self-harm", "hurt myself"]
    is_crisis = any(keyword in query.lower() for keyword in crisis_keywords)
    if is_crisis:
        prompt = """URGENT SAFETY RESPONSE NEEDED. Someone has expressed thoughts of self-harm.
Your response MUST:
1. Express immediate concern and caring
2. Provide crisis hotline numbers
3. Encourage immediate professional help
4. Be brief and direct

Crisis Resources:
- Samaritans: 116 123
- Crisis Text Line: Text HOME to 85258
- Emergency: 999

User message: {query}

Your caring, urgent response:"""
        return prompt.format(query=query)
    prompt_parts = [ENHANCED_SYSTEM_PROMPT, "\n"]
    if contexts:
        prompt_parts.append("Relevant information to consider:\n")
        pro_contexts = [c for c in contexts if any(x in c for x in ["Mind.org.uk", "WHO", "Professional Therapist"])]
        community_contexts = [c for c in contexts if "Community Discussion" in c]
        for ctx in pro_contexts[:2]:
            prompt_parts.append(f"- {ctx[:200]}...\n")
        for ctx in community_contexts[:1]:
            prompt_parts.append(f"- {ctx[:200]}...\n")
        prompt_parts.append("\n")
    prompt_parts.append(f"User's concern: {query}\n\n")
    prompt_parts.append("Your compassionate response (acknowledge feelings, provide helpful information, suggest practical steps):")
    return "".join(prompt_parts)

def enhance_response(response: str, query: str) -> str:
    """Post-process LLM response for empathy, suggestions, and length."""
    response = response.replace("System:", "").replace("User:", "").replace("Assistant:", "").strip()
    has_empathy = any(word in response.lower() for word in ["understand", "difficult", "challenging", "sorry", "tough", "hear you"])
    has_suggestion = any(word in response.lower() for word in ["try", "consider", "might help", "suggest", "could"])
    word_count = len(response.split())
    if word_count < 50:
        response += "\n\nIs there anything specific about this situation you'd like to discuss further? I'm here to help."
    if not has_empathy:
        response = "I can understand this is challenging for you. " + response
    if not has_suggestion and "crisis" not in response.lower():
        response += "\n\nOne thing that might help is talking to someone you trust about these feelings, whether that's a friend, family member, or mental health professional."
    if word_count > 250:
        sentences = response.split('. ')
        response = '. '.join(sentences[:10]) + '.'
    return response.strip()

def rerank_contexts(query: str, contexts: List[Dict], top_k: int = 5) -> List[Dict]:
    """Rerank retrieved contexts for better relevance."""
    if not contexts or len(contexts) <= 1:
        return contexts
    reranker = get_reranker_model()
    pairs = [(query, ctx.get('text', '')) for ctx in contexts]
    scores = reranker.predict(pairs)
    ranked_contexts = sorted(zip(contexts, scores), key=lambda x: x[1], reverse=True)
    return [ctx for ctx, score in ranked_contexts[:top_k]]

def evaluate_response_similarity(response: str, reference: str) -> float:
    """Compute semantic similarity between response and reference."""
    model = get_semantic_model()
    embeddings = model.encode([response, reference])
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return float(similarity)

def evaluate_context_relevance(query: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate how relevant contexts are to the query."""
    if not contexts:
        return {"avg_relevance": 0.0, "max_relevance": 0.0}
    model = get_semantic_model()
    query_embedding = model.encode(query)
    context_embeddings = model.encode(contexts)
    similarities = util.cos_sim(query_embedding, context_embeddings)[0]
    return {
        "avg_relevance": float(similarities.mean()),
        "max_relevance": float(similarities.max()),
        "min_relevance": float(similarities.min())
    }

def enhance_rag_response(
    query: str,
    contexts: List[Dict],
    original_response: str,
    use_reranking: bool = True
) -> Tuple[str, List[Dict]]:
    """Enhance RAG response and optionally rerank contexts."""
    if use_reranking and contexts:
        contexts = rerank_contexts(query, contexts, top_k=5)
    context_texts = [ctx.get('text', '') for ctx in contexts]
    prompt = build_better_prompt(query, context_texts)
    enhanced_response = enhance_response(original_response, query)
    return enhanced_response, contexts

def integrate_quick_fixes():
    """Example integration for chains.py (see docstring in codebase for details)."""
    pass

if __name__ == "__main__":
    test_query = "I'm feeling very anxious about my upcoming job interview"
    test_contexts = [
        {"text": "[Professional Therapist Response] Anxiety before interviews is common. Try deep breathing exercises and positive visualization."},
        {"text": "[Community Discussion] I found that practicing mock interviews really helped reduce my anxiety."}
    ]
    test_response = "You should try to relax."
    print("🧪 Testing Quick Fixes\n")
    print(f"Original response: {test_response}")
    enhanced_response, reranked = enhance_rag_response(
        test_query,
        test_contexts,
        test_response
    )
    print(f"\nEnhanced response: {enhanced_response}")
    reference = "It's completely normal to feel anxious before a job interview. This shows you care about the opportunity. Some strategies that might help include practicing deep breathing exercises, preparing answers to common questions, and doing mock interviews with a friend."
    similarity = evaluate_response_similarity(enhanced_response, reference)
    print(f"\nSemantic similarity to reference: {similarity:.3f}")