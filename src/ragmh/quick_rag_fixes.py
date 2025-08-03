#!/usr/bin/env python
"""
Quick fixes for Mental Health RAG system
Drop this into src/ragmh/ and import the functions you need
"""

from sentence_transformers import SentenceTransformer, CrossEncoder, util
import numpy as np
from typing import List, Dict, Optional, Tuple

# Initialize models globally for efficiency
semantic_model = None
reranker_model = None

def get_semantic_model():
    """Lazy load semantic model"""
    global semantic_model
    if semantic_model is None:
        semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    return semantic_model

def get_reranker_model():
    """Lazy load reranker model"""
    global reranker_model
    if reranker_model is None:
        reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return reranker_model

# -----------------------------------------------------------------------------
# Enhanced System Prompt
# -----------------------------------------------------------------------------

ENHANCED_SYSTEM_PROMPT = """You are a compassionate mental health support assistant with expertise in providing evidence-based guidance.

Your approach:
1. **Validate feelings first** - Always acknowledge what the person is experiencing
2. **Use provided context** - Reference the information given, but make it conversational
3. **Be specific and practical** - Offer concrete strategies they can try
4. **Encourage professional help** - When appropriate, gently suggest professional support
5. **Stay hopeful** - End with encouragement without minimizing their experience

Important:
- Never diagnose or prescribe medication
- For any mention of self-harm or suicide, immediately provide crisis resources
- Keep responses between 100-200 words
- Use warm, conversational language
- Avoid clinical jargon unless explaining it
"""

# -----------------------------------------------------------------------------
# Quick Fix 1: Better Prompt Building
# -----------------------------------------------------------------------------

def build_better_prompt(query: str, contexts: List[str]) -> str:
    """Build an improved prompt that produces better responses"""
    
    # Check for crisis keywords
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
- National Suicide Prevention Lifeline: 988 or 1-800-273-8255
- Crisis Text Line: Text HOME to 741741
- Emergency: 911

User message: {query}

Your caring, urgent response:"""
        return prompt.format(query=query)
    
    # Regular prompt with better structure
    prompt_parts = [ENHANCED_SYSTEM_PROMPT, "\n"]
    
    if contexts:
        prompt_parts.append("Relevant information to consider:\n")
        # Prioritize professional sources
        pro_contexts = [c for c in contexts if any(x in c for x in ["Mind.org.uk", "WHO", "Professional Therapist"])]
        community_contexts = [c for c in contexts if "Community Discussion" in c]
        
        for ctx in pro_contexts[:2]:  # Top 2 professional
            prompt_parts.append(f"- {ctx[:200]}...\n")
        
        for ctx in community_contexts[:1]:  # 1 community perspective
            prompt_parts.append(f"- {ctx[:200]}...\n")
        
        prompt_parts.append("\n")
    
    prompt_parts.append(f"User's concern: {query}\n\n")
    prompt_parts.append("Your compassionate response (acknowledge feelings, provide helpful information, suggest practical steps):")
    
    return "".join(prompt_parts)

# -----------------------------------------------------------------------------
# Quick Fix 2: Response Quality Enhancement
# -----------------------------------------------------------------------------

def enhance_response(response: str, query: str) -> str:
    """Post-process response to ensure quality"""
    
    # Remove any system prompt leakage
    response = response.replace("System:", "").replace("User:", "").replace("Assistant:", "").strip()
    
    # Check response quality
    has_empathy = any(word in response.lower() for word in 
                     ["understand", "difficult", "challenging", "sorry", "tough", "hear you"])
    has_suggestion = any(word in response.lower() for word in 
                        ["try", "consider", "might help", "suggest", "could"])
    
    word_count = len(response.split())
    
    # Fix common issues
    if word_count < 50:
        response += "\n\nIs there anything specific about this situation you'd like to discuss further? I'm here to help."
    
    if not has_empathy:
        response = "I can understand this is challenging for you. " + response
    
    if not has_suggestion and "crisis" not in response.lower():
        response += "\n\nOne thing that might help is talking to someone you trust about these feelings, whether that's a friend, family member, or mental health professional."
    
    # Ensure it's not too long
    if word_count > 250:
        sentences = response.split('. ')
        response = '. '.join(sentences[:10]) + '.'
    
    return response.strip()

# -----------------------------------------------------------------------------
# Quick Fix 3: Better Retrieval with Reranking
# -----------------------------------------------------------------------------

def rerank_contexts(query: str, contexts: List[Dict], top_k: int = 5) -> List[Dict]:
    """Rerank retrieved contexts for better relevance"""
    
    if not contexts or len(contexts) <= 1:
        return contexts
    
    reranker = get_reranker_model()
    
    # Prepare pairs for reranking
    pairs = [(query, ctx.get('text', '')) for ctx in contexts]
    
    # Get reranking scores
    scores = reranker.predict(pairs)
    
    # Sort by score
    ranked_contexts = sorted(zip(contexts, scores), key=lambda x: x[1], reverse=True)
    
    return [ctx for ctx, score in ranked_contexts[:top_k]]

# -----------------------------------------------------------------------------
# Quick Fix 4: Semantic Similarity Evaluation
# -----------------------------------------------------------------------------

def evaluate_response_similarity(response: str, reference: str) -> float:
    """Compute semantic similarity between response and reference"""
    
    model = get_semantic_model()
    embeddings = model.encode([response, reference])
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    
    return float(similarity)

def evaluate_context_relevance(query: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate how relevant contexts are to the query"""
    
    if not contexts:
        return {"avg_relevance": 0.0, "max_relevance": 0.0}
    
    model = get_semantic_model()
    
    # Encode query and contexts
    query_embedding = model.encode(query)
    context_embeddings = model.encode(contexts)
    
    # Compute similarities
    similarities = util.cos_sim(query_embedding, context_embeddings)[0]
    
    return {
        "avg_relevance": float(similarities.mean()),
        "max_relevance": float(similarities.max()),
        "min_relevance": float(similarities.min())
    }

# -----------------------------------------------------------------------------
# Quick Fix 5: Integrated Enhancement Function
# -----------------------------------------------------------------------------

def enhance_rag_response(
    query: str,
    contexts: List[Dict],
    original_response: str,
    use_reranking: bool = True
) -> Tuple[str, List[Dict]]:
    """
    All-in-one function to enhance RAG response
    Returns: (enhanced_response, reranked_contexts)
    """
    
    # Step 1: Rerank contexts if needed
    if use_reranking and contexts:
        contexts = rerank_contexts(query, contexts, top_k=5)
    
    # Step 2: Extract text from contexts
    context_texts = [ctx.get('text', '') for ctx in contexts]
    
    # Step 3: Build better prompt
    prompt = build_better_prompt(query, context_texts)
    
    # Step 4: Enhance the response
    enhanced_response = enhance_response(original_response, query)
    
    return enhanced_response, contexts

# -----------------------------------------------------------------------------
# Usage Example - Add to your chains.py
# -----------------------------------------------------------------------------

def integrate_quick_fixes():
    """
    Example of how to integrate these fixes into your existing code
    
    In your chains.py, modify the rag_query function:
    
    from .quick_rag_fixes import enhance_rag_response, build_better_prompt
    
    # In rag_query function, after getting response:
    response = generate_mental_health_response(...)
    
    # Enhance it
    enhanced_response, reranked_contexts = enhance_rag_response(
        query=query,
        contexts=retrieved_results,
        original_response=response
    )
    
    # Use enhanced_response instead of response
    """
    pass

# -----------------------------------------------------------------------------
# Standalone Testing
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Test the enhancements
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
    
    # Test semantic evaluation
    reference = "It's completely normal to feel anxious before a job interview. This shows you care about the opportunity. Some strategies that might help include practicing deep breathing exercises, preparing answers to common questions, and doing mock interviews with a friend."
    
    similarity = evaluate_response_similarity(enhanced_response, reference)
    print(f"\nSemantic similarity to reference: {similarity:.3f}")