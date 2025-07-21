"""Local LLM wrapper for ollama/phi3:mini - functional style"""
import requests
import json
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_MODEL = "phi3:mini"
BASE_URL = "http://localhost:11434"
DEFAULT_TEMP = 0.3
DEFAULT_MAX_TOKENS = 1024

def verify_ollama_connection(base_url: str = BASE_URL, model: str = DEFAULT_MODEL) -> bool:
    """Check if ollama is running and model is available"""
    try:
        response = requests.get(f"{base_url}/api/tags")
        models = response.json()
        available_models = [m['name'] for m in models.get('models', [])]
        
        if model not in available_models:
            logger.error(f"Model {model} not found. Available: {available_models}")
            return False
        
        logger.info(f"Connected to Ollama with model: {model}")
        return True
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama. Is it running?")
        return False

def generate_response(prompt: str, 
                     context: Optional[List[str]] = None,
                     system_prompt: Optional[str] = None,
                     model: str = DEFAULT_MODEL,
                     temperature: float = DEFAULT_TEMP,
                     max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """Generate response from ollama with optional context"""
    
    # Build the full prompt
    full_prompt = ""
    
    if system_prompt:
        full_prompt += f"System: {system_prompt}\n\n"
    
    if context:
        full_prompt += "Context:\n"
        for i, ctx in enumerate(context):
            full_prompt += f"{i+1}. {ctx}\n"
        full_prompt += "\n"
    
    full_prompt += f"User: {prompt}\n"
    full_prompt += "Assistant: "
    
    # Call Ollama API
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/generate",
            json=payload
        )
        response.raise_for_status()
        
        result = response.json()
        return result['response'].strip()
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return f"Error: {str(e)}"

def generate_mental_health_response(user_query: str,
                                  retrieved_contexts: List[str]) -> str:
    """Generate a mental health focused response"""
    
    system_prompt = """You are a supportive mental health assistant. Your role is to:
- Provide empathetic, evidence-based responses
- Draw from the provided context when relevant
- Avoid diagnosing or prescribing medication
- Encourage professional help when appropriate
- Maintain a warm, non-judgmental tone"""
    
    return generate_response(
        prompt=user_query,
        context=retrieved_contexts,
        system_prompt=system_prompt
    )


# Quick test
if __name__ == "__main__":
    if verify_ollama_connection():
        # Test basic generation
        response = generate_response("What is anxiety?")
        print("Basic response:", response[:200], "...")
        
        # Test with context
        context = ["Anxiety is a feeling of worry or fear", 
                   "Common symptoms include rapid heartbeat"]
        response = generate_mental_health_response(
            "I feel anxious all the time",
            context
        )
        print("\nMental health response:", response[:200], "...")