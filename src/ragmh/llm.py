import os
import json
import logging
from typing import List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_MODEL      = "phi3:mini"
BASE_URL           = "http://localhost:11434"
DEFAULT_TEMP       = 0.3
DEFAULT_MAX_TOKENS = 1024

from .quick_rag_fixes import (
    ENHANCED_SYSTEM_PROMPT,
    build_better_prompt,
    enhance_response,
)

def verify_ollama_connection(
    base_url: str = BASE_URL,
    model: str = DEFAULT_MODEL,
) -> bool:
    """Check Ollama connectivity and model availability."""
    try:
        res = requests.get(f"{base_url}/api/tags")
        res.raise_for_status()
        models = [m["name"] for m in res.json().get("models", [])]
        if model not in models:
            logger.error(f"Model “{model}” not found. Available: {models}")
            return False
        return True
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama. Is it running?")
        return False

def build_prompt(
    user_query: str,
    contexts: List[str],
    system_prompt: Optional[str] = None,
) -> str:
    """Kept for compatibility; use build_better_prompt instead."""
    prompt = ""
    if system_prompt:
        prompt += f"System: {system_prompt}\n\n"
    if contexts:
        prompt += "Relevant Information:\n"
        for i, ctx in enumerate(contexts, 1):
            prompt += f"[{i}] {ctx}\n"
        prompt += "\n"
    prompt += f"User: {user_query}\nAssistant: "
    return prompt

def generate_response(
    prompt: str,
    context: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMP,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    llm: str = "ollama",
) -> str:
    """Backend-agnostic generation with quality fixes."""
    final_prompt = build_better_prompt(prompt, context or [])
    if llm == "ollama":
        payload = {
            "model": model,
            "prompt": final_prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            res = requests.post(f"{BASE_URL}/api/generate", json=payload)
            res.raise_for_status()
            raw = res.json()["response"].strip()
            return enhance_response(raw, prompt)
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return f"Error: {e}"
    elif llm == "gpt-3.5-turbo-0125":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return "Error: OPENAI_API_KEY not set."
        try:
            import openai
            client = openai.OpenAI(api_key=openai_api_key)
            chat = client.chat.completions.create(
                model=llm,
                messages=[
                    {"role": "system", "content": system_prompt or ENHANCED_SYSTEM_PROMPT},
                    {"role": "user", "content": final_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw = chat.choices[0].message.content.strip()
            return enhance_response(raw, prompt)
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return f"Error: {e}"
    elif llm == "gemini-1.5-flash-8b":
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            return "Error: GEMINI_API_KEY not set."
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_api_key)
            model_ = genai.GenerativeModel("models/gemini-1.5-flash-8b")
            resp = model_.generate_content(final_prompt)
            raw = resp.text.strip()
            return enhance_response(raw, prompt)
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return f"Error: {e}"
    else:
        return f"Error: Unknown backend “{llm}”"

def generate_mental_health_response(
    user_query: str,
    retrieved_contexts: List[str],
    llm: str = "ollama",
    temperature: float = 0.7,
    max_tokens: int = 300,
) -> str:
    """High-level wrapper used by the RAG pipeline."""
    return generate_response(
        prompt=user_query,
        context=retrieved_contexts,
        system_prompt=ENHANCED_SYSTEM_PROMPT,
        llm=llm,
        temperature=temperature,
        max_tokens=max_tokens,
    )

if __name__ == "__main__":
    if verify_ollama_connection():
        print(generate_response("What is anxiety?")[:250], "…")
