# RAG Mental Health

A Retrieval-Augmented Generation (RAG) system for mental health support using Ollama, ChromaDB, and advanced prompt/response enhancements.

---

## Project Architecture

```
rag-mental-health/
├── src/
│   └── ragmh/
│        ├── chains.py            # Main RAG pipeline (retrieval, generation, post-processing)
│        ├── llm.py               # LLM backend abstraction (Ollama, OpenAI, Gemini)
│        ├── vectordb.py          # ChromaDB vector DB utilities (init, search, hybrid, rerank)
│        ├── quick_rag_fixes.py   # Drop-in prompt, rerank, and response quality enhancements
│        ├── cli.py               # Command-line interface (setup, query, chat, etc.)
│        ├── ingest.py            # Data ingestion (counselchat, reddit, mind.org.uk, etc.)
│        ├── chunk.py             # Document chunking utilities
│        ├── embed.py             # Embedding generation and management
│        ├── config.py            # Centralized configuration
│        ├── response_quality.py  # (Optional) Extra response post-processing
│        ├── enhanced_prompts.py  # (Optional) Prompt templates
│        └── ...                  # Other helpers and __init__.py
├── data/                        # Data storage (not tracked by git)
│   ├── chunks/                  # Chunked data
│   ├── embeddings/              # Embedding files
│   ├── vectordb/                # ChromaDB persistent storage
│   └── logs/                    # Query and evaluation logs
├── tests/
│   └── eval_pipeline.py         # Evaluation pipeline (semantic similarity, toxicity, etc.)
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Poetry/PEP 621 project config
└── README.md                    # This file
```

### Pipeline Overview
1. **Ingestion**: Download and preprocess mental health data from multiple sources.
2. **Chunking**: Split documents into semantically meaningful chunks.
3. **Embedding**: Generate vector embeddings for all chunks.
4. **Indexing**: Store chunks and embeddings in ChromaDB for fast retrieval.
5. **Retrieval**: Given a user query, retrieve relevant chunks (vector search + cross-encoder rerank).
6. **Prompting**: Build a context-rich prompt for the LLM using enhanced templates.
7. **Generation**: Generate a response using Ollama, OpenAI, or Gemini LLMs.
8. **Post-processing**: Polish the response for empathy, safety, and actionable advice.
9. **Evaluation**: Analyze system outputs for semantic similarity, context relevance, and toxicity.

---

## Prerequisites
- Python 3.8+
- Ollama installed (for local LLM)
- 8GB RAM minimum (more recommended for large models)
- (Optional) OpenAI and Gemini API keys for cloud LLMs

## Setup Instructions

### 1. Clone the Repository
```sh
git clone https://github.com/mfarvaresh/rag-mental-health.git
cd rag-mental-health
```

### 2. Install Ollama
Download from: https://ollama.com/download

### 3. Pull phi3 model (or other supported models)
```sh
ollama pull phi3:mini
```

### 4. Install Python Dependencies
```sh
pip install -r requirements.txt
```

### 5. (Optional) Set up API Keys
- For OpenAI (gpt-3.5-turbo-0125): set `OPENAI_API_KEY` in your environment or a `.env` file.
- For Gemini (gemini-1.5-flash-8b): set `GEMINI_API_KEY` in your environment or a `.env` file.

### 6. Run Setup
```sh
python -m src.ragmh.cli setup --source counselchat
```

---

## Usage
### Query the system
```sh
python -m src.ragmh.cli query "How to deal with anxiety?" --llm ollama
python -m src.ragmh.cli query "How to deal with anxiety?" --llm gpt-3.5-turbo-0125
python -m src.ragmh.cli query "How to deal with anxiety?" --llm gemini-1.5-flash-8b
```

### Interactive chat
```sh
python -m src.ragmh.cli chat
```

### Evaluation Pipeline
To evaluate system outputs (semantic similarity, context relevance, response quality, toxicity):
```sh
python tests/eval_pipeline.py --file <log_or_results.json>
```
- Logs are saved in `data/logs/` by default.
- Evaluation results are saved as `<logfile>_eval_results.json`.

### Compare Command
- The `compare` command is currently **unavailable**. To restore, re-implement `compare_with_vanilla_llm` in `chains.py` and wire it up in the CLI.

---

## Enhancements & Features
- **Prompt engineering** and **response post-processing** are applied throughout the pipeline (see `src/ragmh/quick_rag_fixes.py`).
- **Context reranking** uses a cross-encoder for improved relevance.
- **Evaluation** includes semantic similarity, context relevance, and toxicity checks.
- **Multiple LLM backends**: Ollama (local), OpenAI (gpt-3.5-turbo-0125), Gemini (gemini-1.5-flash-8b).
- **Modular design**: Each pipeline step is a separate module for easy extension.

## API Key Setup
- Create a `.env` file in the project root with:
```
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...your_key...
```

## Troubleshooting
- If you see errors about missing models or API keys, check your `.env` and ensure Ollama is running.
- For GPU acceleration, ensure you have a compatible version of PyTorch installed.
- If you see `[compare] This feature is currently unavailable...`, see the note above.

## License
MIT