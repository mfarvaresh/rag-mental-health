# RAG Mental Health

A Retrieval-Augmented Generation (RAG) system for mental health support using multiple LLM backends (Ollama, OpenAI, Gemini) with advanced prompt engineering and evaluation capabilities.

---

## Project Architectural Structure

```
rag-mental-health/
├── 📄 Configuration & Documentation
│   ├── README.md                    # Main documentation and usage guide
│   ├── PROJECT_REVIEW.md            # Technical review and analysis
│   ├── pyproject.toml               # Project metadata and dependencies
│   ├── requirements.txt             # Python dependencies
│   └── .gitignore                   # Git ignore patterns
│
├── 🖥️ User Interface
│   └── app_streamlit.py             # Advanced Streamlit UI (502 lines)
│
├── 📊 Data Pipeline
│   └── data/
│       ├── 📄 Raw Data Sources
│       │   ├── counselchat/         # Professional therapist responses
│       │   ├── reddit/              # Community discussions
│       │   ├── mind/                # Mind.org.uk resources
│       │   ├── pubmed/              # Medical research papers
│       │   └── who/                 # WHO mental health guidelines
│       │
│       ├── 📁 Processed Data
│       │   ├── chunks/              # Text chunks for RAG
│       │   ├── embeddings/          # Vector embeddings
│       │   └── vectordb/            # ChromaDB vector database
│       │
│       ├── 📁 Evaluation Data
│       │   ├── evaluations/         # User evaluations and metrics
│       │   └── logs/                # Interaction logs
│       │
│       └── 📁 Documentation
│           └── docs/                # Reference documents
│
├── 🔧 Core RAG Engine (src/ragmh/)
│   ├── __init__.py                  # Package exports
│   ├── __main__.py                  # CLI entry point
│   │
│   ├── 🔄 Data Processing
│   │   ├── ingest.py                # Data ingestion (376 lines)
│   │   ├── chunk.py                 # Text chunking (306 lines)
│   │   └── embed.py                 # Embedding generation (242 lines)
│   │
│   ├── �� AI/ML Components
│   │   ├── llm.py                   # LLM backends (167 lines)
│   │   ├── vanilla_llm.py           # Vanilla LLM comparison (196 lines)
│   │   └── response_quality.py      # Response quality metrics (22 lines)
│   │
│   ├── 🔍 Retrieval & Storage
│   │   ├── vectordb.py              # ChromaDB operations (342 lines)
│   │   └── quick_rag_fixes.py      # RAG enhancements (277 lines)
│   │
│   ├── ⚡ Core Pipeline
│   │   ├── chains.py                # Main RAG pipeline (188 lines)
│   │   └── config.py                # Configuration management (25 lines)
│   │
│   ├── ��️ Interface
│   │   ├── cli.py                   # Command-line interface (217 lines)
│   │   └── enhanced_prompts.py      # Advanced prompts (0 lines)
│   │
│   └── 🧪 Testing & Evaluation
│       └── tests/
│           └── eval_pipeline.py     # Automated evaluation (329 lines)
│
└── ��️ Utilities
    └── scripts/
        ├── build_index.py           # Vector index builder (53 lines)
        ├── generate_eval_data.py    # Evaluation data generator (344 lines)
        └── test_improvements.py     # Improvement testing (23 lines)
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

## Advanced Features

### 🤖 Enhanced Embeddings with Reason-ModernColBERT

The system supports **Reason-ModernColBERT** for enhanced semantic understanding and reasoning capabilities:

```bash
# Generate embeddings with Reason-ModernColBERT
python -m src.ragmh.embed reason

# Generate hybrid embeddings (combines both models)
python -m src.ragmh.embed hybrid

# Use in Streamlit app (add to sidebar)
```

**Benefits for Mental Health RAG:**
- **Enhanced Reasoning**: Better understanding of complex mental health queries
- **Context Awareness**: Improved distinction between symptoms, treatments, and support
- **Empathy Detection**: Better recognition of emotional content and support-seeking language
- **Multi-Source Optimization**: Enhanced retrieval across clinical and community content

**Usage Examples:**
```python
# In your code
from src.ragmh.embed import load_embedding_model, REASON_MODERN_COLBERT

# Load Reason-ModernColBERT
model = load_embedding_model(REASON_MODERN_COLBERT)

# Generate embeddings
embeddings = model.encode(["I'm feeling anxious about my future"])
```

---

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