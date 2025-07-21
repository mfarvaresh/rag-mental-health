# RAG Mental Health

# RAG Mental Health Support System

A Retrieval-Augmented Generation system for mental health support using Ollama and ChromaDB.

## Team Members
- Mokhtar Farvaresh (@mfarvaresh)
- Jitendrah (@jitendrasah123)
- kartik
- baral
- mithma

## Prerequisites
- Python 3.8+
- Ollama installed
- 8GB RAM minimum

## Setup Instructions

### 1. Clone the Repository

git clone https://github.com/YOUR_USERNAME/rag-mental-health.git
cd rag-mental-health



### 2. Install Ollama
Download from: https://ollama.com/download

### 3. Pull phi3 model
ollama pull phi3:mini

### 4. Install Python Dependencies
pip install -r requirements.txt

### 5. Run Setup
python -m ragmh setup --source counselchat

Project Structure

rag-mental-health/
├── src/ragmh/          # Main package
├── data/               # Data storage (git-ignored)
├── tests/              # Unit tests
└── requirements.txt    # Dependencies

Usage
# Query the system

python -m ragmh query "How to deal with anxiety?"

# Interactive chat

python -m ragmh chat