# Local RAG

A simple Retrieval-Augmented Generation (RAG) system built from scratch using a local LLM.

## 🚀 Overview

This project demonstrates the core RAG pipeline:

Document → Chunking → Embeddings → Vector Database → Retrieval → LLM → Answer

The system retrieves relevant information from a local document and uses a locally running Qwen3 8B model through Ollama to generate an answer based on the retrieved context.

## 🧠 Architecture

```text
                Local RAG Pipeline

                  document.txt
                       │
                       ▼
                    Chunking
                       │
                       ▼
              Sentence Transformers
                       │
                       ▼
                   Embeddings
                       │
                       ▼
                   ChromaDB
                       │
              ┌────────┴────────┐
              │                 │
        User Question      Stored Vectors
              │                 │
              ▼                 │
          Embedding             │
              │                 │
              └───────┬─────────┘
                      ▼
                Similarity Search
                      │
                      ▼
               Retrieved Context
                      │
                      ▼
                 Qwen3 8B
                  (Ollama)
                      │
                      ▼
                    Answer
🛠️ Technologies
Python
Sentence Transformers
ChromaDB
Ollama
Qwen3 8B
OpenAI-compatible Python client
📁 Project Structure
local-rag/
│
├── data/
│   └── document.txt
│
├── main.py
├── requirements.txt
└── .gitignore
⚙️ Setup
1. Clone the repository
git clone https://github.com/aarush006-cyber/local-rag.git
cd local-rag
2. Install dependencies
pip install -r requirements.txt
3. Install Ollama

Make sure Ollama is installed and running.

Then pull the model:

ollama pull qwen3:8b
4. Run the RAG system
python main.py

Ask a question when prompted.

🔍 How It Works
The document is loaded from data/document.txt.
The document is split into chunks.
Each chunk is converted into an embedding using all-MiniLM-L6-v2.
Embeddings are stored in ChromaDB.
The user's question is converted into an embedding.
ChromaDB retrieves the most relevant chunks.
The retrieved context is passed to Qwen3 8B.
Qwen3 generates the final answer using the retrieved information.
🔮 Future Improvements

This project is the baseline for experimenting with more advanced RAG techniques.

Planned improvements:

Improved chunking and chunk overlap
Metadata filtering
Multi-document support
Hybrid search
Reranking
Query transformation
RAG evaluation
Retrieval and answer quality benchmarking
👨‍💻 Author

Aarush Kumar

GitHub: https://github.com/aarush006-cyber