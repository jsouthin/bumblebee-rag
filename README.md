# LangChain + OpenAI RAG Project

A Retrieval-Augmented Generation (RAG) system built with LangChain and OpenAI, supporting multiple document sources including web pages, PDFs, and Google Drive documents.

## Project Structure

```
root/
├── src/
│   ├── __init__.py
│   ├── document_loaders.py   # Document loading utilities
│   ├── rag_pipeline.py       # Main RAG pipeline implementation
│   └── vector_store.py       # Vector store management
├── notebooks/                   # Jupyter notebooks for demos and experiments
│   └── rag_demo.ipynb           # Main notebook.  Start here
├── tests/                       # Test files and optional configurations (for test discovery etc.)
│   └── test_document_loaders.py # Test file for document_loaders.py
├── data/                        # Data storage (vector store, etc.) - ignored in .gitignore
├── docs/                        # Documentation
├── logs/                        # Log files
├── configs/                     # Configuration files
├── README.md                    # This README document
└── requirements.txt             # Project dependencies
```

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export LANGSMITH_TRACING=true
export LANGSMITH_ENDPOINT="https://{langsmith_region}.api.smith.langchain.com"
export LANGSMITH_API_KEY="your_langsmith_api_key"
export OPENAI_API_KEY="your_openai_api_key"
```

## Usage

Check the notebooks in the `notebooks/` directory for usage examples and demonstrations.
