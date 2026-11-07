# Hugging Face RAG – Document Analysis System

An AI-powered document analysis application using Retrieval-Augmented Generation (RAG).

## Features

- PDF document processing
- Semantic document search
- Hugging Face embeddings
- Context-based question answering
- Source page references
- Local LLM integration

## Technology Stack

- Python
- Hugging Face Transformers
- Sentence Transformers
- FAISS
- Ollama
- Gradio

## Installation

Clone the repository:

```bash
git clone https://github.com/nethisai95-cell/Hugging_face_RAG.git
cd Hugging_face_RAG
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the application

```bash
python app.py
```

## Notes

Place your own PDF documents in the local documents folder. Private documents, model weights, and environment files should not be committed to GitHub.