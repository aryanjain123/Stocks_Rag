# RAG Question Answering System

This is a Retrieval-Augmented Generation (RAG) system that allows you to upload PDF documents and ask questions about their content.

## Features

- Upload one or more PDF documents
- Extract and chunk text from PDFs
- Create embeddings for text chunks
- Retrieve relevant chunks for a user query
- Generate answers using retrieved context
- Optional LLM integration (OpenAI)

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   pip install -U pip
   pip install -r requirements.txt
   ```

3. (Optional) For LLM mode, copy `.env.example` to `.env` and add your OpenAI API key:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` to add your API key.

## Usage

Run the application:
```bash
python app.py
```

Then open the provided URL in your browser to use the Gradio interface.

## How It Works

1. Upload PDF documents using the "Index Documents" tab
2. The system will extract text, chunk it, and create embeddings
3. Switch to the "Ask Questions" tab
4. Enter your question and click "Get Answer"
5. The system will retrieve relevant chunks and generate an answer

## Architecture

- **PDF Processing**: Uses `pypdf` to extract text from PDFs
- **Chunking**: Sliding window approach (800 characters with 120 character overlap)
- **Embeddings**: SentenceTransformer (`all-MiniLM-L6-v2`) for creating normalized vectors
- **Storage**: In-memory storage for document chunks and embeddings
- **Retrieval**: Cosine similarity to find relevant chunks
- **Question Answering**: Baseline synthesis or LLM integration (if configured)
- **UI**: Gradio interface with tabs for document indexing and question answering

## Development

### Project Structure

```
repo/
  app.py              # Main application entry point
  requirements.txt    # Python dependencies
  .env.example        # Environment variables example
  src/
    app/              # Gradio app & adapters
      gradio_app.py
    core/             # Core logic (chunking, embeddings, retrieval, qa)
      chunking.py
      embedder.py
      store.py
      retrieval.py
      qa.py
    utils/            # Utility functions
      io.py
  tests/              # Unit tests
    test_chunking.py
```

### Running Tests

```bash
pytest tests/
```