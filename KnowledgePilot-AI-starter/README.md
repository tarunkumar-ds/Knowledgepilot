# KnowledgePilot AI

A local-first Retrieval-Augmented Generation (RAG) assistant built with Python, Streamlit, Ollama, Sentence Transformers, and ChromaDB. No Google or OpenAI API keys are required.

## Features in this starter version

- Upload PDF and TXT documents
- Extract text locally and split it into overlapping chunks
- Create local embeddings with `all-MiniLM-L6-v2`
- Persist vectors in ChromaDB
- Retrieve relevant passages for a question
- Generate grounded answers with a local Ollama model
- Display source file and page/chunk references

## Requirements

- Windows 10/11
- Python 3.10 or newer recommended
- 16 GB RAM recommended for this setup
- Internet for the initial downloads only

## Setup on Windows PowerShell

1. Install Ollama from https://ollama.com/download/windows
2. Open a new terminal and download the model:

   ```powershell
   ollama pull llama3.2:3b
   ollama run llama3.2:3b
   ```

   Type a test prompt, then enter `/bye` to exit. Keep Ollama running in the background.

3. In another terminal, open this project folder and create an environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Start the application:

   ```powershell
   streamlit run app.py
   ```

5. In the browser, upload a PDF or TXT file in the sidebar, click **Index uploaded documents**, then ask a question.

The first run downloads the embedding model from Hugging Face. After the model files and Ollama model are downloaded, inference and embeddings run locally. Keep internet access enabled if you need to download dependencies or models later.

## Run tests

```powershell
pytest -q
```

## Troubleshooting

- **`ollama` is not recognized:** finish installing Ollama, restart the terminal, and run `ollama --version`.
- **Model not found:** run `ollama pull llama3.2:3b`.
- **Connection refused on port 11434:** start the Ollama desktop app.
- **Slow responses:** a CPU-only system may generate slowly. Close memory-heavy applications and use short questions/context.
- **PDF has no text:** this version extracts embedded text; scanned PDFs need OCR, which can be added later.

## Planned roadmap

1. Better retrieval evaluation and source citation checks
2. Tool calling with an allowlisted calculator and local document-search tool
3. MCP server exposing approved tools
4. Tests, evaluation dataset, and deployment documentation
