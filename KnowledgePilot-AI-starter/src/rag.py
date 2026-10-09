import hashlib
import json
from pathlib import Path
from typing import Iterable

import chromadb
import requests
from sentence_transformers import SentenceTransformer
import fitz

COLLECTION_NAME = "knowledgepilot_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

_embedder = None


def get_embedder():
    """Load the embedding model once per Streamlit process."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def get_collection(data_dir: Path):
    client = chromadb.PersistentClient(path=str(data_dir / "chroma_db"))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def extract_upload(upload):
    name = Path(upload.name).name
    raw = upload.getvalue()
    records = []
    if name.lower().endswith(".pdf"):
        with fitz.open(stream=raw, filetype="pdf") as pdf:
            for page_number, page in enumerate(pdf, start=1):
                page_text = page.get_text("text")
                for chunk_index, chunk in enumerate(split_text(page_text)):
                    records.append({
                        "text": chunk,
                        "file": name,
                        "location": f"page {page_number}",
                        "chunk_index": chunk_index,
                    })
    elif name.lower().endswith(".txt"):
        text = raw.decode("utf-8", errors="replace")
        for chunk_index, chunk in enumerate(split_text(text)):
            records.append({
                "text": chunk,
                "file": name,
                "location": f"chunk {chunk_index + 1}",
                "chunk_index": chunk_index,
            })
    else:
        raise ValueError(f"Unsupported file type: {name}")
    return records


def ingest_files(uploads: Iterable, data_dir: Path) -> int:
    data_dir.mkdir(parents=True, exist_ok=True)
    collection = get_collection(data_dir)
    embedder = get_embedder()
    total = 0

    for upload in uploads:
        records = extract_upload(upload)
        if not records:
            continue
        ids, documents, metadatas = [], [], []
        file_hash = hashlib.sha256(upload.getvalue()).hexdigest()[:12]
        for i, record in enumerate(records):
            ids.append(f"{file_hash}-{i}")
            documents.append(record["text"])
            metadatas.append({
                "file": record["file"],
                "location": record["location"],
                "chunk_index": record["chunk_index"],
            })
        embeddings = embedder.encode(documents, normalize_embeddings=True).tolist()
        # Upsert makes re-indexing the same uploaded file idempotent.
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        total += len(records)
    return total


def retrieve(question: str, data_dir: Path, top_k: int = 4):
    collection = get_collection(data_dir)
    if collection.count() == 0:
        return []
    query_embedding = get_embedder().encode(
        [question], normalize_embeddings=True
    ).tolist()[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    passages = []
    for i, text in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i] or {}
        passages.append({
            "text": text,
            "file": metadata.get("file", "Unknown file"),
            "location": metadata.get("location", "Unknown location"),
            "distance": float(results["distances"][0][i]),
        })
    return passages


def answer_question(question: str, passages: list[dict], model: str = "llama3.2:3b"):
    context = "\n\n".join(
        f"[Source {i + 1}: {p['file']}, {p['location']}]\n{p['text']}"
        for i, p in enumerate(passages)
    )
    prompt = f"""You are KnowledgePilot, a careful document question-answering assistant.
Answer the user's question using only the supplied context.
If the context does not contain enough information, say you cannot find the answer in the indexed documents.
Do not invent facts. Refer to sources as [Source 1], [Source 2], etc.

CONTEXT:
{context}

QUESTION:
{question}
"""
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.1},
        },
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("message", {}).get("content", "").strip()


def list_indexed_files(data_dir: Path):
    try:
        collection = get_collection(data_dir)
        if collection.count() == 0:
            return []
        result = collection.get(include=["metadatas"])
        return sorted({
            metadata.get("file", "Unknown file")
            for metadata in (result.get("metadatas") or [])
        })
    except Exception:
        return []
