"""
rag_utils.py — Retrieval-Augmented Generation (RAG) Pipeline
=============================================================
This module implements a complete RAG pipeline for CV analysis:

Pipeline steps:
  1. EXTRACT  — Read text from PDF or TXT file
  2. CHUNK    — Split text into overlapping windows
  3. EMBED    — Convert chunks to semantic vectors (sentence-transformers)
  4. INDEX    — Store vectors for fast similarity search (numpy / FAISS)
  5. RETRIEVE — Find most relevant chunks for a query
  6. AUGMENT  — Pass retrieved context to the LLM for grounded answers

Why RAG?
  Instead of dumping an entire CV into the LLM prompt (which can exceed
  token limits), we retrieve only the most relevant sections.
  This makes the LLM answer more accurate and cost-efficient.

Embedding Model: all-MiniLM-L6-v2
  - Small (80 MB), fast, runs locally — no API key needed for embeddings!
  - 384-dimensional vectors
  - Great for semantic similarity tasks
"""

import os
import re
import numpy as np
from typing import List, Tuple, Optional

# ── Lazy-load the embedding model to avoid slow startup ──
_embedding_model = None

def _get_embedding_model():
    """
    Load the SentenceTransformer model on first use.
    The model is downloaded automatically on first run (~80 MB).
    After that it's cached locally.
    """
    global _embedding_model
    if _embedding_model is None:
        print("[RAG] Loading embedding model (first time may take 30 seconds)...")
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[RAG] Embedding model loaded ✓")
    return _embedding_model


# ──────────────────────────────────────────────
# STEP 1: TEXT EXTRACTION
# ──────────────────────────────────────────────

def extract_text_from_file(file_path: str) -> str:
    """
    Extract plain text from an uploaded CV file.

    Supports:
      - .pdf  (uses PyPDF2)
      - .txt  (reads directly)
      - .docx (uses python-docx if installed)

    Args:
        file_path (str): Path to the uploaded file

    Returns:
        str: Extracted text, or an error string starting with "Error:"
    """
    if not file_path or not os.path.exists(file_path):
        return "Error: File not found. Please re-upload your CV."

    ext = os.path.splitext(file_path)[1].lower()

    # ── PDF Extraction ──
    if ext == ".pdf":
        try:
            import PyPDF2
            text = ""
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n[Page {page_num + 1}]\n{page_text}"

            if not text.strip():
                return "Error: Could not extract text from PDF. It may be image-based (scanned). Try a text-based PDF or copy-paste as TXT."

            return text.strip()

        except ImportError:
            return "Error: PyPDF2 not installed. Run: pip install PyPDF2"
        except Exception as e:
            return f"Error reading PDF: {str(e)}"

    # ── TXT Extraction ──
    elif ext == ".txt":
        try:
            # Try UTF-8 first, then fall back to latin-1
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="latin-1") as f:
                    return f.read().strip()
        except Exception as e:
            return f"Error reading TXT file: {str(e)}"

    # ── DOCX Extraction ──
    elif ext == ".docx":
        try:
            from docx import Document
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs if para.text])
            return text.strip()
        except ImportError:
            return "Error: python-docx not installed. Run: pip install python-docx"
        except Exception as e:
            return f"Error reading DOCX: {str(e)}"

    else:
        return f"Error: Unsupported file type '{ext}'. Please upload a PDF, TXT, or DOCX file."


# ──────────────────────────────────────────────
# STEP 2: TEXT CHUNKING
# ──────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping word-level chunks.

    Why overlap? So that sentences near chunk boundaries aren't cut off.
    Example: chunks of 300 words with 50-word overlap.

    Args:
        text (str): Full document text
        chunk_size (int): Words per chunk (default 300)
        overlap (int): Words shared between adjacent chunks (default 50)

    Returns:
        list[str]: List of text chunks
    """
    # Clean up excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    words = text.split()

    if len(words) == 0:
        return []

    # If text is very short, return as single chunk
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    step = chunk_size - overlap  # How far to advance each iteration

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)

        if end >= len(words):
            break  # Reached the end

    print(f"[RAG] Chunked into {len(chunks)} pieces (size={chunk_size}, overlap={overlap})")
    return chunks


# ──────────────────────────────────────────────
# STEP 3 & 4: EMBEDDING + INDEXING
# ──────────────────────────────────────────────

def create_embeddings(chunks: List[str]) -> np.ndarray:
    """
    Convert text chunks to embedding vectors using SentenceTransformers.

    Each chunk becomes a 384-dimensional vector.
    These vectors capture semantic meaning — similar text → similar vectors.

    Args:
        chunks (list[str]): Text chunks from chunk_text()

    Returns:
        np.ndarray: Shape (num_chunks, 384) — float32 embeddings
    """
    if not chunks:
        return np.array([])

    model = _get_embedding_model()

    # batch_size=32 speeds up processing for large CVs
    embeddings = model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True
    )

    # Normalize to unit length — enables cosine similarity via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)   # Avoid divide-by-zero
    embeddings = embeddings / norms

    print(f"[RAG] Created {len(embeddings)} embeddings (dim={embeddings.shape[1]})")
    return embeddings.astype(np.float32)


# ──────────────────────────────────────────────
# STEP 5: RETRIEVAL
# ──────────────────────────────────────────────

def search_relevant_chunks(
    query: str,
    chunks: List[str],
    embeddings: np.ndarray,
    top_k: int = 4
) -> List[str]:
    """
    Find the most semantically relevant chunks for a given query.

    Uses cosine similarity (dot product of normalized vectors).
    Higher score = more relevant to the query.

    Args:
        query (str): The search query (e.g., job description or question)
        chunks (list[str]): Original text chunks
        embeddings (np.ndarray): Pre-computed chunk embeddings
        top_k (int): Number of top chunks to return (default 4)

    Returns:
        list[str]: Top-k most relevant text chunks
    """
    if not chunks or embeddings is None or len(embeddings) == 0:
        return []

    model = _get_embedding_model()

    # Embed the query (and normalize it)
    query_emb = model.encode([query], convert_to_numpy=True).astype(np.float32)
    query_norm = np.linalg.norm(query_emb)
    if query_norm > 0:
        query_emb = query_emb / query_norm

    # Compute cosine similarity scores: shape (num_chunks,)
    scores = np.dot(embeddings, query_emb.T).flatten()

    # Get indices of top_k highest scores
    top_k = min(top_k, len(chunks))
    top_indices = np.argsort(scores)[-top_k:][::-1]   # Sort descending

    relevant = [chunks[i] for i in top_indices]

    # Debug output for students
    top_scores = [round(float(scores[i]), 3) for i in top_indices]
    print(f"[RAG] Retrieved {len(relevant)} chunks — similarity scores: {top_scores}")

    return relevant


def build_rag_context(query: str, chunks: List[str], embeddings: np.ndarray, top_k: int = 4) -> str:
    """
    Convenience function: retrieve chunks and format them into a context string
    ready to be injected into an LLM prompt.

    Args:
        query (str): Question or job description
        chunks (list): CV text chunks
        embeddings (np.ndarray): Precomputed embeddings
        top_k (int): How many chunks to retrieve

    Returns:
        str: Formatted context block for LLM prompt
    """
    relevant = search_relevant_chunks(query, chunks, embeddings, top_k)

    if not relevant:
        return "No CV content available."

    context_parts = []
    for i, chunk in enumerate(relevant, 1):
        context_parts.append(f"[CV Section {i}]\n{chunk}")

    return "\n\n".join(context_parts)
