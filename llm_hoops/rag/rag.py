"""Provide RAG capabilities to a model."""

from mistralai import Mistral
from llm_hoops.utils import load_email
from pathlib import Path
import os
import faiss
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
import tiktoken
import time
import re

CHUNK_SIZE = 2048
MAX_TOKENS_PER_CHUNK = 500  # ~500 tokens per chunk for embeddings
NUMBER_OF_CHUNKS = 2
USE_TOKENIZED_CHUNKS = True
MAX_RETRIES = 5


@dataclass
class RAGOutput:
    """Hold all outputs from a RAG."""

    embeddings: np.ndarray
    vector_db: faiss.IndexFlatL2
    chunks: list[str]


def remove_urls(text: str) -> str:
    """Remove all URLs from the text."""
    url_pattern = r'https?://\S+|www\.\S+'
    return re.sub(url_pattern, '', text)

def load_emails() -> Dict[str, str]:
    """Load e-mails from database."""
    email_path = Path("data/emails/")

    eml_dict: Dict[str, str] = {}

    for filename in os.listdir(email_path):
        if filename.lower().endswith(".eml"):
            file_path = email_path / filename
            content = load_email(file_path)
            eml_dict[os.path.splitext(filename)[0]] = remove_urls(content)

    return eml_dict


def concat_emails(loaded_emails: Dict[str, str]) -> str:
    """Concatenate all e-mails to a single database."""
    parts = []
    for title, body in loaded_emails.items():
        section = f"TITLE: {title}\n{body.strip()}"
        parts.append(section)
    return "\n\n" + "\n\n".join(parts) + "\n\n"


def get_text_embedding(client: Mistral, input_text: str) -> Optional[list[float]]:
    """Get text embedding for a given string."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(
                model="mistral-embed", inputs=input_text
            )
            embedding = response.data[0].embedding
            if embedding is None:
                raise ValueError("Failed to retrieve embedding.")
            return embedding
        except Exception as e:
            if "rate limit" in str(e).lower():
                wait = 2**attempt
                print(f"Rate limit hit, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed to embed text after {MAX_RETRIES} retries.")


def chunk_text(base_text: str) -> list[str]:
    """Chunk text for embedding."""
    if USE_TOKENIZED_CHUNKS:
        enc = tiktoken.get_encoding("cl100k_base")  # Mistral embedding tokenizer
        tokens = enc.encode(base_text)
        chunks = []
        for i in range(0, len(tokens), MAX_TOKENS_PER_CHUNK):
            chunk_tokens = tokens[i : i + MAX_TOKENS_PER_CHUNK]
            chunks.append(enc.decode(chunk_tokens))
        return chunks
    return [base_text[i : i + CHUNK_SIZE] for i in range(0, len(base_text), CHUNK_SIZE)]


def embed_email_db(client: Mistral, email_db: str) -> tuple[np.ndarray, list[str]]:
    """Split into chunks and embed."""
    chunks = chunk_text(email_db)
    embeddings = []
    for chunk in chunks:
        embeddings.append(get_text_embedding(client, chunk))
    return np.array(embeddings, dtype=np.float32), chunks


def create_vector_db(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Create vector db."""
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")
    embeddings = embeddings.astype(np.float32)  # Ensure embeddings are float32
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)  # type: ignore[arg-type]
    return index


def retrieve_similar_chunks(
    embedded_question: np.ndarray, vector_db: faiss.IndexFlatL2, chunks: list[str]
) -> list[str]:
    """Retrieve similar chunks."""
    embedded_question = embedded_question.reshape(1, -1).astype(
        np.float32
    )  # Reshape to 2D float32
    _distances, indices = vector_db.search(embedded_question, NUMBER_OF_CHUNKS)  # type: ignore[arg-type]
    return [chunks[i] for i in indices.tolist()[0]]

def startup_rag(client:Mistral) -> RAGOutput:
    """Startup RAG."""
    emails = load_emails()
    email_db = concat_emails(emails)
    embeddings, chunks = embed_email_db(client, email_db)
    vector_db = create_vector_db(embeddings)
    return RAGOutput(embeddings, vector_db, chunks)

def add_rag_capabilities(client: Mistral,prompt:str) -> list[str]:
    """Add rag capabilities."""
    rag = startup_rag(client)

    embedded_question = np.array([get_text_embedding(client,prompt)])
    return retrieve_similar_chunks(embedded_question,rag.vector_db,rag.chunks)
    
