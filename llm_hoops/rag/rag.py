"""Provide RAG capabilities to a model."""

from mistralai import Mistral
from llm_hoops.utils import load_email
from pathlib import Path
import os
import faiss
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

CHUNK_SIZE = 2048
NUMBER_OF_CHUNKS = 2

@dataclass
class RAGOutput:
    """Hold all outputs from a RAG."""
    embeddings: np.ndarray
    vector_db: faiss.IndexFlatL2
    chunks: list[str]

def load_emails() -> Dict[str, str]:
    """Load e-mails from database."""
    email_path = Path("data/emails/")

    eml_dict: Dict[str, str] = {}

    for filename in os.listdir(email_path):
        if filename.lower().endswith(".eml"):
            file_path = email_path / filename
            content = load_email(file_path)
            eml_dict[os.path.splitext(filename)[0]] = content

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
    embedding_batch_response = client.embeddings.create(
        model="mistral-embed",
        inputs=input_text
    )
    embedding = embedding_batch_response.data[0].embedding
    if embedding is None:
        raise ValueError("Failed to retrieve embedding for the input text.")
    return embedding

def embed_email_db(client: Mistral, email_db: str) -> tuple[np.ndarray, list[str]]:
    """Split into chunks and embed."""
    chunks = [email_db[i:i + CHUNK_SIZE] for i in range(0, len(email_db), CHUNK_SIZE)]
    return np.array([get_text_embedding(client, chunk) for chunk in chunks]), chunks

def create_vector_db(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Create vector db."""
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")
    embeddings = embeddings.astype(np.float32)  # Ensure embeddings are float32
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)  # type: ignore[arg-type]
    return index

def retrieve_similar_chunks(embedded_question: np.ndarray, vector_db: faiss.IndexFlatL2, chunks: list[str]) -> list[str]:
    """Retrieve similar chunks."""
    if embedded_question.ndim != 1:
        raise ValueError("Embedded question must be a 1D array.")
    embedded_question = embedded_question.reshape(1, -1).astype(np.float32)  # Reshape to 2D float32
    _distances, indices = vector_db.search(embedded_question, NUMBER_OF_CHUNKS)  # type: ignore[arg-type]
    return [chunks[i] for i in indices.tolist()[0]]

def add_rag_capabilities(client: Mistral) -> RAGOutput:
    """Add rag capabilities."""
    emails = load_emails()
    email_db = concat_emails(emails)
    embeddings, chunks = embed_email_db(client, email_db)
    vector_db = create_vector_db(embeddings)
    return RAGOutput(embeddings, vector_db, chunks)
