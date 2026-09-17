"""Embedding and vector store utilities."""

from __future__ import annotations

import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from utils.helpers import ensure_directory


EMBEDDING_MODEL_ENV_VAR = "GEMINI_EMBEDDING_MODEL"
DEFAULT_EMBEDDING_MODEL = "text-embedding-004"
DEFAULT_PERSIST_DIRECTORY = "chroma_db"
DEFAULT_EMBEDDING_BATCH_SIZE = 100


def resolve_embedding_model(model: str | None = None) -> str:
    """Resolve the embedding model from an explicit value, environment, or default."""
    resolved_model = model or os.getenv(EMBEDDING_MODEL_ENV_VAR) or DEFAULT_EMBEDDING_MODEL
    return resolved_model.strip().strip("'\"") or DEFAULT_EMBEDDING_MODEL


def create_embeddings(model: str | None = None) -> GoogleGenerativeAIEmbeddings:
    """Create Google Generative AI embeddings."""
    return GoogleGenerativeAIEmbeddings(model=resolve_embedding_model(model))


def create_vector_store(
    chunks: list[Document],
    collection_name: str,
    persist_directory: str | Path | None = DEFAULT_PERSIST_DIRECTORY,
    embedding_model: str | None = None,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> Chroma:
    """Create a Chroma vector store from document chunks with cosine distance and batching."""
    if not chunks:
        raise ValueError("No chunks were provided for embedding.")

    persist_path = None
    if persist_directory is not None:
        persist_path = str(ensure_directory(persist_directory))

    embeddings = create_embeddings(embedding_model)
    first_batch = chunks[:batch_size]
    store = Chroma.from_documents(
        documents=first_batch,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_path,
        collection_metadata={"hnsw:space": "cosine"},
    )

    remaining_chunks = chunks[batch_size:]
    if remaining_chunks:
        for i in range(0, len(remaining_chunks), batch_size):
            store.add_documents(remaining_chunks[i : i + batch_size])

    return store


def vector_store_document_count(vector_store: Chroma) -> int:
    """Return the number of stored vectors in a Chroma collection."""
    collection = getattr(vector_store, "_collection", None)
    if collection is not None and hasattr(collection, "count"):
        return int(collection.count())

    records = vector_store.get(include=[])
    return len(records.get("ids", []))


def load_vector_store(
    collection_name: str,
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
    embedding_model: str | None = None,
) -> Chroma:
    """Load an existing Chroma vector store collection with cosine distance."""
    directory = ensure_directory(persist_directory)
    return Chroma(
        collection_name=collection_name,
        embedding_function=create_embeddings(embedding_model),
        persist_directory=str(directory),
        collection_metadata={"hnsw:space": "cosine"},
    )
