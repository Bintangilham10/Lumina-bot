"""Embedding and vector store utilities."""

from __future__ import annotations

import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from utils.helpers import ensure_directory


EMBEDDING_MODEL_ENV_VAR = "GEMINI_EMBEDDING_MODEL"
DEFAULT_EMBEDDING_MODEL = "models/text-embedding-004"
DEFAULT_PERSIST_DIRECTORY = "chroma_db"


def resolve_embedding_model(model: str | None = None) -> str:
    """Resolve the embedding model from an explicit value, environment, or default."""
    resolved_model = model or os.getenv(EMBEDDING_MODEL_ENV_VAR) or DEFAULT_EMBEDDING_MODEL
    return resolved_model.strip() or DEFAULT_EMBEDDING_MODEL


def create_embeddings(model: str | None = None) -> GoogleGenerativeAIEmbeddings:
    """Create Google Generative AI embeddings."""
    return GoogleGenerativeAIEmbeddings(model=resolve_embedding_model(model))


def create_vector_store(
    chunks: list[Document],
    collection_name: str,
    persist_directory: str | Path | None = DEFAULT_PERSIST_DIRECTORY,
    embedding_model: str | None = None,
) -> Chroma:
    """Create a Chroma vector store from document chunks."""
    if not chunks:
        raise ValueError("No chunks were provided for embedding.")

    persist_path = None
    if persist_directory is not None:
        persist_path = str(ensure_directory(persist_directory))

    embeddings = create_embeddings(embedding_model)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_path,
    )


def load_vector_store(
    collection_name: str,
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
    embedding_model: str | None = None,
) -> Chroma:
    """Load an existing Chroma vector store collection."""
    directory = ensure_directory(persist_directory)
    return Chroma(
        collection_name=collection_name,
        embedding_function=create_embeddings(embedding_model),
        persist_directory=str(directory),
    )
