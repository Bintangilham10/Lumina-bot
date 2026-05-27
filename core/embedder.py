"""Embedding and vector store utilities."""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from utils.helpers import ensure_directory


DEFAULT_EMBEDDING_MODEL = "models/text-embedding-004"
DEFAULT_PERSIST_DIRECTORY = "chroma_db"


def create_embeddings(model: str = DEFAULT_EMBEDDING_MODEL) -> GoogleGenerativeAIEmbeddings:
    """Create Google Generative AI embeddings."""
    return GoogleGenerativeAIEmbeddings(model=model)


def create_vector_store(
    chunks: list[Document],
    collection_name: str,
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
) -> Chroma:
    """Create and persist a Chroma vector store from document chunks."""
    if not chunks:
        raise ValueError("No chunks were provided for embedding.")

    directory = ensure_directory(persist_directory)
    embeddings = create_embeddings()
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(directory),
    )


def load_vector_store(
    collection_name: str,
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
) -> Chroma:
    """Load an existing Chroma vector store collection."""
    directory = ensure_directory(persist_directory)
    return Chroma(
        collection_name=collection_name,
        embedding_function=create_embeddings(),
        persist_directory=str(directory),
    )
