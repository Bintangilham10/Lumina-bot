"""Shared helpers for Lumina Doc."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".epub"}
DEFAULT_MAX_FILE_SIZE_MB = 50
DEFAULT_MAX_PAGES = 500
DEFAULT_MAX_CHUNKS = 1000


def load_environment() -> str:
    """Load environment variables and return the Google API key."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not configured. Copy .env.example to .env and add your key."
        )
    return api_key


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 hash for a file."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_supported_file(path: str | Path) -> bool:
    """Return whether a file extension is supported."""
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def supported_extensions_text() -> str:
    """Return a human-readable list of supported extensions."""
    return ", ".join(sorted(SUPPORTED_EXTENSIONS))


def document_collection_name(
    filename_stem: str,
    document_hash: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str | None = None,
) -> str:
    """Build a stable collection name for a document and chunking settings."""
    return safe_collection_name(
        [
            "lumina",
            document_hash[:16],
            f"cs{chunk_size}",
            f"co{chunk_overlap}",
            embedding_model or "",
            filename_stem,
        ]
    )


def validate_file_size(file_size_bytes: int, max_file_size_mb: int) -> None:
    """Reject files that exceed the configured size guard."""
    if max_file_size_mb <= 0:
        return

    max_file_size_bytes = max_file_size_mb * 1024 * 1024
    if file_size_bytes > max_file_size_bytes:
        actual_mb = file_size_bytes / (1024 * 1024)
        raise ValueError(
            f"Document size {actual_mb:.1f} MB exceeds the configured "
            f"{max_file_size_mb} MB limit."
        )


def validate_document_limits(
    total_pages: int,
    total_chunks: int,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> None:
    """Reject documents that are too large to index comfortably."""
    if max_pages > 0 and total_pages > max_pages:
        raise ValueError(
            f"Document has {total_pages} pages/sections, which exceeds the "
            f"configured limit of {max_pages}."
        )
    if max_chunks > 0 and total_chunks > max_chunks:
        raise ValueError(
            f"Document produced {total_chunks} chunks, which exceeds the "
            f"configured limit of {max_chunks}."
        )


def clean_text(text: str) -> str:
    """Normalize extracted text while preserving paragraph boundaries."""
    text = str(text).replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_collection_name(parts: Iterable[str]) -> str:
    """Build a Chroma-compatible collection name from arbitrary text parts."""
    raw_name = "-".join(part for part in parts if part).lower()
    name = re.sub(r"[^a-z0-9_-]+", "-", raw_name).strip("-_")
    name = re.sub(r"-{2,}", "-", name)
    if len(name) < 3:
        name = f"lumina-{name or 'doc'}"
    return name[:63].strip("-_") or "lumina-doc"
