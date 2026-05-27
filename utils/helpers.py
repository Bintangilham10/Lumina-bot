"""Shared helpers for Lumina Doc."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".epub"}


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


def is_supported_file(path: str | Path) -> bool:
    """Return whether a file extension is supported."""
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def supported_extensions_text() -> str:
    """Return a human-readable list of supported extensions."""
    return ", ".join(sorted(SUPPORTED_EXTENSIONS))


def clean_text(text: str) -> str:
    """Normalize extracted text while preserving paragraph boundaries."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
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
