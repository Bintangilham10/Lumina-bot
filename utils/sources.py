"""Source citation helpers for retrieved document chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document


@dataclass(frozen=True)
class SourceReference:
    """Display metadata for one deduplicated source."""

    number: int
    key: tuple[str, str, str]
    filename: str
    page: str
    section: str
    snippet: str


def normalize_source_snippet(text: str, max_length: int) -> str:
    """Normalize and truncate source text for display."""
    snippet = " ".join(str(text).split())
    if max_length <= 0:
        return ""
    if len(snippet) > max_length:
        if max_length <= 3:
            return snippet[:max_length]
        snippet = f"{snippet[: max_length - 3].rstrip()}..."
    return snippet


def source_metadata(document: Document) -> tuple[str, str, str]:
    """Return stable source metadata from a retrieved document."""
    metadata = document.metadata or {}
    fallback_filename = Path(str(metadata.get("source") or "Document")).name
    filename = str(metadata.get("filename") or fallback_filename or "Document")
    page = str(metadata.get("page", "-"))
    section = str(metadata.get("section", "")).strip()
    return filename, page, section


def build_source_references(
    documents: list[Document],
    max_sources: int | None = None,
    snippet_length: int = 280,
) -> list[SourceReference]:
    """Build numbered, deduplicated source references from retrieved documents."""
    references: list[SourceReference] = []
    seen: set[tuple[str, str, str]] = set()

    for document in documents:
        key = source_metadata(document)
        if key in seen:
            continue
        seen.add(key)

        filename, page, section = key
        references.append(
            SourceReference(
                number=len(references) + 1,
                key=key,
                filename=filename,
                page=page,
                section=section,
                snippet=normalize_source_snippet(document.page_content, snippet_length),
            )
        )
        if max_sources is not None and len(references) >= max_sources:
            break

    return references


def format_source_context(documents: list[Document]) -> str:
    """Format retrieved documents with numbered labels for cited QA prompts."""
    references = build_source_references(documents, snippet_length=0)
    source_numbers = {reference.key: reference.number for reference in references}
    blocks: list[str] = []

    for document in documents:
        filename, page, section = source_metadata(document)
        source_number = source_numbers[(filename, page, section)]
        label = f"Source [{source_number}]: {filename} | page/section {page}"
        if section and section not in {page, f"Page {page}"}:
            label = f"{label} | {section}"
        blocks.append(f"{label}\n{document.page_content}")

    return "\n\n".join(blocks)
