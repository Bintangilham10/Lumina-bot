"""Document loading utilities for PDF, DOCX, and EPUB files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import docx
import fitz
from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT
from ebooklib import epub
from langchain_core.documents import Document

from utils.helpers import clean_text, is_supported_file, supported_extensions_text


@dataclass(frozen=True)
class LoadedDocument:
    """Loaded document payload and display metadata."""

    filename: str
    file_path: Path
    file_type: str
    total_pages: int
    documents: list[Document]


def load_document(file_path: str | Path) -> LoadedDocument:
    """Load a PDF, DOCX, or EPUB file into LangChain Document objects."""
    path = Path(file_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    if not path.is_file():
        raise ValueError(f"Document path is not a file: {path}")
    if not is_supported_file(path):
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. Supported formats: {supported_extensions_text()}."
        )

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        documents = _load_pdf(path)
    elif suffix == ".docx":
        documents = _load_docx(path)
    else:
        documents = _load_epub(path)

    if not documents:
        raise ValueError("No readable text was found in this document.")

    return LoadedDocument(
        filename=path.name,
        file_path=path.resolve(),
        file_type=suffix.lstrip(".").upper(),
        total_pages=len(documents),
        documents=documents,
    )


def _base_metadata(path: Path, file_type: str) -> dict[str, str]:
    return {
        "source": str(path),
        "filename": path.name,
        "file_type": file_type,
    }


def _load_pdf(path: Path) -> list[Document]:
    documents: list[Document] = []
    metadata = _base_metadata(path, "PDF")

    with fitz.open(path) as pdf:
        for index, page in enumerate(pdf, start=1):
            text = clean_text(page.get_text("text"))
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={**metadata, "page": index, "section": f"Page {index}"},
                    )
                )

    return documents


def _load_docx(path: Path) -> list[Document]:
    document = docx.Document(path)
    paragraphs = [clean_text(paragraph.text) for paragraph in document.paragraphs]
    table_rows: list[str] = []

    for table in document.tables:
        for row in table.rows:
            cells = [clean_text(cell.text) for cell in row.cells]
            row_text = " | ".join(cell for cell in cells if cell)
            if row_text:
                table_rows.append(row_text)

    text_parts = [paragraph for paragraph in paragraphs if paragraph] + table_rows
    text = "\n\n".join(text_parts)

    if not text:
        return []

    return [
        Document(
            page_content=text,
            metadata={**_base_metadata(path, "DOCX"), "page": 1, "section": "Document"},
        )
    ]


def _load_epub(path: Path) -> list[Document]:
    book = epub.read_epub(str(path))
    documents: list[Document] = []
    metadata = _base_metadata(path, "EPUB")

    for item in book.get_items_of_type(ITEM_DOCUMENT):
        if _is_epub_navigation_item(item):
            continue

        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = clean_text(soup.get_text(separator="\n"))
        if text:
            index = len(documents) + 1
            title = item.get_name() or f"Section {index}"
            documents.append(
                Document(
                    page_content=text,
                    metadata={**metadata, "page": index, "section": title},
                )
            )

    return documents


def _is_epub_navigation_item(item) -> bool:
    """Return whether an EPUB document item is navigation-only content."""
    if isinstance(item, epub.EpubNav):
        return True
    name = str(item.get_name() or "").lower()
    return name in {"nav.xhtml", "toc.xhtml", "toc.html"}
