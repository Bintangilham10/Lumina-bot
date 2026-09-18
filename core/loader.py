"""Document loading utilities for PDF, DOCX, EPUB, TXT, and MD files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
import zipfile

import docx
import fitz
from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT
from ebooklib import epub
from langchain_core.documents import Document

from utils.helpers import clean_text, is_supported_file, supported_extensions_text


MAX_ZIP_ENTRY_COUNT = 2000
MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES = 25 * 1024 * 1024
MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
MAX_ZIP_COMPRESSION_RATIO = 1000.0
MIN_RATIO_CHECK_BYTES = 1024 * 1024


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
    _validate_file_signature(path, suffix)
    if suffix == ".pdf":
        documents, total_pages = _load_pdf(path)
    elif suffix == ".docx":
        documents = _load_docx(path)
        total_pages = len(documents)
    elif suffix == ".epub":
        documents = _load_epub(path)
        total_pages = len(documents)
    elif suffix in {".txt", ".md"}:
        documents = _load_text(path, suffix)
        total_pages = len(documents)
    else:
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. Supported formats: {supported_extensions_text()}."
        )

    if not documents:
        if suffix == ".pdf":
            raise ValueError(
                "Tidak ada teks yang dapat dibaca di berkas PDF ini. "
                "Jika berkas berupa pindaian (scan) atau gambar, pastikan dokumen telah diproses OCR terlebih dahulu."
            )
        raise ValueError("No readable text was found in this document.")

    return LoadedDocument(
        filename=path.name,
        file_path=path.resolve(),
        file_type=suffix.lstrip(".").upper(),
        total_pages=total_pages,
        documents=documents,
    )


def _base_metadata(path: Path, file_type: str) -> dict[str, str]:
    return {
        "source": str(path),
        "filename": path.name,
        "file_type": file_type,
    }


def _validate_file_signature(path: Path, suffix: str) -> None:
    """Reject common extension spoofing before handing files to parsers."""
    if suffix == ".pdf":
        with path.open("rb") as file:
            header = file.read(5)
        if header != b"%PDF-":
            raise ValueError("File content does not look like a PDF document.")
        return

    if suffix == ".docx":
        _validate_zip_members(path, {"[Content_Types].xml", "word/document.xml"}, "DOCX")
        return

    if suffix == ".epub":
        _validate_epub_signature(path)
        return

    if suffix in {".txt", ".md"}:
        try:
            with path.open("r", encoding="utf-8") as file:
                sample = file.read(4096)
                if "\x00" in sample:
                    raise ValueError(f"File content does not look like a valid {suffix.lstrip('.').upper()} document.")
        except UnicodeDecodeError:
            try:
                with path.open("r", encoding="latin-1") as file:
                    sample = file.read(4096)
                    if "\x00" in sample:
                        raise ValueError(f"File content does not look like a valid {suffix.lstrip('.').upper()} document.")
            except Exception as exc:
                raise ValueError(f"File content cannot be read as text for {suffix.lstrip('.').upper()} document.") from exc
        return


def _validate_zip_members(path: Path, required_members: set[str], file_type: str) -> None:
    if not zipfile.is_zipfile(path):
        raise ValueError(f"File content does not look like a {file_type} document.")
    with zipfile.ZipFile(path) as archive:
        _validate_zip_archive_safety(archive, file_type)
        names = set(archive.namelist())
    if not required_members.issubset(names):
        raise ValueError(f"File content does not look like a {file_type} document.")


def _validate_epub_signature(path: Path) -> None:
    if not zipfile.is_zipfile(path):
        raise ValueError("File content does not look like an EPUB document.")
    with zipfile.ZipFile(path) as archive:
        _validate_zip_archive_safety(archive, "EPUB")
        names = set(archive.namelist())
        if "mimetype" not in names:
            raise ValueError("File content does not look like an EPUB document.")
        mimetype = archive.read("mimetype").decode("utf-8", errors="ignore").strip()
    if mimetype != "application/epub+zip":
        raise ValueError("File content does not look like an EPUB document.")


def _validate_zip_archive_safety(archive: zipfile.ZipFile, file_type: str) -> None:
    entries = [info for info in archive.infolist() if not info.is_dir()]
    if len(entries) > MAX_ZIP_ENTRY_COUNT:
        raise ValueError(
            f"{file_type} archive has too many files to process safely "
            f"({len(entries)} > {MAX_ZIP_ENTRY_COUNT})."
        )

    total_uncompressed = 0
    for entry in entries:
        total_uncompressed += entry.file_size
        if entry.file_size > MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES:
            raise ValueError(
                f"{file_type} archive entry '{entry.filename}' is too large to process safely."
            )
        compressed_size = max(entry.compress_size, 1)
        compression_ratio = entry.file_size / compressed_size
        if (
            entry.file_size >= MIN_RATIO_CHECK_BYTES
            and compression_ratio > MAX_ZIP_COMPRESSION_RATIO
        ):
            raise ValueError(
                f"{file_type} archive entry '{entry.filename}' has an unsafe compression ratio."
            )

    if total_uncompressed > MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES:
        raise ValueError(
            f"{file_type} archive expands beyond the safe processing limit "
            f"({total_uncompressed} bytes > {MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES} bytes)."
        )


def _build_pdf_page_sections(pdf: fitz.Document) -> dict[int, str]:
    """Map PDF page numbers to bookmark/outline section names when available."""
    page_sections: dict[int, str] = {}
    try:
        toc = pdf.get_toc()
    except Exception:
        return page_sections

    if not toc:
        return page_sections

    sorted_toc = [
        entry for entry in toc
        if len(entry) >= 3 and isinstance(entry[2], int) and entry[2] > 0
    ]
    sorted_toc.sort(key=lambda item: item[2])
    if not sorted_toc:
        return page_sections

    current_title = ""
    toc_idx = 0
    total_pages = len(pdf)

    for p in range(1, total_pages + 1):
        while toc_idx < len(sorted_toc) and p >= sorted_toc[toc_idx][2]:
            current_title = str(sorted_toc[toc_idx][1]).strip()
            toc_idx += 1
        if current_title:
            page_sections[p] = current_title

    return page_sections


def _load_pdf(path: Path) -> tuple[list[Document], int]:
    documents: list[Document] = []
    metadata = _base_metadata(path, "PDF")

    try:
        pdf = fitz.open(path)
    except fitz.FileDataError as exc:
        raise ValueError(f"Berkas PDF rusak atau tidak valid: {exc}") from exc

    with pdf:
        if pdf.is_encrypted:
            raise ValueError(
                "Berkas PDF ini terkunci atau dilindungi password. "
                "Buka proteksi dokumen PDF terlebih dahulu."
            )
        total_physical_pages = len(pdf)
        page_sections = _build_pdf_page_sections(pdf)
        for index, page in enumerate(pdf, start=1):
            text = clean_text(page.get_text("text"))
            if text:
                section_title = page_sections.get(index) or f"Page {index}"
                documents.append(
                    Document(
                        page_content=text,
                        metadata={**metadata, "page": index, "section": section_title},
                    )
                )

    return documents, total_physical_pages


def _load_docx(path: Path) -> list[Document]:
    try:
        document = docx.Document(path)
    except Exception as exc:
        raise ValueError(f"Berkas DOCX rusak atau tidak valid: {exc}") from exc

    metadata = _base_metadata(path, "DOCX")
    sections: list[Document] = []
    current_section_name = "Document"
    current_parts: list[str] = []

    for item in document.iter_inner_content():
        if isinstance(item, docx.text.paragraph.Paragraph):
            text = clean_text(item.text)
            if not text:
                continue
            style_name = getattr(getattr(item, "style", None), "name", "")
            if style_name.startswith("Heading"):
                if current_parts:
                    section_text = "\n\n".join(current_parts)
                    idx = len(sections) + 1
                    sections.append(
                        Document(
                            page_content=section_text,
                            metadata={**metadata, "page": idx, "section": current_section_name},
                        )
                    )
                    current_parts = []
                current_section_name = text
            current_parts.append(text)
        elif isinstance(item, docx.table.Table):
            table_rows: list[str] = []
            for r_idx, row in enumerate(item.rows):
                seen_cells: set[int] = set()
                cells: list[str] = []
                for cell in row.cells:
                    cell_id = getattr(cell, "_tc", id(cell))
                    if cell_id not in seen_cells:
                        seen_cells.add(cell_id)
                        cleaned = clean_text(cell.text)
                        if cleaned:
                            cells.append(cleaned)
                row_text = " | ".join(cells)
                if row_text:
                    table_rows.append(row_text)
                    if r_idx == 0 and len(item.rows) > 1:
                        table_rows.append(" | ".join("---" for _ in cells))
            if table_rows:
                current_parts.append("\n".join(table_rows))

    if current_parts:
        section_text = "\n\n".join(current_parts)
        idx = len(sections) + 1
        sections.append(
            Document(
                page_content=section_text,
                metadata={**metadata, "page": idx, "section": current_section_name},
            )
        )

    return sections


def _load_epub(path: Path) -> list[Document]:
    try:
        book = epub.read_epub(str(path))
    except Exception as exc:
        raise ValueError(f"Berkas EPUB rusak atau tidak valid: {exc}") from exc

    documents: list[Document] = []
    metadata = _base_metadata(path, "EPUB")

    for item in book.get_items_of_type(ITEM_DOCUMENT):
        if _is_epub_navigation_item(item):
            continue

        soup = BeautifulSoup(item.get_content(), "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = clean_text(soup.get_text(separator="\n"))
        if text:
            index = len(documents) + 1
            heading_tag = soup.find(["h1", "h2", "title"])
            heading_text = clean_text(heading_tag.get_text()) if heading_tag else ""
            title = heading_text or item.get_name() or f"Section {index}"
            documents.append(
                Document(
                    page_content=text,
                    metadata={**metadata, "page": index, "section": title},
                )
            )

    return documents


def _load_text(path: Path, suffix: str) -> list[Document]:
    file_type = suffix.lstrip(".").upper()
    metadata = _base_metadata(path, file_type)
    try:
        content = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        content = path.read_text(encoding="latin-1", errors="replace")

    text = clean_text(content)
    if not text:
        return []

    if suffix == ".md" and ("\n#" in text or text.startswith("#")):
        sections: list[Document] = []
        parts = re.split(r"(?m)^(#{1,3}\s+.+)$", text)
        current_title = "Document"
        current_body: list[str] = []

        for part in parts:
            if re.match(r"^#{1,3}\s+", part):
                if current_body:
                    body_text = clean_text("\n".join(current_body))
                    if body_text:
                        idx = len(sections) + 1
                        sections.append(
                            Document(
                                page_content=body_text,
                                metadata={**metadata, "page": idx, "section": current_title},
                            )
                        )
                current_title = clean_text(re.sub(r"^#{1,3}\s+", "", part))
                current_body = []
            else:
                current_body.append(part)

        if current_body:
            body_text = clean_text("\n".join(current_body))
            if body_text:
                idx = len(sections) + 1
                sections.append(
                    Document(
                        page_content=body_text,
                        metadata={**metadata, "page": idx, "section": current_title},
                    )
                )
        if sections:
            return sections

    return [
        Document(
            page_content=text,
            metadata={**metadata, "page": 1, "section": "Document"},
        )
    ]


def _is_epub_navigation_item(item) -> bool:
    """Return whether an EPUB document item is navigation-only content."""
    if isinstance(item, (epub.EpubNav, epub.EpubNcx)):
        return True
    name = Path(str(item.get_name() or "")).name.lower()
    return name in {"nav.xhtml", "toc.xhtml", "toc.html", "toc.ncx"}
