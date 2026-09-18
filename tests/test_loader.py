"""Tests for document loading validation."""

from __future__ import annotations

import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest.mock import patch

import docx
import fitz
from ebooklib import epub

from core.loader import load_document


class LoaderTests(unittest.TestCase):
    def test_load_document_rejects_missing_file(self) -> None:
        missing_path = Path.cwd() / "missing-document.pdf"

        with self.assertRaisesRegex(FileNotFoundError, "Document not found"):
            load_document(missing_path)

    def test_load_document_rejects_unsupported_file_type(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "notes.xyz"
            path.write_text("Hello world", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unsupported file type"):
                load_document(path)

    def test_load_document_rejects_directory_paths(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "book.pdf"
            path.mkdir()

            with self.assertRaisesRegex(ValueError, "not a file"):
                load_document(path)

    def test_load_document_rejects_spoofed_pdf_content(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "sample.pdf"
            path.write_text("not actually a pdf", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "PDF"):
                load_document(path)

    def test_load_document_rejects_spoofed_docx_content(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "sample.docx"
            path.write_text("not actually a docx", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "DOCX"):
                load_document(path)

    def test_load_document_rejects_spoofed_epub_content(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "sample.epub"
            path.write_text("not actually an epub", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "EPUB"):
                load_document(path)

    def test_load_document_rejects_docx_zip_with_too_many_entries(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "many-files.docx"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("[Content_Types].xml", "<Types />")
                archive.writestr("word/document.xml", "<document />")
                archive.writestr("extra.txt", "extra")

            with patch("core.loader.MAX_ZIP_ENTRY_COUNT", 2):
                with self.assertRaisesRegex(ValueError, "too many files"):
                    load_document(path)

    def test_load_document_rejects_docx_zip_with_large_entry(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "large-entry.docx"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("[Content_Types].xml", "<Types />")
                archive.writestr("word/document.xml", "A" * 11)

            with patch("core.loader.MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES", 10):
                with self.assertRaisesRegex(ValueError, "too large"):
                    load_document(path)

    def test_load_document_rejects_epub_zip_with_large_entry(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "large-entry.epub"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("mimetype", "application/epub+zip")
                archive.writestr("chapter.xhtml", "A" * 11)

            with patch("core.loader.MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES", 10):
                with self.assertRaisesRegex(ValueError, "too large"):
                    load_document(path)

    def test_load_document_reads_pdf_text(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "sample.pdf"
            pdf = fitz.open()
            page = pdf.new_page()
            page.insert_text((72, 72), "Lumina PDF content.")
            pdf.save(path)
            pdf.close()

            loaded = load_document(path)

            self.assertEqual(loaded.file_type, "PDF")
            self.assertEqual(loaded.total_pages, 1)
            self.assertIn("Lumina PDF content.", loaded.documents[0].page_content)

    def test_load_document_reads_docx_paragraphs_and_tables(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "sample.docx"
            document = docx.Document()
            document.add_paragraph("Lumina DOCX paragraph.")
            table = document.add_table(rows=1, cols=2)
            table.rows[0].cells[0].text = "Column A"
            table.rows[0].cells[1].text = "Column B"
            document.save(path)

            loaded = load_document(path)

            self.assertEqual(loaded.file_type, "DOCX")
            self.assertEqual(loaded.total_pages, 1)
            self.assertIn("Lumina DOCX paragraph.", loaded.documents[0].page_content)
            self.assertIn("Column A | Column B", loaded.documents[0].page_content)

    def test_load_document_reads_epub_content_without_navigation_page(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "sample.epub"
            book = epub.EpubBook()
            book.set_identifier("lumina-smoke")
            book.set_title("Lumina Smoke")
            book.set_language("en")
            chapter = epub.EpubHtml(title="Chapter 1", file_name="chapter.xhtml", lang="en")
            chapter.content = (
                "<html><body><h1>Chapter</h1>"
                "<p>Lumina EPUB content.</p></body></html>"
            )
            book.add_item(chapter)
            book.toc = (chapter,)
            book.spine = ["nav", chapter]
            book.add_item(epub.EpubNcx())
            book.add_item(epub.EpubNav())
            epub.write_epub(str(path), book)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                loaded = load_document(path)

            self.assertEqual(loaded.file_type, "EPUB")
            self.assertEqual(loaded.total_pages, 1)
            self.assertIn("Lumina EPUB content.", loaded.documents[0].page_content)

    def test_load_document_reads_txt_content(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "notes.txt"
            path.write_text("Lumina plain text content.", encoding="utf-8")

            loaded = load_document(path)

            self.assertEqual(loaded.file_type, "TXT")
            self.assertEqual(loaded.total_pages, 1)
            self.assertIn("Lumina plain text content.", loaded.documents[0].page_content)

    def test_load_document_reads_markdown_sections(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "document.md"
            path.write_text(
                "# Bab 1: Pendahuluan\n\nIsi bab satu.\n\n# Bab 2: Metode\n\nIsi bab dua.",
                encoding="utf-8",
            )

            loaded = load_document(path)

            self.assertEqual(loaded.file_type, "MD")
            self.assertEqual(loaded.total_pages, 2)
            self.assertEqual(loaded.documents[0].metadata["section"], "Bab 1: Pendahuluan")
            self.assertIn("Isi bab satu.", loaded.documents[0].page_content)
            self.assertEqual(loaded.documents[1].metadata["section"], "Bab 2: Metode")
            self.assertIn("Isi bab dua.", loaded.documents[1].page_content)

    def test_load_document_rejects_binary_in_txt(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "corrupted.txt"
            path.write_bytes(b"Hello\x00Binary\x00Data")

            with self.assertRaisesRegex(ValueError, "valid TXT document"):
                load_document(path)

    def test_load_document_rejects_encrypted_pdf(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "locked.pdf"
            pdf = fitz.open()
            page = pdf.new_page()
            page.insert_text((72, 72), "Secret content.")
            pdf.save(
                path,
                encryption=fitz.PDF_ENCRYPT_AES_256,
                user_pw="password123",
                owner_pw="owner123",
            )
            pdf.close()

            with self.assertRaisesRegex(ValueError, "dilindungi password"):
                load_document(path)

    def test_load_document_informative_error_for_empty_pdf(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "empty.pdf"
            pdf = fitz.open()
            _ = pdf.new_page()  # blank page without text
            pdf.save(path)
            pdf.close()

            with self.assertRaisesRegex(ValueError, "OCR"):
                load_document(path)

    def test_load_document_maps_pdf_toc_to_sections(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "toc_doc.pdf"
            pdf = fitz.open()
            p1 = pdf.new_page()
            p1.insert_text((72, 72), "Content of chapter one.")
            p2 = pdf.new_page()
            p2.insert_text((72, 72), "Content of chapter two.")
            pdf.set_toc([[1, "Bab 1: Pendahuluan", 1], [1, "Bab 2: Metodologi", 2]])
            pdf.save(path)
            pdf.close()

            loaded = load_document(path)
            self.assertEqual(loaded.total_pages, 2)
            self.assertEqual(loaded.documents[0].metadata["section"], "Bab 1: Pendahuluan")
            self.assertEqual(loaded.documents[1].metadata["section"], "Bab 2: Metodologi")

    def test_load_document_docx_deduplicates_merged_cells(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "merged.docx"
            document = docx.Document()
            t = document.add_table(rows=1, cols=2)
            t.rows[0].cells[0].merge(t.rows[0].cells[1])
            t.rows[0].cells[0].text = "Header Gabungan"
            document.save(path)

            loaded = load_document(path)
            self.assertEqual(loaded.total_pages, 1)
            self.assertEqual(loaded.documents[0].page_content, "Header Gabungan")


if __name__ == "__main__":
    unittest.main()
