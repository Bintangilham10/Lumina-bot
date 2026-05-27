"""Tests for document loading validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.loader import load_document


class LoaderTests(unittest.TestCase):
    def test_load_document_rejects_missing_file(self) -> None:
        missing_path = Path.cwd() / "missing-document.pdf"

        with self.assertRaisesRegex(FileNotFoundError, "Document not found"):
            load_document(missing_path)

    def test_load_document_rejects_unsupported_file_type(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "notes.txt"
            path.write_text("Hello world", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unsupported file type"):
                load_document(path)


if __name__ == "__main__":
    unittest.main()
