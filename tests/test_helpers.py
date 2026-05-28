"""Tests for shared helper utilities."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from utils.helpers import clean_text, file_sha256, is_supported_file, safe_collection_name


class HelperTests(unittest.TestCase):
    def test_clean_text_normalizes_whitespace(self) -> None:
        text = "  Hello\tworld\n\n\nSecond   line\x00  "

        self.assertEqual(clean_text(text), "Hello world\n\nSecond line")

    def test_clean_text_normalizes_windows_line_endings(self) -> None:
        text = " First line \r\n  Second line\rThird line "

        self.assertEqual(clean_text(text), "First line\nSecond line\nThird line")

    def test_safe_collection_name_normalizes_text_parts(self) -> None:
        name = safe_collection_name(["Lumina Doc", "File #1", "!!!"])

        self.assertEqual(name, "lumina-doc-file-1")

    def test_safe_collection_name_enforces_minimum_length(self) -> None:
        self.assertEqual(safe_collection_name(["x"]), "lumina-x")

    def test_supported_file_detection_is_case_insensitive(self) -> None:
        self.assertTrue(is_supported_file("Document.PDF"))
        self.assertTrue(is_supported_file("Document.Docx"))
        self.assertTrue(is_supported_file("Document.EPUB"))
        self.assertFalse(is_supported_file("Document.txt"))

    def test_file_sha256_hashes_file_content(self) -> None:
        content = b"Lumina Doc hash test"
        expected_hash = hashlib.sha256(content).hexdigest()

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "sample.txt"
            path.write_bytes(content)

            self.assertEqual(file_sha256(path), expected_hash)


if __name__ == "__main__":
    unittest.main()
