"""Tests for Streamlit app formatting helpers."""

from __future__ import annotations

import unittest

from langchain_core.documents import Document

from app import format_source_snippet, format_sources


class AppFormattingTests(unittest.TestCase):
    def test_format_source_snippet_normalizes_and_escapes_text(self) -> None:
        snippet = format_source_snippet("  Alpha\n\n<script>   beta  ")

        self.assertEqual(snippet, "Alpha &lt;script&gt; beta")

    def test_format_source_snippet_truncates_long_text(self) -> None:
        snippet = format_source_snippet("one two three four", max_length=12)

        self.assertEqual(snippet, "one two t...")

    def test_format_sources_includes_snippet_and_deduplicates_metadata(self) -> None:
        documents = [
            Document(
                page_content="First answer paragraph with details.",
                metadata={"filename": "doc.pdf", "page": 2, "section": "Page 2"},
            ),
            Document(
                page_content="Duplicate source should not appear.",
                metadata={"filename": "doc.pdf", "page": 2, "section": "Page 2"},
            ),
        ]

        sources = format_sources(documents)

        self.assertEqual(len(sources), 1)
        self.assertIn("doc.pdf", sources[0])
        self.assertIn("Halaman/bagian: 2", sources[0])
        self.assertIn("First answer paragraph with details.", sources[0])
        self.assertNotIn("Duplicate source should not appear.", sources[0])


if __name__ == "__main__":
    unittest.main()
