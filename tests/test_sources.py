"""Tests for source citation helpers."""

from __future__ import annotations

import unittest

from langchain_core.documents import Document

from utils.sources import (
    build_source_references,
    format_source_context,
    normalize_source_snippet,
    source_metadata,
    source_relevance_score,
)


class SourceHelperTests(unittest.TestCase):
    def test_normalize_source_snippet_truncates_text(self) -> None:
        snippet = normalize_source_snippet("  Alpha\n\n beta   gamma  ", max_length=14)

        self.assertEqual(snippet, "Alpha beta...")

    def test_normalize_source_snippet_returns_empty_for_zero_max_length(self) -> None:
        self.assertEqual(normalize_source_snippet("Hello world", max_length=0), "")

    def test_normalize_source_snippet_handles_max_length_under_four(self) -> None:
        self.assertEqual(normalize_source_snippet("Hello world", max_length=3), "Hel")

    def test_build_source_references_deduplicates_and_numbers_sources(self) -> None:
        documents = [
            Document(
                page_content="First source paragraph.",
                metadata={
                    "filename": "doc.pdf",
                    "page": 2,
                    "section": "Page 2",
                    "relevance_score": 0.88,
                },
            ),
            Document(
                page_content="Duplicate source paragraph.",
                metadata={"filename": "doc.pdf", "page": 2, "section": "Page 2"},
            ),
            Document(
                page_content="Second source paragraph.",
                metadata={"filename": "doc.pdf", "page": 3, "section": "Page 3"},
            ),
        ]

        references = build_source_references(documents)

        self.assertEqual([reference.number for reference in references], [1, 2])
        self.assertEqual(references[0].filename, "doc.pdf")
        self.assertEqual(references[0].page, "2")
        self.assertEqual(references[0].relevance_score, 0.88)
        self.assertIn("First source paragraph.", references[0].snippet)
        self.assertIn("Second source paragraph.", references[1].snippet)

    def test_build_source_references_respects_max_sources(self) -> None:
        documents = [
            Document(page_content=f"Content {i}", metadata={"filename": f"doc{i}.pdf", "page": 1})
            for i in range(5)
        ]
        refs = build_source_references(documents, max_sources=2)
        self.assertEqual(len(refs), 2)

    def test_format_source_context_reuses_number_for_duplicate_source(self) -> None:
        documents = [
            Document(
                page_content="First chunk.",
                metadata={
                    "filename": "private-report.pdf",
                    "page": 2,
                    "section": "Page 2",
                },
            ),
            Document(
                page_content="Second chunk.",
                metadata={
                    "filename": "private-report.pdf",
                    "page": 2,
                    "section": "Page 2",
                },
            ),
        ]

        context = format_source_context(documents)

        self.assertIn("Source [1]: page/section 2\nFirst chunk.", context)
        self.assertIn("Source [1]: page/section 2\nSecond chunk.", context)
        self.assertNotIn("private-report.pdf", context)
        self.assertNotIn("Source [2]", context)

    def test_build_source_references_keeps_highest_relevance_score(self) -> None:
        documents = [
            Document(
                page_content="Lower score chunk on page 1.",
                metadata={"filename": "doc.pdf", "page": 1, "section": "Intro", "relevance_score": 0.65},
            ),
            Document(
                page_content="Higher score chunk on same page 1.",
                metadata={"filename": "doc.pdf", "page": 1, "section": "Intro", "relevance_score": 0.92},
            ),
        ]
        refs = build_source_references(documents)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].relevance_score, 0.92)

    def test_format_source_lines_produces_readable_summaries(self) -> None:
        from utils.sources import format_source_lines
        documents = [
            Document(
                page_content="Important insight regarding economics.",
                metadata={"filename": "report.pdf", "page": 4, "section": "Bab 2", "relevance_score": 0.89},
            )
        ]
        lines = format_source_lines(documents)
        self.assertEqual(len(lines), 1)
        self.assertIn("[1] report.pdf", lines[0])
        self.assertIn("page/section 4", lines[0])
        self.assertIn("Bab 2", lines[0])
        self.assertIn("relevance 0.89", lines[0])
        self.assertIn("Important insight", lines[0])

    def test_source_metadata_falls_back_through_keys(self) -> None:
        doc_with_source = Document(page_content="x", metadata={"source": "/path/to/file.pdf"})
        filename, page, section = source_metadata(doc_with_source)
        self.assertEqual(filename, "file.pdf")
        self.assertEqual(page, "-")
        self.assertEqual(section, "")

        doc_empty = Document(page_content="x", metadata={})
        filename2, _, _ = source_metadata(doc_empty)
        self.assertEqual(filename2, "Document")

    def test_source_relevance_score_handles_non_numeric(self) -> None:
        doc = Document(page_content="x", metadata={"relevance_score": "not-a-number"})
        self.assertIsNone(source_relevance_score(doc))

        doc2 = Document(page_content="x", metadata={"relevance_score": None})
        self.assertIsNone(source_relevance_score(doc2))


if __name__ == "__main__":
    unittest.main()
