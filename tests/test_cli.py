"""Tests for command-line helpers."""

from __future__ import annotations

import argparse
import unittest

from langchain_core.documents import Document

from main import (
    build_parser,
    format_cli_snippet,
    format_cli_sources,
    non_negative_int,
    positive_int,
    relevance_score_value,
    temperature_value,
)


class CliTests(unittest.TestCase):
    def test_argparse_value_helpers_validate_ranges(self) -> None:
        self.assertEqual(positive_int("3"), 3)
        self.assertEqual(non_negative_int("0"), 0)
        self.assertEqual(temperature_value("0.8"), 0.8)
        self.assertEqual(relevance_score_value("0.75"), 0.75)

        with self.assertRaises(argparse.ArgumentTypeError):
            positive_int("0")
        with self.assertRaises(argparse.ArgumentTypeError):
            non_negative_int("-1")
        with self.assertRaises(argparse.ArgumentTypeError):
            temperature_value("1.5")
        with self.assertRaises(argparse.ArgumentTypeError):
            relevance_score_value("-0.1")

    def test_build_parser_exposes_retrieval_options(self) -> None:
        args = build_parser().parse_args(
            [
                "sample.pdf",
                "--chunk-size",
                "1200",
                "--chunk-overlap",
                "150",
                "--retrieval-k",
                "5",
                "--temperature",
                "0",
                "--min-relevance-score",
                "0.65",
                "--hide-sources",
                "--rebuild-index",
                "--max-file-size-mb",
                "25",
                "--max-pages",
                "250",
                "--max-chunks",
                "750",
            ]
        )

        self.assertEqual(args.document, "sample.pdf")
        self.assertEqual(args.chunk_size, 1200)
        self.assertEqual(args.chunk_overlap, 150)
        self.assertEqual(args.retrieval_k, 5)
        self.assertEqual(args.temperature, 0)
        self.assertEqual(args.min_relevance_score, 0.65)
        self.assertTrue(args.hide_sources)
        self.assertTrue(args.rebuild_index)
        self.assertEqual(args.max_file_size_mb, 25)
        self.assertEqual(args.max_pages, 250)
        self.assertEqual(args.max_chunks, 750)

    def test_format_cli_snippet_normalizes_and_truncates_text(self) -> None:
        snippet = format_cli_snippet("  Alpha\n\n beta   gamma  ", max_length=14)

        self.assertEqual(snippet, "Alpha beta...")

    def test_format_cli_sources_deduplicates_metadata(self) -> None:
        documents = [
            Document(
                page_content="First source paragraph with useful details.",
                metadata={
                    "filename": "doc.pdf",
                    "page": 2,
                    "section": "Page 2",
                    "relevance_score": 0.83,
                },
            ),
            Document(
                page_content="Duplicate source should not appear.",
                metadata={"filename": "doc.pdf", "page": 2, "section": "Page 2"},
            ),
        ]

        sources = format_cli_sources(documents)

        self.assertEqual(len(sources), 1)
        self.assertIn("[1] doc.pdf", sources[0])
        self.assertIn("doc.pdf", sources[0])
        self.assertIn("page/section 2", sources[0])
        self.assertIn("relevance 0.83", sources[0])
        self.assertIn("First source paragraph", sources[0])
        self.assertNotIn("Duplicate source", sources[0])


if __name__ == "__main__":
    unittest.main()
