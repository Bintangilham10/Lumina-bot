"""Tests for document splitting utilities."""

from __future__ import annotations

import unittest

from langchain_core.documents import Document

from core.splitter import split_documents


class SplitterTests(unittest.TestCase):
    def test_split_documents_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "No documents"):
            split_documents([])

    def test_split_documents_rejects_invalid_chunk_settings(self) -> None:
        documents = [Document(page_content="Alpha beta", metadata={})]

        with self.assertRaisesRegex(ValueError, "chunk_size"):
            split_documents(documents, chunk_size=0)
        with self.assertRaisesRegex(ValueError, "chunk_overlap"):
            split_documents(documents, chunk_overlap=-1)
        with self.assertRaisesRegex(ValueError, "smaller than chunk_size"):
            split_documents(documents, chunk_size=100, chunk_overlap=100)

    def test_split_documents_adds_chunk_metadata(self) -> None:
        documents = [
            Document(
                page_content="Alpha beta gamma delta epsilon zeta eta theta iota kappa",
                metadata={"source": "sample.pdf"},
            )
        ]

        chunks = split_documents(documents, chunk_size=20, chunk_overlap=5)

        self.assertGreaterEqual(len(chunks), 2)
        for index, chunk in enumerate(chunks, start=1):
            self.assertEqual(chunk.metadata["chunk"], index)
            self.assertEqual(chunk.metadata["total_chunks"], len(chunks))
            self.assertEqual(chunk.metadata["source"], "sample.pdf")


if __name__ == "__main__":
    unittest.main()
