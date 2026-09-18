"""Tests for vector store helper utilities."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from core.embedder import create_vector_store, vector_store_document_count


class FakeDoc:
    def __init__(self, page_content: str) -> None:
        self.page_content = page_content


class FakeCollection:
    def __init__(self, count: int) -> None:
        self._count = count

    def count(self) -> int:
        return self._count


class FakeStoreWithCollection:
    def __init__(self, count: int) -> None:
        self._collection = FakeCollection(count)


class FakeStoreWithGet:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids
        self.include: list[str] | None = None

    def get(self, include: list[str]) -> dict[str, list[str]]:
        self.include = include
        return {"ids": self.ids}


class EmbedderHelperTests(unittest.TestCase):
    def test_vector_store_document_count_prefers_collection_count(self) -> None:
        self.assertEqual(vector_store_document_count(FakeStoreWithCollection(7)), 7)

    def test_vector_store_document_count_falls_back_to_get(self) -> None:
        store = FakeStoreWithGet(["a", "b"])

        self.assertEqual(vector_store_document_count(store), 2)
        self.assertEqual(store.include, [])

    @patch("core.embedder.create_embeddings")
    @patch("core.embedder.Chroma")
    def test_create_vector_store_batches_documents(self, mock_chroma, mock_embeddings) -> None:
        mock_store = MagicMock()
        mock_chroma.from_documents.return_value = mock_store
        chunks = [FakeDoc(f"Chunk {i}") for i in range(15)]

        store = create_vector_store(chunks, "test-col", persist_directory=None, batch_size=5)

        self.assertEqual(store, mock_store)
        mock_chroma.from_documents.assert_called_once()
        self.assertEqual(mock_store.add_documents.call_count, 2)

    def test_create_vector_store_rejects_empty_chunks(self) -> None:
        with self.assertRaisesRegex(ValueError, "No chunks were provided"):
            create_vector_store([], "test-col", persist_directory=None)


if __name__ == "__main__":
    unittest.main()

