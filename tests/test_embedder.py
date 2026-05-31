"""Tests for vector store helper utilities."""

from __future__ import annotations

import unittest

from core.embedder import vector_store_document_count


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


if __name__ == "__main__":
    unittest.main()
