"""Optional live Gemini and Chroma smoke test.

Run with LUMINA_LIVE_TEST=1 and GOOGLE_API_KEY set.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from core.chatbot import ask_question, create_qa_chain
from core.embedder import create_vector_store
from core.splitter import split_documents
from utils.helpers import document_collection_name


LIVE_TEST_ENABLED = os.getenv("LUMINA_LIVE_TEST") == "1" and bool(
    os.getenv("GOOGLE_API_KEY")
)


@unittest.skipUnless(
    LIVE_TEST_ENABLED,
    "Set LUMINA_LIVE_TEST=1 and GOOGLE_API_KEY to run live Gemini smoke tests.",
)
class LiveSmokeTests(unittest.TestCase):
    def test_small_document_rag_pipeline_answers_with_source(self) -> None:
        documents = [
            Document(
                page_content="Lumina Doc smoke test answer: the project codename is Aurora.",
                metadata={"filename": "smoke.txt", "page": 1, "section": "Smoke"},
            )
        ]
        chunks = split_documents(documents, chunk_size=160, chunk_overlap=20)

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            vector_store = create_vector_store(
                chunks=chunks,
                collection_name=document_collection_name("smoke", "0" * 64, 160, 20),
                persist_directory=temp_dir,
            )
            qa_chain = create_qa_chain(vector_store, k=1, temperature=0)

            response = ask_question(qa_chain, "What is the project codename?")

        self.assertIn("Aurora", response["result"])
        self.assertEqual(response["source_documents"], chunks)


if __name__ == "__main__":
    unittest.main()
