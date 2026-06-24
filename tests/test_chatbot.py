"""Tests for chatbot retrieval and streaming helpers."""

from __future__ import annotations

import unittest

from langchain_core.documents import Document

from core.chatbot import (
    DocumentQaChain,
    ask_question,
    format_documents_context,
    stream_question,
)


class FakeChunk:
    def __init__(self, content) -> None:
        self.content = content


class FakeRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.queries: list[str] = []

    def invoke(self, question: str) -> list[Document]:
        self.queries.append(question)
        return self.documents


class FakeLlm:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def stream(self, prompt: str):
        self.prompts.append(prompt)
        yield FakeChunk("Alpha ")
        yield FakeChunk([{"text": "beta"}])

    def invoke(self, prompt: str) -> FakeChunk:
        self.prompts.append(prompt)
        return FakeChunk("Alpha beta [1]")


class FakeLlmChain:
    def __init__(self, llm: FakeLlm) -> None:
        self.llm = llm


class FakeCombineDocumentsChain:
    def __init__(self, llm: FakeLlm) -> None:
        self.llm_chain = FakeLlmChain(llm)


class FakeQaChain:
    def __init__(self, documents: list[Document], llm: FakeLlm) -> None:
        self.retriever = FakeRetriever(documents)
        self.combine_documents_chain = FakeCombineDocumentsChain(llm)


class FakeScoredVectorStore:
    def __init__(self, results) -> None:
        self.results = results
        self.queries: list[tuple[str, int]] = []

    def similarity_search_with_relevance_scores(self, question: str, k: int):
        self.queries.append((question, k))
        return self.results


class ChatbotStreamingTests(unittest.TestCase):
    def test_stream_question_streams_answer_and_returns_sources(self) -> None:
        documents = [
            Document(page_content="First source paragraph.", metadata={"page": 1}),
            Document(page_content="Second source paragraph.", metadata={"page": 2}),
        ]
        llm = FakeLlm()
        qa_chain = FakeQaChain(documents, llm)

        answer_stream, sources = stream_question(qa_chain, "  Apa isi dokumen?  ")
        answer = "".join(answer_stream)

        self.assertEqual(answer, "Alpha beta")
        self.assertEqual(sources, documents)
        self.assertEqual(qa_chain.retriever.queries, ["Apa isi dokumen?"])
        self.assertIn("First source paragraph.", llm.prompts[0])
        self.assertIn("Second source paragraph.", llm.prompts[0])
        self.assertIn("Source [1]", llm.prompts[0])
        self.assertIn("Source [2]", llm.prompts[0])
        self.assertIn("Question: Apa isi dokumen?", llm.prompts[0])

    def test_ask_question_uses_cited_context(self) -> None:
        documents = [
            Document(
                page_content="First source paragraph.",
                metadata={"filename": "doc.pdf", "page": 1},
            )
        ]
        llm = FakeLlm()
        qa_chain = FakeQaChain(documents, llm)

        response = ask_question(qa_chain, "Apa isi dokumen?")

        self.assertEqual(response["result"], "Alpha beta [1]")
        self.assertEqual(response["source_documents"], documents)
        self.assertIn("Source [1]: page/section 1", llm.prompts[0])
        self.assertNotIn("doc.pdf", llm.prompts[0])

    def test_ask_question_filters_sources_by_relevance_score(self) -> None:
        strong = Document(
            page_content="Strong source paragraph.",
            metadata={"filename": "doc.pdf", "page": 1},
        )
        weak = Document(
            page_content="Weak source paragraph.",
            metadata={"filename": "doc.pdf", "page": 2},
        )
        vector_store = FakeScoredVectorStore([(strong, 0.91), (weak, 0.42)])
        llm = FakeLlm()
        qa_chain = DocumentQaChain(
            retriever=FakeRetriever([]),
            llm=llm,  # type: ignore[arg-type]
            vector_store=vector_store,  # type: ignore[arg-type]
            retrieval_k=2,
            min_relevance_score=0.7,
        )

        response = ask_question(qa_chain, "What is covered?")

        self.assertEqual(response["result"], "Alpha beta [1]")
        self.assertEqual(vector_store.queries, [("What is covered?", 2)])
        self.assertEqual(len(response["source_documents"]), 1)
        self.assertEqual(
            response["source_documents"][0].metadata["relevance_score"],
            0.91,
        )
        self.assertIn("Strong source paragraph.", llm.prompts[0])
        self.assertNotIn("Weak source paragraph.", llm.prompts[0])

    def test_ask_question_returns_not_found_when_no_source_passes_threshold(self) -> None:
        document = Document(
            page_content="Weak source paragraph.",
            metadata={"filename": "doc.pdf", "page": 1},
        )
        vector_store = FakeScoredVectorStore([(document, 0.2)])
        llm = FakeLlm()
        qa_chain = DocumentQaChain(
            retriever=FakeRetriever([]),
            llm=llm,  # type: ignore[arg-type]
            vector_store=vector_store,  # type: ignore[arg-type]
            retrieval_k=1,
            min_relevance_score=0.8,
        )

        response = ask_question(qa_chain, "Apa isi rahasianya?")

        self.assertEqual(
            response["result"],
            "Informasi tersebut tidak ditemukan di dokumen.",
        )
        self.assertEqual(response["source_documents"], [])
        self.assertEqual(llm.prompts, [])

    def test_stream_question_rejects_empty_question(self) -> None:
        qa_chain = FakeQaChain([], FakeLlm())

        with self.assertRaisesRegex(ValueError, "Question cannot be empty"):
            stream_question(qa_chain, " ")

    def test_format_documents_context_adds_source_labels(self) -> None:
        context = format_documents_context(
            [
                Document(page_content="Alpha", metadata={"filename": "doc.pdf", "page": 1}),
                Document(page_content="Beta", metadata={"filename": "doc.pdf", "page": 2}),
            ]
        )

        self.assertEqual(
            context,
            (
                "Source [1]: page/section 1\nAlpha\n\n"
                "Source [2]: page/section 2\nBeta"
            ),
        )
        self.assertNotIn("doc.pdf", context)


if __name__ == "__main__":
    unittest.main()
