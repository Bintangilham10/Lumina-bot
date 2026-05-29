"""Tests for chatbot retrieval and streaming helpers."""

from __future__ import annotations

import unittest

from langchain_core.documents import Document

from core.chatbot import format_documents_context, stream_question


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
        self.assertIn("Question: Apa isi dokumen?", llm.prompts[0])

    def test_stream_question_rejects_empty_question(self) -> None:
        qa_chain = FakeQaChain([], FakeLlm())

        with self.assertRaisesRegex(ValueError, "Question cannot be empty"):
            stream_question(qa_chain, " ")

    def test_format_documents_context_joins_page_content(self) -> None:
        context = format_documents_context(
            [
                Document(page_content="Alpha", metadata={}),
                Document(page_content="Beta", metadata={}),
            ]
        )

        self.assertEqual(context, "Alpha\n\nBeta")


if __name__ == "__main__":
    unittest.main()
