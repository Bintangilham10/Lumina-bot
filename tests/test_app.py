"""Tests for Streamlit app formatting helpers."""

from __future__ import annotations

import unittest

from langchain_core.documents import Document

from app import (
    PROCESSING_STEPS,
    format_source_snippet,
    format_sources,
    render_processing_step,
    user_safe_error_message,
)


class FakeStatus:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def write(self, message: str) -> None:
        self.messages.append(message)


class FakeProgress:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def progress(self, value: int, text: str) -> None:
        self.calls.append((value, text))


class AppFormattingTests(unittest.TestCase):
    def test_processing_steps_are_ordered_and_complete(self) -> None:
        percentages = [percent for percent, _ in PROCESSING_STEPS]

        self.assertEqual(len(PROCESSING_STEPS), 5)
        self.assertEqual(percentages, sorted(percentages))
        self.assertGreater(percentages[0], 0)
        self.assertLess(percentages[-1], 100)

    def test_render_processing_step_updates_status_and_progress(self) -> None:
        status = FakeStatus()
        progress = FakeProgress()

        render_processing_step(status, progress, 1)

        percent, label = PROCESSING_STEPS[1]
        self.assertEqual(status.messages, [label])
        self.assertEqual(progress.calls, [(percent, label)])

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
                metadata={
                    "filename": "doc.pdf",
                    "page": 2,
                    "section": "Page 2",
                    "relevance_score": 0.76,
                },
            ),
            Document(
                page_content="Duplicate source should not appear.",
                metadata={"filename": "doc.pdf", "page": 2, "section": "Page 2"},
            ),
        ]

        sources = format_sources(documents)

        self.assertEqual(len(sources), 1)
        self.assertIn("[1] doc.pdf", sources[0])
        self.assertIn("doc.pdf", sources[0])
        self.assertIn("Halaman/bagian: 2", sources[0])
        self.assertIn("Relevansi: 0.76", sources[0])
        self.assertIn("First answer paragraph with details.", sources[0])
        self.assertNotIn("Duplicate source should not appear.", sources[0])

    def test_user_safe_error_messages_do_not_expose_internal_details(self) -> None:
        sensitive_detail = "C:\\secrets\\api-key.txt"

        document_message = user_safe_error_message("document_processing")
        answer_message = user_safe_error_message("question_answering")
        fallback_message = user_safe_error_message("unknown")

        self.assertNotIn(sensitive_detail, document_message)
        self.assertNotIn(sensitive_detail, answer_message)
        self.assertNotIn(sensitive_detail, fallback_message)
        self.assertNotIn("Traceback", document_message)
        self.assertNotIn("Traceback", answer_message)
        self.assertIn("Gagal memproses dokumen", document_message)
        self.assertIn("Gagal menjawab pertanyaan", answer_message)


if __name__ == "__main__":
    unittest.main()
