"""Tests for Streamlit app formatting helpers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from app import (
    PROCESSING_STEPS,
    evaluate_auth_attempt_limit,
    evaluate_question_rate_limit,
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

    def test_auth_attempt_limit_blocks_when_window_is_full(self) -> None:
        allowed, timestamps, retry_after = evaluate_auth_attempt_limit(
            [1.0, 2.0],
            now=3.0,
            max_attempts=2,
        )

        self.assertFalse(allowed)
        self.assertEqual(timestamps, [1.0, 2.0])
        self.assertEqual(retry_after, 58)

    def test_question_limit_does_not_consume_session_when_global_blocks(self) -> None:
        with patch("app.check_global_rate_limit", return_value=(False, 42)):
            allowed, timestamps, retry_after, limit_scope = evaluate_question_rate_limit(
                [1.0],
                now=2.0,
                max_questions=2,
                max_global_questions=1,
            )

        self.assertFalse(allowed)
        self.assertEqual(timestamps, [1.0])
        self.assertEqual(retry_after, 42)
        self.assertEqual(limit_scope, "global")

    def test_question_limit_blocks_session_before_global_check(self) -> None:
        with patch("app.check_global_rate_limit") as global_limit:
            allowed, timestamps, retry_after, limit_scope = evaluate_question_rate_limit(
                [1.0, 2.0],
                now=3.0,
                max_questions=2,
                max_global_questions=1,
            )

        global_limit.assert_not_called()
        self.assertFalse(allowed)
        self.assertEqual(timestamps, [1.0, 2.0])
        self.assertEqual(retry_after, 58)
        self.assertEqual(limit_scope, "session")


if __name__ == "__main__":
    unittest.main()
