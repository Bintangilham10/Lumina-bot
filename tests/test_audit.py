"""Tests for optional audit logging."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from utils.audit import (
    audit_event,
    audit_filename_fields,
    audit_log_path,
    document_text_stats,
    duration_ms,
    estimate_token_count,
)


class AuditTests(unittest.TestCase):
    def test_audit_log_path_returns_none_when_disabled(self) -> None:
        self.assertIsNone(audit_log_path(""))

    def test_audit_event_writes_jsonl_record_without_complex_values(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "audit.jsonl"

            audit_event(
                "document_processed",
                log_path=path,
                filename="sample.pdf",
                total_chunks=3,
                complex_value={"raw": "value"},
            )

            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record["event"], "document_processed")
        self.assertEqual(record["filename"], "sample.pdf")
        self.assertEqual(record["total_chunks"], 3)
        self.assertEqual(record["complex_value"], "{'raw': 'value'}")
        self.assertIn("timestamp", record)

    def test_audit_filename_fields_do_not_expose_raw_filename(self) -> None:
        fields = audit_filename_fields("Bintang private report.pdf")

        self.assertEqual(set(fields), {"filename_hash", "file_extension"})
        self.assertEqual(fields["file_extension"], ".pdf")
        self.assertEqual(len(fields["filename_hash"]), 16)
        serialized_fields = " ".join(fields.values())
        self.assertNotIn("Bintang", serialized_fields)
        self.assertNotIn("private", serialized_fields)

    def test_duration_ms_returns_non_negative_elapsed_time(self) -> None:
        self.assertEqual(duration_ms(1.0, 1.234), 234)
        self.assertEqual(duration_ms(2.0, 1.0), 0)

    def test_estimate_token_count_uses_privacy_safe_text_length(self) -> None:
        self.assertEqual(estimate_token_count(""), 0)
        self.assertEqual(estimate_token_count("abcd"), 1)
        self.assertEqual(estimate_token_count("abcdef"), 2)

    def test_document_text_stats_aggregates_document_sizes(self) -> None:
        stats = document_text_stats(
            [
                Document(page_content="abcd", metadata={}),
                Document(page_content="abcdefgh", metadata={}),
            ]
        )

        self.assertEqual(stats["document_count"], 2)
        self.assertEqual(stats["text_chars"], 12)
        self.assertEqual(stats["estimated_tokens"], 3)


if __name__ == "__main__":
    unittest.main()
