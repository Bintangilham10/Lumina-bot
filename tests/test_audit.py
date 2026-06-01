"""Tests for optional audit logging."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from utils.audit import audit_event, audit_log_path


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


if __name__ == "__main__":
    unittest.main()
