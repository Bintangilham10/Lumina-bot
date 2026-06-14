"""Optional audit logging for production deployments."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.helpers import ensure_directory


AUDIT_LOG_ENV_VAR = "LUMINA_AUDIT_LOG_PATH"


def audit_log_path(log_path: str | Path | None = None) -> Path | None:
    """Resolve the configured audit log path, returning None when disabled."""
    configured_path = str(log_path or os.getenv(AUDIT_LOG_ENV_VAR, "")).strip()
    if not configured_path:
        return None
    return Path(configured_path).expanduser()


def audit_event(
    event: str,
    log_path: str | Path | None = None,
    **fields: Any,
) -> None:
    """Append a privacy-conscious JSONL audit event when logging is enabled."""
    path = audit_log_path(log_path)
    if path is None:
        return

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **{key: _safe_value(value) for key, value in fields.items()},
    }

    try:
        ensure_directory(path.parent)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError:
        return


def _safe_value(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def duration_ms(started_at: float, ended_at: float | None = None) -> int:
    """Return elapsed milliseconds from a monotonic start timestamp."""
    finished_at = time.perf_counter() if ended_at is None else ended_at
    return max(0, int(round((finished_at - started_at) * 1000)))


def estimate_token_count(text: str) -> int:
    """Estimate token count without storing text or calling a tokenizer."""
    normalized = " ".join(str(text).split())
    if not normalized:
        return 0
    return max(1, (len(normalized) + 3) // 4)


def document_text_stats(documents: Iterable[Any]) -> dict[str, int]:
    """Return privacy-safe aggregate text metrics for LangChain documents."""
    total_chars = 0
    total_estimated_tokens = 0
    document_count = 0

    for document in documents:
        text = str(getattr(document, "page_content", ""))
        total_chars += len(text)
        total_estimated_tokens += estimate_token_count(text)
        document_count += 1

    return {
        "document_count": document_count,
        "text_chars": total_chars,
        "estimated_tokens": total_estimated_tokens,
    }
