"""Optional audit logging for production deployments."""

from __future__ import annotations

import json
import os
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
