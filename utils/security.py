"""Security helpers for Streamlit production hardening."""

from __future__ import annotations

import hmac
import os
from threading import Lock

from dotenv import load_dotenv


APP_PASSWORD_ENV_VAR = "LUMINA_APP_PASSWORD"
ALLOWED_CHAT_MODELS_ENV_VAR = "LUMINA_ALLOWED_CHAT_MODELS"
ALLOWED_EMBEDDING_MODELS_ENV_VAR = "LUMINA_ALLOWED_EMBEDDING_MODELS"
MAX_QUESTIONS_PER_MINUTE_ENV_VAR = "LUMINA_MAX_QUESTIONS_PER_MINUTE"
MAX_GLOBAL_QUESTIONS_PER_MINUTE_ENV_VAR = "LUMINA_MAX_GLOBAL_QUESTIONS_PER_MINUTE"
MAX_AUTH_ATTEMPTS_PER_MINUTE_ENV_VAR = "LUMINA_MAX_AUTH_ATTEMPTS_PER_MINUTE"
DEFAULT_MAX_QUESTIONS_PER_MINUTE = 20
DEFAULT_MAX_GLOBAL_QUESTIONS_PER_MINUTE = 120
DEFAULT_MAX_AUTH_ATTEMPTS_PER_MINUTE = 5
RATE_LIMIT_WINDOW_SECONDS = 60
_global_question_timestamps: list[float] = []
_global_rate_limit_lock = Lock()


def configured_password() -> str:
    """Return the optional app password configured by the operator."""
    load_dotenv()
    return os.getenv(APP_PASSWORD_ENV_VAR, "").strip()


def verify_password(candidate: str, expected: str) -> bool:
    """Compare passwords using constant-time comparison."""
    if not expected:
        return True
    return hmac.compare_digest(candidate, expected)


def int_from_env(name: str, default: int, minimum: int = 0) -> int:
    """Read an integer environment variable with a floor and fallback."""
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return max(minimum, value)


def configured_model_options(env_var: str, default_model: str) -> list[str]:
    """Return a de-duplicated allowlist of model options for production UI."""
    raw_value = os.getenv(env_var, "")
    options = [value.strip() for value in raw_value.split(",") if value.strip()]
    if default_model not in options:
        options.insert(0, default_model)

    deduplicated: list[str] = []
    for option in options:
        if option not in deduplicated:
            deduplicated.append(option)
    return deduplicated


def check_rate_limit(
    timestamps: list[float],
    now: float,
    max_events: int,
    window_seconds: int = RATE_LIMIT_WINDOW_SECONDS,
) -> tuple[bool, list[float], int]:
    """Return whether a request is allowed, updated timestamps, and retry delay."""
    if max_events <= 0 or window_seconds <= 0:
        return True, [], 0

    recent = active_rate_limit_timestamps(timestamps, now, window_seconds)
    if len(recent) >= max_events:
        oldest = min(recent)
        retry_after = max(1, int(window_seconds - (now - oldest)))
        return False, recent, retry_after

    recent.append(now)
    return True, recent, 0


def active_rate_limit_timestamps(
    timestamps: list[float],
    now: float,
    window_seconds: int = RATE_LIMIT_WINDOW_SECONDS,
) -> list[float]:
    """Return timestamps still active in a rate-limit window."""
    if window_seconds <= 0:
        return []
    return [
        timestamp for timestamp in timestamps if now - timestamp < window_seconds
    ]


def check_global_rate_limit(
    now: float,
    max_events: int,
    window_seconds: int = RATE_LIMIT_WINDOW_SECONDS,
) -> tuple[bool, int]:
    """Apply a process-wide question rate limit across Streamlit sessions."""
    global _global_question_timestamps

    with _global_rate_limit_lock:
        allowed, timestamps, retry_after = check_rate_limit(
            list(_global_question_timestamps),
            now,
            max_events,
            window_seconds,
        )
        _global_question_timestamps = timestamps
        return allowed, retry_after
