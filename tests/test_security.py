"""Tests for production security helpers."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from utils.security import (
    active_rate_limit_timestamps,
    check_global_rate_limit,
    check_rate_limit,
    configured_model_options,
    int_from_env,
    verify_password,
)


class SecurityHelperTests(unittest.TestCase):
    def test_verify_password_uses_expected_value(self) -> None:
        self.assertTrue(verify_password("secret", "secret"))
        self.assertFalse(verify_password("wrong", "secret"))
        self.assertTrue(verify_password("anything", ""))

    def test_int_from_env_uses_fallback_for_missing_or_invalid_values(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(int_from_env("LUMINA_TEST_INT", 7), 7)
        with patch.dict(os.environ, {"LUMINA_TEST_INT": "nope"}, clear=True):
            self.assertEqual(int_from_env("LUMINA_TEST_INT", 7), 7)
        with patch.dict(os.environ, {"LUMINA_TEST_INT": "-5"}, clear=True):
            self.assertEqual(int_from_env("LUMINA_TEST_INT", 7, minimum=0), 0)

    def test_configured_model_options_deduplicates_and_includes_default(self) -> None:
        with patch.dict(
            os.environ,
            {"LUMINA_MODELS": "model-a, model-b, model-a"},
            clear=True,
        ):
            self.assertEqual(
                configured_model_options("LUMINA_MODELS", "default-model"),
                ["default-model", "model-a", "model-b"],
            )

    def test_check_rate_limit_blocks_when_window_is_full(self) -> None:
        allowed, timestamps, retry_after = check_rate_limit(
            [1.0, 2.0],
            now=3.0,
            max_events=2,
            window_seconds=60,
        )

        self.assertFalse(allowed)
        self.assertEqual(timestamps, [1.0, 2.0])
        self.assertEqual(retry_after, 58)

    def test_check_rate_limit_allows_when_disabled(self) -> None:
        allowed, timestamps, retry_after = check_rate_limit(
            [1.0, 2.0],
            now=3.0,
            max_events=0,
            window_seconds=60,
        )

        self.assertTrue(allowed)
        self.assertEqual(timestamps, [])
        self.assertEqual(retry_after, 0)

    def test_active_rate_limit_timestamps_prunes_expired_entries(self) -> None:
        self.assertEqual(
            active_rate_limit_timestamps([1.0, 2.0, 70.0], now=70.0, window_seconds=60),
            [70.0],
        )

    def test_check_global_rate_limit_blocks_across_calls(self) -> None:
        with patch("utils.security._global_question_timestamps", []):
            allowed, retry_after = check_global_rate_limit(
                now=1.0,
                max_events=1,
                window_seconds=60,
            )
            self.assertTrue(allowed)
            self.assertEqual(retry_after, 0)

            allowed, retry_after = check_global_rate_limit(
                now=2.0,
                max_events=1,
                window_seconds=60,
            )
            self.assertFalse(allowed)
            self.assertEqual(retry_after, 59)

    def test_check_global_rate_limit_allows_when_disabled(self) -> None:
        with patch("utils.security._global_question_timestamps", [1.0, 2.0]):
            allowed, retry_after = check_global_rate_limit(
                now=3.0,
                max_events=0,
                window_seconds=60,
            )

        self.assertTrue(allowed)
        self.assertEqual(retry_after, 0)


if __name__ == "__main__":
    unittest.main()
