"""Tests for model configuration helpers."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from core.chatbot import DEFAULT_CHAT_MODEL, resolve_chat_model
from core.embedder import DEFAULT_EMBEDDING_MODEL, resolve_embedding_model


class ModelConfigTests(unittest.TestCase):
    def test_chat_model_uses_default_without_override(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_chat_model(), DEFAULT_CHAT_MODEL)

    def test_chat_model_prefers_explicit_value(self) -> None:
        with patch.dict(os.environ, {"GEMINI_CHAT_MODEL": "env-model"}, clear=True):
            self.assertEqual(resolve_chat_model("explicit-model"), "explicit-model")

    def test_chat_model_uses_environment_override(self) -> None:
        with patch.dict(os.environ, {"GEMINI_CHAT_MODEL": " env-model "}, clear=True):
            self.assertEqual(resolve_chat_model(), "env-model")

    def test_embedding_model_uses_default_without_override(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_embedding_model(), DEFAULT_EMBEDDING_MODEL)

    def test_embedding_model_prefers_explicit_value(self) -> None:
        with patch.dict(os.environ, {"GEMINI_EMBEDDING_MODEL": "env-model"}, clear=True):
            self.assertEqual(resolve_embedding_model("explicit-model"), "explicit-model")

    def test_embedding_model_uses_environment_override(self) -> None:
        with patch.dict(os.environ, {"GEMINI_EMBEDDING_MODEL": " env-model "}, clear=True):
            self.assertEqual(resolve_embedding_model(), "env-model")


if __name__ == "__main__":
    unittest.main()
