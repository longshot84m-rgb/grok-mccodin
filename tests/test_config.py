"""Tests for grok_mccodin.config."""

from __future__ import annotations

import os
from unittest.mock import patch

from grok_mccodin.config import Config


class TestConfig:
    def test_defaults(self):
        cfg = Config()
        assert cfg.grok_api_key == ""
        assert cfg.grok_model == "grok-3"
        assert cfg.safe_lock is False

    def test_has_grok_key(self):
        assert Config(grok_api_key="abc").has_grok_key
        assert not Config().has_grok_key

    def test_has_x_credentials(self):
        cfg = Config(
            x_api_key="a", x_api_secret="b",
            x_access_token="c", x_access_secret="d",
        )
        assert cfg.has_x_credentials
        assert not Config().has_x_credentials

    @patch.dict(os.environ, {"GROK_API_KEY": "env-key", "GROK_MODEL": "grok-3-mini"})
    def test_from_env(self):
        cfg = Config.from_env()
        assert cfg.grok_api_key == "env-key"
        assert cfg.grok_model == "grok-3-mini"
