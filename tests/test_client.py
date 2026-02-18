"""Tests for grok_mccodin.client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from grok_mccodin.client import GrokAPIError, GrokClient


class TestGrokClient:
    def test_build_messages(self, config):
        client = GrokClient(config)
        history = [{"role": "user", "content": "hi"}]
        msgs = client.build_messages(history, "do something", context="project files")

        assert msgs[0]["role"] == "system"
        assert "Grok McCodin" in msgs[0]["content"]
        # Context should be injected
        assert any("project files" in m["content"] for m in msgs)
        # History + new user message
        assert msgs[-1]["content"] == "do something"

    def test_build_messages_no_context(self, config):
        client = GrokClient(config)
        msgs = client.build_messages([], "hello")
        # Only system + user
        assert len(msgs) == 2

    @patch("grok_mccodin.client.requests.Session.post")
    def test_chat_success(self, mock_post, config):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": [{"message": {"content": "Hello from Grok!"}}]}
        mock_post.return_value = mock_resp

        client = GrokClient(config)
        reply = client.chat([{"role": "user", "content": "hi"}])
        assert reply == "Hello from Grok!"

    @patch("grok_mccodin.client.requests.Session.post")
    def test_chat_api_error(self, mock_post, config):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "rate limited"
        mock_post.return_value = mock_resp

        client = GrokClient(config)
        with pytest.raises(GrokAPIError) as exc_info:
            client.chat([{"role": "user", "content": "hi"}])
        assert exc_info.value.status_code == 429


class TestGrokAPIError:
    def test_message(self):
        err = GrokAPIError(500, "internal error")
        assert "500" in str(err)
        assert "internal error" in str(err)
