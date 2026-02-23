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

    @patch("grok_mccodin.client.requests.Session.request")
    def test_chat_success(self, mock_request, config):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": [{"message": {"content": "Hello from Grok!"}}]}
        mock_request.return_value = mock_resp

        client = GrokClient(config)
        reply = client.chat([{"role": "user", "content": "hi"}])
        assert reply == "Hello from Grok!"

    @patch("grok_mccodin.client.requests.Session.request")
    def test_chat_api_error(self, mock_request, config):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "unauthorized"
        mock_request.return_value = mock_resp

        client = GrokClient(config)
        with pytest.raises(GrokAPIError) as exc_info:
            client.chat([{"role": "user", "content": "hi"}])
        assert exc_info.value.status_code == 401


class TestGrokAPIError:
    def test_message(self):
        err = GrokAPIError(500, "internal error")
        assert "500" in str(err)
        assert "internal error" in str(err)


class TestGrokClientStream:
    @patch("grok_mccodin.client.requests.Session.request")
    def test_chat_stream_success(self, mock_request, config):
        """Test that chat_stream yields content tokens from SSE chunks."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(
            [
                'data: {"choices":[{"delta":{"role":"assistant"}}]}',
                'data: {"choices":[{"delta":{"content":"Hello"}}]}',
                'data: {"choices":[{"delta":{"content":" World"}}]}',
                'data: {"choices":[{"delta":{"content":"!"}}]}',
                "data: [DONE]",
            ]
        )
        mock_request.return_value = mock_resp

        client = GrokClient(config)
        chunks = list(client.chat_stream([{"role": "user", "content": "hi"}]))
        assert chunks == ["Hello", " World", "!"]

    @patch("grok_mccodin.client.requests.Session.request")
    def test_chat_stream_api_error(self, mock_request, config):
        """Test that chat_stream raises GrokAPIError on non-200 status."""
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "unauthorized"
        mock_request.return_value = mock_resp

        client = GrokClient(config)
        with pytest.raises(GrokAPIError) as exc_info:
            list(client.chat_stream([{"role": "user", "content": "hi"}]))
        assert exc_info.value.status_code == 401

    @patch("grok_mccodin.client.requests.Session.request")
    def test_chat_stream_empty_deltas(self, mock_request, config):
        """Test that empty deltas are skipped gracefully."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(
            [
                'data: {"choices":[{"delta":{}}]}',
                'data: {"choices":[{"delta":{"content":"ok"}}]}',
                "data: [DONE]",
            ]
        )
        mock_request.return_value = mock_resp

        client = GrokClient(config)
        chunks = list(client.chat_stream([{"role": "user", "content": "hi"}]))
        assert chunks == ["ok"]

    @patch("grok_mccodin.client.requests.Session.request")
    def test_chat_stream_malformed_json(self, mock_request, config):
        """Test that malformed JSON lines are skipped without error."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(
            [
                "data: {invalid json",
                'data: {"choices":[{"delta":{"content":"ok"}}]}',
                "data: [DONE]",
            ]
        )
        mock_request.return_value = mock_resp

        client = GrokClient(config)
        chunks = list(client.chat_stream([{"role": "user", "content": "hi"}]))
        assert chunks == ["ok"]


class TestRetryLogic:
    @patch("grok_mccodin.client.time.sleep")
    @patch("grok_mccodin.client.requests.Session.request")
    def test_retries_on_429(self, mock_request, mock_sleep, config):
        """Test that 429 triggers retry and eventually succeeds."""
        rate_limit_resp = MagicMock()
        rate_limit_resp.status_code = 429
        rate_limit_resp.text = "rate limited"
        rate_limit_resp.headers = {}

        success_resp = MagicMock()
        success_resp.status_code = 200
        success_resp.json.return_value = {"choices": [{"message": {"content": "Hello!"}}]}

        mock_request.side_effect = [rate_limit_resp, success_resp]

        client = GrokClient(config)
        reply = client.chat([{"role": "user", "content": "hi"}])

        assert reply == "Hello!"
        assert mock_request.call_count == 2
        mock_sleep.assert_called_once()

    @patch("grok_mccodin.client.time.sleep")
    @patch("grok_mccodin.client.requests.Session.request")
    def test_retries_on_500(self, mock_request, mock_sleep, config):
        """Test that 500 triggers retry."""
        error_resp = MagicMock()
        error_resp.status_code = 500
        error_resp.text = "internal error"
        error_resp.headers = {}

        success_resp = MagicMock()
        success_resp.status_code = 200
        success_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

        mock_request.side_effect = [error_resp, success_resp]

        client = GrokClient(config)
        reply = client.chat([{"role": "user", "content": "hi"}])
        assert reply == "ok"

    @patch("grok_mccodin.client.time.sleep")
    @patch("grok_mccodin.client.requests.Session.request")
    def test_exhausts_retries(self, mock_request, mock_sleep, config):
        """Test that after max retries, the error is raised."""
        error_resp = MagicMock()
        error_resp.status_code = 503
        error_resp.text = "service unavailable"
        error_resp.headers = {}

        # 4 calls: initial + 3 retries
        mock_request.return_value = error_resp

        client = GrokClient(config)
        with pytest.raises(GrokAPIError) as exc_info:
            client.chat([{"role": "user", "content": "hi"}])
        assert exc_info.value.status_code == 503
        assert mock_request.call_count == 4  # 1 + 3 retries
        assert mock_sleep.call_count == 3

    @patch("grok_mccodin.client.time.sleep")
    @patch("grok_mccodin.client.requests.Session.request")
    def test_respects_retry_after_header(self, mock_request, mock_sleep, config):
        """Test that Retry-After header value is used for delay."""
        rate_limit_resp = MagicMock()
        rate_limit_resp.status_code = 429
        rate_limit_resp.text = "rate limited"
        rate_limit_resp.headers = {"Retry-After": "5"}

        success_resp = MagicMock()
        success_resp.status_code = 200
        success_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

        mock_request.side_effect = [rate_limit_resp, success_resp]

        client = GrokClient(config)
        client.chat([{"role": "user", "content": "hi"}])

        # Retry-After=5 > base_delay=1, so delay should be 5.0
        mock_sleep.assert_called_once_with(5.0)

    @patch("grok_mccodin.client.time.sleep")
    @patch("grok_mccodin.client.requests.Session.request")
    def test_no_retry_on_400(self, mock_request, mock_sleep, config):
        """Test that 400 (non-retryable) is NOT retried."""
        error_resp = MagicMock()
        error_resp.status_code = 400
        error_resp.text = "bad request"

        mock_request.return_value = error_resp

        client = GrokClient(config)
        with pytest.raises(GrokAPIError) as exc_info:
            client.chat([{"role": "user", "content": "hi"}])
        assert exc_info.value.status_code == 400
        assert mock_request.call_count == 1
        mock_sleep.assert_not_called()

    @patch("grok_mccodin.client.time.sleep")
    @patch("grok_mccodin.client.requests.Session.request")
    def test_exponential_backoff_delays(self, mock_request, mock_sleep, config):
        """Test that delays follow exponential backoff: 1s, 2s, 4s."""
        error_resp = MagicMock()
        error_resp.status_code = 502
        error_resp.text = "bad gateway"
        error_resp.headers = {}

        mock_request.return_value = error_resp

        client = GrokClient(config)
        with pytest.raises(GrokAPIError):
            client.chat([{"role": "user", "content": "hi"}])

        # Should have slept 3 times with exponential delays
        sleep_calls = [call.args[0] for call in mock_sleep.call_args_list]
        assert sleep_calls == [1.0, 2.0, 4.0]
