"""Tests for grok_mccodin.web."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from grok_mccodin.web import (
    _parse_ddg_fallback,
    _parse_ddg_lite,
    html_to_text,
    web_fetch,
    web_search,
)


class TestHtmlToText:
    def test_basic_html(self):
        html = "<p>Hello <b>world</b></p>"
        text = html_to_text(html)
        assert "Hello" in text
        assert "world" in text

    def test_strips_scripts(self):
        html = "<p>Visible</p><script>var x = 1;</script><p>Also visible</p>"
        text = html_to_text(html)
        assert "Visible" in text
        assert "Also visible" in text
        assert "var x" not in text

    def test_strips_style(self):
        html = "<style>.foo { color: red; }</style><p>Content</p>"
        text = html_to_text(html)
        assert "Content" in text
        assert "color" not in text

    def test_empty_html(self):
        assert html_to_text("") == ""

    def test_line_breaks(self):
        html = "<p>Line 1</p><p>Line 2</p>"
        text = html_to_text(html)
        assert "Line 1" in text
        assert "Line 2" in text


class TestParseDDGLite:
    def test_no_results(self):
        assert _parse_ddg_lite("<html><body>Nothing</body></html>", 5) == []

    def test_fallback_extracts_links(self):
        html = '<a href="https://example.com/page">Example Page</a>'
        results = _parse_ddg_fallback(html, 5)
        assert len(results) == 1
        assert results[0]["url"] == "https://example.com/page"
        assert results[0]["title"] == "Example Page"

    def test_skips_ddg_internal_links(self):
        html = '<a href="https://duckduckgo.com/about">About DDG</a>'
        results = _parse_ddg_fallback(html, 5)
        assert len(results) == 0


class TestWebSearch:
    @patch("grok_mccodin.web._session.post")
    def test_search_returns_results(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '<a href="https://example.com/result">Test Result</a>'
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        results = web_search("test query")
        assert isinstance(results, list)

    @patch("grok_mccodin.web._session.post")
    def test_search_handles_error(self, mock_post):
        import requests

        mock_post.side_effect = requests.RequestException("connection failed")
        results = web_search("test")
        assert results == []


class TestWebFetch:
    @patch("grok_mccodin.web._session.get")
    def test_fetch_html(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.text = "<html><title>Test</title><body><p>Hello</p></body></html>"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = web_fetch("https://example.com")
        assert result["title"] == "Test"
        assert "Hello" in result["text"]
        assert result["error"] == ""

    @patch("grok_mccodin.web._session.get")
    def test_fetch_error(self, mock_get):
        import requests

        mock_get.side_effect = requests.RequestException("timeout")
        result = web_fetch("https://example.com")
        assert result["error"] == "timeout"
        assert result["text"] == ""

    @patch("grok_mccodin.web._session.get")
    def test_fetch_truncates(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.text = "<p>" + "x" * 10000 + "</p>"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = web_fetch("https://example.com", max_chars=100)
        assert "truncated" in result["text"]
