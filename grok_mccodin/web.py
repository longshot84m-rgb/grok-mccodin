"""Web search and browsing — search via DuckDuckGo, fetch/extract page content."""

from __future__ import annotations

import ipaddress
import logging
import re
from html.parser import HTMLParser
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

# Shared session for connection reuse
_session = requests.Session()
_session.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (compatible; GrokMcCodin/0.1; +https://github.com/longshot84m-rgb/grok-mccodin)"
        ),
    }
)

DEFAULT_TIMEOUT = 15


# ---------------------------------------------------------------------------
# Lightweight HTML-to-text extractor (no bs4 dependency required)
# ---------------------------------------------------------------------------


class _TextExtractor(HTMLParser):
    """Minimal HTML-to-text converter using only stdlib."""

    SKIP_TAGS = {"script", "style", "noscript", "svg", "head"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        if tag in ("br", "p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"):
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6"):
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        # Collapse whitespace runs
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def html_to_text(html: str) -> str:
    """Convert HTML to plain text without external dependencies."""
    parser = _TextExtractor()
    parser.feed(html)
    return parser.get_text()


# ---------------------------------------------------------------------------
# Web Search (DuckDuckGo Lite — no API key required)
# ---------------------------------------------------------------------------


def web_search(query: str, max_results: int = 8) -> list[dict[str, str]]:
    """Search the web using DuckDuckGo Lite.

    Returns a list of ``{"title": ..., "url": ..., "snippet": ...}`` dicts.
    No API key required — uses the public HTML endpoint.
    """
    url = "https://lite.duckduckgo.com/lite/"
    try:
        resp = _session.post(
            url,
            data={"q": query},
            timeout=DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Web search failed: %s", exc)
        return []

    return _parse_ddg_lite(resp.text, max_results)


def _parse_ddg_lite(html: str, max_results: int) -> list[dict[str, str]]:
    """Parse DuckDuckGo Lite HTML results into structured data."""
    results: list[dict[str, str]] = []

    # DDG Lite uses a table layout with specific patterns
    # Links are in <a> tags with class="result-link"
    link_pattern = re.compile(
        r'<a[^>]+class="result-link"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    # Snippets follow in <td> with class="result-snippet"
    snippet_pattern = re.compile(
        r'<td[^>]+class="result-snippet"[^>]*>(.*?)</td>',
        re.DOTALL,
    )

    links = link_pattern.findall(html)
    snippets = snippet_pattern.findall(html)

    for i, (href, title_html) in enumerate(links[:max_results]):
        title = re.sub(r"<[^>]+>", "", title_html).strip()
        snippet = ""
        if i < len(snippets):
            snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()
        if href and title:
            results.append({"title": title, "url": href, "snippet": snippet})

    # Fallback: try generic link extraction if structured parsing found nothing
    if not results:
        results = _parse_ddg_fallback(html, max_results)

    return results


def _parse_ddg_fallback(html: str, max_results: int) -> list[dict[str, str]]:
    """Fallback parser for DDG results using generic link patterns."""
    results: list[dict[str, str]] = []
    # Find all links that look like external results (not DDG internal)
    link_re = re.compile(r'<a[^>]+href="(https?://(?!duckduckgo)[^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
    seen: set[str] = set()
    for href, title_html in link_re.findall(html):
        title = re.sub(r"<[^>]+>", "", title_html).strip()
        if href not in seen and title and len(title) > 3:
            seen.add(href)
            results.append({"title": title, "url": href, "snippet": ""})
            if len(results) >= max_results:
                break
    return results


# ---------------------------------------------------------------------------
# URL safety validation (SSRF prevention)
# ---------------------------------------------------------------------------

_BLOCKED_IP_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def _is_safe_url(url: str) -> str:
    """Validate a URL for safe fetching. Returns an error message or empty string."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return "Invalid URL"

    if parsed.scheme not in ("http", "https"):
        return f"Blocked URL scheme: {parsed.scheme!r} (only http/https allowed)"

    hostname = parsed.hostname or ""
    if not hostname:
        return "URL has no hostname"

    # Check for IP-based hostnames pointing to private ranges
    try:
        addr = ipaddress.ip_address(hostname)
        for network in _BLOCKED_IP_NETWORKS:
            if addr in network:
                return f"Blocked private/reserved IP: {hostname}"
    except ValueError:
        # Not an IP literal — that's fine, it's a domain name
        pass

    # Block common localhost aliases
    if hostname.lower() in ("localhost", "0.0.0.0"):
        return f"Blocked hostname: {hostname}"

    return ""


# ---------------------------------------------------------------------------
# Web Browse / Fetch
# ---------------------------------------------------------------------------


def web_fetch(url: str, max_chars: int = 8000) -> dict[str, str]:
    """Fetch a URL and return its text content.

    Returns ``{"url": ..., "title": ..., "text": ..., "error": ""}``.
    On failure, ``text`` is empty and ``error`` is set.
    """
    result: dict[str, str] = {"url": url, "title": "", "text": "", "error": ""}
    safety_err = _is_safe_url(url)
    if safety_err:
        result["error"] = safety_err
        return result
    try:
        resp = _session.get(url, timeout=DEFAULT_TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
    except requests.RequestException as exc:
        result["error"] = str(exc)
        return result

    content_type = resp.headers.get("content-type", "")
    if "text/html" in content_type or "application/xhtml" in content_type:
        text = html_to_text(resp.text)
        # Extract <title>
        title_match = re.search(r"<title[^>]*>(.*?)</title>", resp.text, re.DOTALL | re.IGNORECASE)
        if title_match:
            result["title"] = re.sub(r"<[^>]+>", "", title_match.group(1)).strip()
    elif "application/json" in content_type:
        text = resp.text
    else:
        text = resp.text

    result["text"] = text[:max_chars]
    if len(text) > max_chars:
        result["text"] += f"\n\n... [truncated at {max_chars} chars, total: {len(text)}]"
    return result


def web_fetch_raw(url: str) -> bytes:
    """Fetch raw bytes from a URL (for images, PDFs, etc.)."""
    safety_err = _is_safe_url(url)
    if safety_err:
        raise requests.URLRequired(safety_err)
    resp = _session.get(url, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    content: bytes = resp.content
    return content
