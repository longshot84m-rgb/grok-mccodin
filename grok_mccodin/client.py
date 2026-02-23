"""Grok API client — handles chat completions via the xAI API."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Generator

import requests
from tqdm import tqdm

from grok_mccodin.config import Config

logger = logging.getLogger(__name__)

# Status codes that trigger automatic retry
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Retry configuration
_MAX_RETRIES = 3
_BASE_DELAY = 1.0  # seconds
_BACKOFF_FACTOR = 2.0

# System prompt that instructs Grok to behave as a coding assistant
SYSTEM_PROMPT = (
    "You are Grok McCodin, an expert coding assistant with access to powerful tools. "
    "When asked to edit a file, respond with a fenced code block tagged with the "
    "filename (e.g. ```python:path/to/file.py).  "
    "When asked to create a file, prefix the block with CREATE: <path>.  "
    "When asked to delete a file, respond with DELETE: <path>.  "
    "When asked to run a command, respond with RUN: <command>.  "
    "Always explain what you're doing before showing code.\n\n"
    "Available tools (the user can invoke these via slash commands):\n"
    "- Web search (/search) and page fetching (/browse)\n"
    "- Git operations (/git status, diff, log, commit, branch, push, pull)\n"
    "- Package managers (/pip install/list/show, /npm install/list/run)\n"
    "- SQL queries (/sql SELECT ...) on SQLite databases\n"
    "- Docker management (/docker ps, images, logs, stop, build, up, down)\n"
    "- RAG code search (/rag query) for semantic codebase search\n"
    "- MCP servers (/mcp connect/tools/call) for external tool integrations\n"
    "When suggesting actions, you can reference these tools."
)


class GrokClient:
    """Thin wrapper around the xAI chat-completions endpoint."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.base_url = config.grok_base_url.rstrip("/")
        self.model = config.grok_model
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {config.grok_api_key}",
                "Content-Type": "application/json",
            }
        )

    def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        json_payload: dict[str, Any] | None = None,
        stream: bool = False,
        timeout: int = 120,
    ) -> requests.Response:
        """Execute an HTTP request with retry logic for transient errors.

        Retries on 429/500/502/503/504 with exponential backoff.
        Respects the Retry-After header for 429 responses.
        """
        last_resp: requests.Response | None = None

        for attempt in range(_MAX_RETRIES + 1):
            resp = self.session.request(
                method, url, json=json_payload, stream=stream, timeout=timeout
            )

            if resp.status_code not in _RETRYABLE_STATUS_CODES:
                return resp

            last_resp = resp

            if attempt >= _MAX_RETRIES:
                break

            # Calculate delay
            delay = _BASE_DELAY * (_BACKOFF_FACTOR**attempt)

            # Respect Retry-After header for 429
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass  # Non-numeric Retry-After; use calculated delay

            logger.warning(
                "Retryable error %d from %s (attempt %d/%d, retrying in %.1fs)",
                resp.status_code,
                url,
                attempt + 1,
                _MAX_RETRIES,
                delay,
            )
            time.sleep(delay)

        # All retries exhausted — return last response (caller will handle the error)
        assert last_resp is not None
        return last_resp

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Send a chat completion request and return the assistant reply.

        Shows a tqdm spinner while waiting for the API response.
        """
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        logger.debug("POST %s  model=%s  msgs=%d", url, self.model, len(messages))

        # Use tqdm as a simple spinner for the blocking request
        with tqdm(total=0, desc="Thinking", bar_format="{desc}...", leave=False):
            resp = self._request_with_retry("POST", url, json_payload=payload)

        if resp.status_code != 200:
            logger.error("Grok API error %d: %s", resp.status_code, resp.text[:500])
            raise GrokAPIError(resp.status_code, resp.text)

        try:
            data = resp.json()
            reply: str = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            logger.error("Unexpected API response structure: %s", exc)
            raise GrokAPIError(resp.status_code, f"Malformed response: {resp.text[:300]}") from exc
        logger.debug("Reply length: %d chars", len(reply))
        return reply

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Generator[str, None, None]:
        """Send a streaming chat completion request, yielding content chunks.

        Uses SSE (Server-Sent Events) to stream tokens as they are generated.
        Each yielded string is a content delta (partial token).
        """
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        logger.debug("POST %s (stream) model=%s msgs=%d", url, self.model, len(messages))

        resp = self._request_with_retry("POST", url, json_payload=payload, stream=True)

        if resp.status_code != 200:
            logger.error("Grok API error %d: %s", resp.status_code, resp.text[:500])
            raise GrokAPIError(resp.status_code, resp.text)

        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue

            # SSE format: "data: {json}" or "data: [DONE]"
            if not raw_line.startswith("data: "):
                continue

            data_str = raw_line[len("data: ") :]

            if data_str.strip() == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
                delta = chunk["choices"][0]["delta"]
                content = delta.get("content", "")
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, IndexError) as exc:
                logger.debug("Skipping unparseable SSE chunk: %s", exc)
                continue

    def build_messages(
        self,
        history: list[dict[str, str]],
        user_input: str,
        context: str = "",
        memory_context: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Assemble the message list with system prompt, optional context, and history.

        If *memory_context* is provided it replaces *history* — used by the
        ConversationMemory system to inject summaries + recalled + recent messages.
        """
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        if context:
            messages.append({"role": "system", "content": f"Project context:\n{context}"})

        if memory_context is not None:
            messages.extend(memory_context)
        else:
            messages.extend(history)

        messages.append({"role": "user", "content": user_input})
        return messages


class GrokAPIError(Exception):
    """Raised when the Grok API returns a non-200 status."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"Grok API {status_code}: {body[:200]}")
