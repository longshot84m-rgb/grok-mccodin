"""Grok API client — handles chat completions via the xAI API."""

from __future__ import annotations

import logging
from typing import Any

import requests
from tqdm import tqdm

from grok_mccodin.config import Config

logger = logging.getLogger(__name__)

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
            resp = self.session.post(url, json=payload, timeout=120)

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

    def build_messages(
        self,
        history: list[dict[str, str]],
        user_input: str,
        context: str = "",
    ) -> list[dict[str, str]]:
        """Assemble the message list with system prompt, optional context, and history."""
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        if context:
            messages.append({"role": "system", "content": f"Project context:\n{context}"})

        messages.extend(history)
        messages.append({"role": "user", "content": user_input})
        return messages


class GrokAPIError(Exception):
    """Raised when the Grok API returns a non-200 status."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"Grok API {status_code}: {body[:200]}")
