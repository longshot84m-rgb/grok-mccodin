"""Conversation memory — rolling compression, TF-IDF recall, and JSONL persistence.

Provides unlimited conversation context by compressing old messages into
summaries, recalling semantically relevant content via TF-IDF, and persisting
full session history to disk.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from grok_mccodin.rag import TFIDFIndex

logger = logging.getLogger(__name__)

# Keywords that signal important decisions or facts worth preserving
_DECISION_KEYWORDS = (
    "decided",
    "agreed",
    "must",
    "requirement",
    "important",
    "error",
    "fix",
    "breaking",
    "critical",
)

# Messages scoring above this threshold are kept verbatim during compression
_IMPORTANCE_THRESHOLD = 0.7

# Chunk size for TF-IDF indexing — large because individual messages are short
_INDEX_CHUNK_LINES = 200


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (~4 chars per token)."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Importance scoring
# ---------------------------------------------------------------------------


@dataclass
class ScoredMessage:
    """A conversation message annotated with importance and timestamp."""

    role: str
    content: str
    importance: float
    timestamp: str


def score_importance(role: str, content: str, index: int, total: int) -> float:
    """Score a message 0.0–1.0 based on role, content signals, and recency.

    Factors:
    - Recency (position in history): 0.0–0.3
    - Role weight: assistant 0.2, user 0.15, other 0.1
    - Code blocks (```): +0.2
    - Decision keywords: +0.15
    - Substantive length (>200 chars): +0.1
    """
    score = 0.0

    # Recency: linear scale from 0.0 (oldest) to 0.3 (newest)
    score += 0.3 * (index / max(total - 1, 1))

    # Role weight
    score += {"assistant": 0.2, "user": 0.15}.get(role, 0.1)

    # Code blocks
    if "```" in content:
        score += 0.2

    # Decision keywords
    lower = content.lower()
    if any(kw in lower for kw in _DECISION_KEYWORDS):
        score += 0.15

    # Substantive length
    if len(content) > 200:
        score += 0.1

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


def _distill_facts(messages: list[dict[str, str]]) -> str:
    """Extract key facts from low-importance messages.

    Keeps: questions, code blocks, lines with decision keywords.
    Drops: greetings, acknowledgments, short pleasantries.
    """
    facts: list[str] = []

    for msg in messages:
        content = msg["content"]

        # Skip short pleasantries
        if len(content) < 20 and "```" not in content:
            continue

        lines = content.split("\n")
        in_code_block = False
        code_block_lines: list[str] = []

        for line in lines:
            stripped = line.strip()

            # Track code blocks
            if stripped.startswith("```"):
                if in_code_block:
                    # End of code block — keep it
                    code_block_lines.append(line)
                    facts.append("\n".join(code_block_lines))
                    code_block_lines = []
                    in_code_block = False
                else:
                    in_code_block = True
                    code_block_lines = [line]
                continue

            if in_code_block:
                code_block_lines.append(line)
                continue

            # Keep questions
            if stripped.endswith("?"):
                facts.append(f"[{msg['role']}]: {stripped}")
                continue

            # Keep lines with decision keywords
            lower = stripped.lower()
            if any(kw in lower for kw in _DECISION_KEYWORDS):
                facts.append(f"[{msg['role']}]: {stripped}")
                continue

    # Deduplicate near-identical facts
    seen: set[str] = set()
    unique: list[str] = []
    for fact in facts:
        normalized = re.sub(r"\s+", " ", fact.strip().lower())
        if normalized not in seen:
            seen.add(normalized)
            unique.append(fact)

    return "\n".join(unique)


def compress_messages(scored_messages: list[ScoredMessage], max_chars: int = 2000) -> str:
    """Compress a batch of messages into a summary string.

    High-importance messages (>= threshold) are kept verbatim.
    Lower-importance messages are distilled to key facts.
    """
    preserved: list[str] = []
    to_distill: list[dict[str, str]] = []

    for msg in scored_messages:
        if msg.importance >= _IMPORTANCE_THRESHOLD:
            preserved.append(f"[{msg.role}]: {msg.content}")
        else:
            to_distill.append({"role": msg.role, "content": msg.content})

    distilled = _distill_facts(to_distill)

    parts: list[str] = []
    if distilled:
        parts.append(distilled)
    if preserved:
        parts.append("Key exchanges:\n" + "\n".join(preserved))

    combined = "\n---\n".join(parts)
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "\n[...truncated]"

    return f"Summary of earlier conversation:\n{combined}"


# ---------------------------------------------------------------------------
# Filename sanitization
# ---------------------------------------------------------------------------


def _sanitize_filename(name: str) -> str:
    """Remove characters unsafe for filenames, collapse whitespace."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name)
    cleaned = re.sub(r"\s+", "_", cleaned.strip())
    return cleaned or "session"


# ---------------------------------------------------------------------------
# ConversationMemory
# ---------------------------------------------------------------------------


class ConversationMemory:
    """Manages unlimited conversation context via compression + recall.

    Architecture:
    - ``_all_messages``: append-only full log (never trimmed, for persistence + indexing)
    - ``_messages``: recent window (trimmed on compression, for context assembly)
    - ``_summaries``: compressed segments of old messages
    - ``_index``: TF-IDF index over ALL messages (built from ``_all_messages``)
    """

    def __init__(
        self,
        token_budget: int = 6000,
        keep_recent: int = 10,
        memory_dir: str = "~/.grok_mccodin/sessions",
        top_k: int = 3,
    ) -> None:
        self._all_messages: list[ScoredMessage] = []
        self._messages: list[ScoredMessage] = []
        self._summaries: list[str] = []
        self._index: TFIDFIndex = TFIDFIndex()
        self._token_budget = token_budget
        self._keep_recent = keep_recent
        self._memory_dir = Path(memory_dir).expanduser()
        self._top_k = top_k
        self._session_name: str | None = None
        self._message_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, role: str, content: str) -> None:
        """Append a message and trigger compression if over budget."""
        idx = len(self._all_messages)
        total = idx + 1
        importance = score_importance(role, content, idx, total)
        ts = datetime.now(timezone.utc).isoformat()
        msg = ScoredMessage(role=role, content=content, importance=importance, timestamp=ts)

        self._all_messages.append(msg)
        self._messages.append(msg)
        self._message_count += 1

        # Index for future recall — persists across compressions
        self._index.index_text(
            f"msg_{self._message_count}", content, chunk_lines=_INDEX_CHUNK_LINES
        )

        self._maybe_compress()

    def build_context(self, user_input: str) -> list[dict[str, str]]:
        """Assemble context for the API call.

        Returns ``[summaries] + [recalled] + [recent]``.
        Does NOT include *user_input* — that is appended by ``build_messages()``.
        """
        parts: list[dict[str, str]] = []

        # 1. Compressed summaries (oldest context)
        if self._summaries:
            combined = "\n---\n".join(self._summaries)
            parts.append(
                {"role": "system", "content": f"Earlier conversation summary:\n{combined}"}
            )

        # 2. TF-IDF recalled content (semantically relevant old messages)
        recalled = self._recall(user_input)
        if recalled:
            parts.append({"role": "system", "content": f"Relevant earlier context:\n{recalled}"})

        # 3. Recent messages (uncompressed)
        recent = self._messages[-self._keep_recent :]
        for msg in recent:
            parts.append({"role": msg.role, "content": msg.content})

        return parts

    def save_session(self, name: str | None = None) -> Path:
        """Persist the full session (``_all_messages`` + ``_summaries``) to JSONL."""
        name = name or self._session_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._session_name = name
        path = self._memory_dir / f"{_sanitize_filename(name)}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as fh:
            for msg in self._all_messages:
                json.dump(
                    {
                        "ts": msg.timestamp,
                        "role": msg.role,
                        "content": msg.content,
                        "importance": msg.importance,
                    },
                    fh,
                )
                fh.write("\n")
            for summary in self._summaries:
                json.dump(
                    {"ts": "", "role": "summary", "content": summary, "importance": 1.0},
                    fh,
                )
                fh.write("\n")

        logger.info("Session saved: %s (%d messages)", path, len(self._all_messages))
        return path

    def load_session(self, name: str) -> None:
        """Load a session from JSONL, rebuilding all internal state."""
        path = self._memory_dir / f"{_sanitize_filename(name)}.jsonl"
        self.clear()

        all_msgs: list[ScoredMessage] = []
        summaries: list[str] = []

        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec["role"] == "summary":
                    summaries.append(rec["content"])
                else:
                    msg = ScoredMessage(
                        role=rec["role"],
                        content=rec["content"],
                        importance=rec["importance"],
                        timestamp=rec["ts"],
                    )
                    all_msgs.append(msg)
                    self._message_count += 1
                    self._index.index_text(
                        f"msg_{self._message_count}",
                        rec["content"],
                        chunk_lines=_INDEX_CHUNK_LINES,
                    )

        self._all_messages = all_msgs
        self._summaries = summaries
        # Rebuild the recent window
        if len(all_msgs) > self._keep_recent:
            self._messages = list(all_msgs[-self._keep_recent :])
        else:
            self._messages = list(all_msgs)
        self._session_name = name

        logger.info(
            "Session loaded: %s (%d messages, %d summaries)", name, len(all_msgs), len(summaries)
        )

    def list_sessions(self) -> list[str]:
        """Return sorted names of saved sessions."""
        if not self._memory_dir.exists():
            return []
        return sorted(p.stem for p in self._memory_dir.glob("*.jsonl"))

    def clear(self) -> None:
        """Reset all internal state."""
        self._all_messages.clear()
        self._messages.clear()
        self._summaries.clear()
        self._index = TFIDFIndex()
        self._message_count = 0

    @property
    def stats(self) -> dict[str, Any]:
        """Return a snapshot of memory statistics."""
        return {
            "messages": len(self._messages),
            "total_messages": len(self._all_messages),
            "summaries": len(self._summaries),
            "total_indexed": self._index.document_count,
            "total_tokens_est": sum(estimate_tokens(m.content) for m in self._messages),
            "session": self._session_name,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_compress(self) -> None:
        """Compress oldest messages when the recent window exceeds the token budget."""
        total_tokens = sum(estimate_tokens(m.content) for m in self._messages)
        if total_tokens <= self._token_budget:
            return

        to_compress = self._messages[: -self._keep_recent]
        if not to_compress:
            return

        summary = compress_messages(to_compress)
        self._summaries.append(summary)

        # Trim recent window — full content preserved in _all_messages + TF-IDF index
        self._messages = self._messages[-self._keep_recent :]

        logger.debug(
            "Compressed %d messages into summary (%d chars). %d summaries total.",
            len(to_compress),
            len(summary),
            len(self._summaries),
        )

    def _recall(self, query: str) -> str:
        """Search the TF-IDF index for relevant old context, deduplicating against recent."""
        results = self._index.search(query, top_k=self._top_k + self._keep_recent)
        if not results:
            return ""

        # Build set of recent message content for deduplication
        recent_content = {m.content for m in self._messages[-self._keep_recent :]}

        recalled_parts: list[str] = []
        for r in results:
            if r.get("score", 0) < 0.1:
                continue
            text = r["text"]
            if text in recent_content:
                continue
            recalled_parts.append(text)
            if len(recalled_parts) >= self._top_k:
                break

        return "\n".join(recalled_parts)
