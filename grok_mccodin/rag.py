"""RAG / Embeddings — local TF-IDF search over project codebases.

Uses only stdlib (no ML dependencies required).  For large codebases,
this provides a fast, dependency-free semantic search that can be used
to feed relevant code context into the Grok prompt.

Optional: If numpy/scikit-learn are installed, can use real embeddings
via the Grok API embedding endpoint.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from grok_mccodin.utils import CODE_EXTENSIONS, _should_skip_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens (identifiers + words)."""
    # Split on non-alphanumeric, keep underscored identifiers together
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text)
    result: list[str] = []
    for token in tokens:
        # Also split camelCase into sub-tokens
        parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)|[a-z]+|\d+", token)
        result.append(token.lower())
        for part in parts:
            lower = part.lower()
            if lower != token.lower():
                result.append(lower)
    return result


# ---------------------------------------------------------------------------
# TF-IDF Index (stdlib only)
# ---------------------------------------------------------------------------


class TFIDFIndex:
    """Simple TF-IDF index for code search.

    Build an index from project files, then search with natural language
    queries to find the most relevant files/chunks.
    """

    def __init__(self) -> None:
        self._documents: list[dict[str, Any]] = []  # {path, chunk, text, tokens}
        self._idf: dict[str, float] = {}
        self._built = False

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def index_folder(
        self,
        folder: str | Path,
        *,
        max_depth: int = 4,
        chunk_lines: int = 50,
        max_files: int = 500,
    ) -> int:
        """Index all code files in a folder. Returns number of chunks indexed."""
        folder = Path(folder)
        if not folder.is_dir():
            return 0

        file_count = 0
        self._documents.clear()
        self._built = False

        for path in self._walk_files(folder, max_depth):
            if file_count >= max_files:
                break
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            file_count += 1
            rel_path = str(path.relative_to(folder))
            lines = text.split("\n")

            # Split into overlapping chunks
            for start in range(0, len(lines), chunk_lines // 2):
                chunk_text = "\n".join(lines[start : start + chunk_lines])
                if not chunk_text.strip():
                    continue
                tokens = _tokenize(chunk_text)
                if not tokens:
                    continue
                self._documents.append(
                    {
                        "path": rel_path,
                        "chunk": start,
                        "text": chunk_text,
                        "tokens": tokens,
                    }
                )

        self._build_idf()
        logger.info("Indexed %d files, %d chunks", file_count, len(self._documents))
        return len(self._documents)

    def index_text(self, name: str, text: str, chunk_lines: int = 50) -> None:
        """Add arbitrary text to the index (useful for docs, READMEs, etc.)."""
        self._built = False
        lines = text.split("\n")
        for start in range(0, len(lines), chunk_lines // 2):
            chunk_text = "\n".join(lines[start : start + chunk_lines])
            if not chunk_text.strip():
                continue
            tokens = _tokenize(chunk_text)
            if tokens:
                self._documents.append(
                    {"path": name, "chunk": start, "text": chunk_text, "tokens": tokens}
                )
        self._build_idf()

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search the index for the most relevant chunks.

        Returns a list of ``{"path", "chunk", "score", "text"}`` sorted by relevance.
        """
        if not self._built:
            self._build_idf()

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        query_tf = Counter(query_tokens)
        query_vec = self._tfidf_vector(query_tf)

        scored: list[tuple[float, int]] = []
        for i, doc in enumerate(self._documents):
            doc_tf = Counter(doc["tokens"])
            doc_vec = self._tfidf_vector(doc_tf)
            score = self._cosine_similarity(query_vec, doc_vec)
            if score > 0:
                scored.append((score, i))

        scored.sort(reverse=True)
        results: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for score, idx in scored:
            doc = self._documents[idx]
            # Deduplicate overlapping chunks from same file
            key = f"{doc['path']}:{doc['chunk']}"
            if key in seen_paths:
                continue
            seen_paths.add(key)
            results.append(
                {
                    "path": doc["path"],
                    "chunk": doc["chunk"],
                    "score": round(score, 4),
                    "text": doc["text"],
                }
            )
            if len(results) >= top_k:
                break

        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _walk_files(self, folder: Path, max_depth: int, depth: int = 0) -> list[Path]:
        """Recursively collect code files."""
        if depth > max_depth:
            return []
        files: list[Path] = []
        try:
            entries = sorted(folder.iterdir(), key=lambda p: p.name.lower())
        except (PermissionError, OSError):
            return []
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                if _should_skip_dir(entry.name):
                    continue
                files.extend(self._walk_files(entry, max_depth, depth + 1))
            elif entry.is_file() and entry.suffix in CODE_EXTENSIONS:
                files.append(entry)
        return files

    def _build_idf(self) -> None:
        """Compute inverse document frequency for all terms."""
        n = len(self._documents)
        if n == 0:
            self._idf = {}
            self._built = True
            return

        doc_freq: Counter[str] = Counter()
        for doc in self._documents:
            unique_tokens = set(doc["tokens"])
            for token in unique_tokens:
                doc_freq[token] += 1

        self._idf = {token: math.log((n + 1) / (freq + 1)) + 1 for token, freq in doc_freq.items()}
        self._built = True

    def _tfidf_vector(self, tf: Counter[str]) -> dict[str, float]:
        """Compute TF-IDF vector from a term frequency counter."""
        vec: dict[str, float] = {}
        for token, count in tf.items():
            idf = self._idf.get(token, 0)
            if idf > 0:
                vec[token] = count * idf
        return vec

    @staticmethod
    def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
        """Compute cosine similarity between two sparse vectors."""
        if not a or not b:
            return 0.0

        # Dot product (only on shared keys)
        shared_keys = set(a.keys()) & set(b.keys())
        if not shared_keys:
            return 0.0

        dot = sum(a[k] * b[k] for k in shared_keys)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Convenience: build context for Grok from search results
# ---------------------------------------------------------------------------


def search_codebase(
    folder: str | Path,
    query: str,
    *,
    top_k: int = 5,
    chunk_lines: int = 50,
) -> str:
    """One-shot: index a folder and search it, returning formatted context.

    This is the main entry point for injecting RAG context into Grok prompts.
    """
    index = TFIDFIndex()
    indexed = index.index_folder(folder, chunk_lines=chunk_lines)
    if indexed == 0:
        return "[no code files found to search]"

    results = index.search(query, top_k=top_k)
    if not results:
        return "[no relevant results found]"

    parts: list[str] = [f"Found {len(results)} relevant code sections:\n"]
    for r in results:
        parts.append(f"--- {r['path']} (line ~{r['chunk']}, score={r['score']}) ---")
        parts.append(r["text"])
        parts.append("")

    return "\n".join(parts)
