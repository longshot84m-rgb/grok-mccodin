"""Utility helpers — folder indexing, receipt logging, screenshot."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Extensions we care about when indexing a project folder
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".cs",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".html",
    ".css",
    ".scss",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".md",
    ".txt",
    ".sh",
    ".bat",
}

# Directories to skip during indexing
SKIP_DIRS = {
    "__pycache__",
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "env",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".tox",
    ".eggs",
    "*.egg-info",
}


def index_folder(folder: str | Path, max_depth: int = 4) -> str:
    """Build a tree-style index of a project folder.

    Returns a string like:
        src/
          main.py (120 lines)
          utils.py (45 lines)
        tests/
          test_main.py (30 lines)
    """
    folder = Path(folder)
    if not folder.is_dir():
        return f"[not a directory: {folder}]"

    lines: list[str] = []

    def _walk(current: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return

        for entry in entries:
            if entry.name.startswith(".") and entry.name not in (".env.example",):
                continue
            if entry.is_dir():
                if entry.name in SKIP_DIRS:
                    continue
                indent = "  " * depth
                lines.append(f"{indent}{entry.name}/")
                _walk(entry, depth + 1)
            elif entry.is_file() and entry.suffix in CODE_EXTENSIONS:
                try:
                    line_count = sum(1 for _ in entry.open(encoding="utf-8", errors="ignore"))
                except OSError:
                    line_count = 0
                indent = "  " * depth
                lines.append(f"{indent}{entry.name} ({line_count} lines)")

    _walk(folder, 0)
    return "\n".join(lines) if lines else "[empty project]"


def read_file_safe(path: str | Path, max_lines: int = 500) -> str:
    """Read a file, returning at most *max_lines* lines."""
    path = Path(path)
    if not path.is_file():
        return f"[file not found: {path}]"
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            content_lines = []
            for i, line in enumerate(fh):
                if i >= max_lines:
                    content_lines.append(f"\n... truncated at {max_lines} lines ...")
                    break
                content_lines.append(line)
        return "".join(content_lines)
    except OSError as exc:
        return f"[error reading {path}: {exc}]"


def log_receipt(
    log_file: str | Path,
    *,
    action: str,
    detail: str = "",
    user_input: str = "",
) -> None:
    """Append a JSON receipt entry to the log file."""
    log_file = Path(log_file)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "detail": detail,
        "input_hash": hashlib.sha256(user_input.encode()).hexdigest()[:16] if user_input else "",
    }

    existing: list[dict] = []
    if log_file.is_file():
        try:
            existing = json.loads(log_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    existing.append(entry)
    log_file.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    logger.debug("Logged receipt: %s", action)


def take_screenshot(output_path: str = "screenshot.png") -> str:
    """Capture a screenshot (requires pyautogui — optional dependency)."""
    try:
        import pyautogui  # type: ignore[import-untyped]

        img = pyautogui.screenshot()
        img.save(output_path)
        return output_path
    except ImportError:
        return "[pyautogui not installed — run: pip install pyautogui]"
    except Exception as exc:
        return f"[screenshot failed: {exc}]"


def file_hash(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
