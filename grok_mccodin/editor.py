"""File editing — parse Grok responses, apply diffs, create/delete files."""

from __future__ import annotations

import difflib
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.syntax import Syntax

logger = logging.getLogger(__name__)
console = Console()


def _safe_resolve(filepath: str | Path, base_dir: str | Path) -> Path | None:
    """Resolve *filepath* relative to *base_dir*, rejecting path traversal.

    Returns the resolved Path if it stays within *base_dir*, or None if it
    would escape (e.g. ``../../etc/passwd``).
    """
    base = Path(base_dir).resolve()
    target = (base / filepath).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        logger.warning("Path traversal blocked: %s escapes %s", filepath, base)
        return None
    return target


def extract_code_blocks(text: str) -> list[dict[str, str]]:
    """Extract fenced code blocks from a Grok response.

    Returns a list of dicts:
        {"lang": "python", "filename": "src/foo.py", "code": "..."}

    Supports formats:
        ```python:path/to/file.py
        ```python
        ```
    """
    pattern = r"```(\w*)?(?::([^\n]+))?\n(.*?)```"
    blocks = []
    for match in re.finditer(pattern, text, re.DOTALL):
        lang = match.group(1) or ""
        filename = match.group(2) or ""
        code = match.group(3).rstrip("\n")
        blocks.append({"lang": lang, "filename": filename.strip(), "code": code})
    return blocks


def extract_commands(text: str) -> list[str]:
    """Extract RUN: <command> lines from a Grok response."""
    return re.findall(r"^RUN:\s*(.+)$", text, re.MULTILINE)


def extract_creates(text: str) -> list[dict[str, str]]:
    """Extract CREATE: <path> directives followed by code blocks."""
    creates = []
    pattern = r"CREATE:\s*(\S+)\s*\n```\w*\n(.*?)```"
    for match in re.finditer(pattern, text, re.DOTALL):
        creates.append({"path": match.group(1).strip(), "code": match.group(2).rstrip("\n")})
    return creates


def extract_deletes(text: str) -> list[str]:
    """Extract DELETE: <path> directives."""
    return re.findall(r"^DELETE:\s*(\S+)$", text, re.MULTILINE)


def show_diff(original: str, modified: str, filename: str = "") -> str:
    """Generate a unified diff between two strings."""
    orig_lines = original.splitlines(keepends=True)
    mod_lines = modified.splitlines(keepends=True)
    diff = difflib.unified_diff(
        orig_lines,
        mod_lines,
        fromfile=f"a/{filename}" if filename else "a/original",
        tofile=f"b/{filename}" if filename else "b/modified",
    )
    return "".join(diff)


def apply_edit(filepath: str | Path, new_content: str, *, base_dir: str | Path = ".") -> str:
    """Write *new_content* to *filepath* (relative to *base_dir*), showing a diff first.

    Returns a status message.  Rejects paths that escape *base_dir*.
    """
    resolved = _safe_resolve(filepath, base_dir)
    if resolved is None:
        return f"[blocked] Path traversal: {filepath}"

    resolved.parent.mkdir(parents=True, exist_ok=True)

    if resolved.is_file():
        original = resolved.read_text(encoding="utf-8", errors="replace")
        diff = show_diff(original, new_content, filename=str(filepath))
        if not diff:
            return f"No changes needed for {filepath}"
        console.print(f"\n[bold yellow]Diff for {filepath}:[/bold yellow]")
        console.print(Syntax(diff, "diff", theme="monokai"))
    else:
        console.print(f"\n[bold green]Creating new file: {filepath}[/bold green]")

    resolved.write_text(new_content, encoding="utf-8")
    logger.info("Wrote %s (%d bytes)", resolved, len(new_content))
    return f"Updated {filepath}"


def apply_create(filepath: str | Path, content: str, *, base_dir: str | Path = ".") -> str:
    """Create a new file at *filepath*.  Rejects paths that escape *base_dir*."""
    resolved = _safe_resolve(filepath, base_dir)
    if resolved is None:
        return f"[blocked] Path traversal: {filepath}"

    if resolved.exists():
        return f"[skip] {filepath} already exists — use edit instead"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    console.print(f"[bold green]Created {filepath}[/bold green]")
    return f"Created {filepath}"


def apply_delete(filepath: str | Path, *, base_dir: str | Path = ".") -> str:
    """Delete a file (moves to .trash/ first for safety).

    Uses timestamped names to avoid collisions in .trash/.
    Rejects paths that escape *base_dir*.
    """
    resolved = _safe_resolve(filepath, base_dir)
    if resolved is None:
        return f"[blocked] Path traversal: {filepath}"

    if not resolved.exists():
        return f"[skip] {filepath} does not exist"

    trash_dir = Path(base_dir).resolve() / ".trash"
    trash_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = trash_dir / f"{timestamp}_{resolved.name}"
    shutil.move(str(resolved), str(dest))
    console.print(f"[bold red]Deleted {filepath}[/bold red] (backed up to .trash/)")
    return f"Deleted {filepath}"
