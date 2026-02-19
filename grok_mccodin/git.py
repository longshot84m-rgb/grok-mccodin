"""Git operations — status, commit, branch, diff, log from the chat loop."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30


class GitError(Exception):
    """Raised when a git operation fails."""


def _run_git(args: list[str], *, cwd: str | Path = ".", timeout: int = DEFAULT_TIMEOUT) -> str:
    """Run a git subcommand and return stdout.

    Raises GitError on non-zero exit.
    """
    cmd = ["git", *args]
    logger.debug("git: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(cwd),
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise GitError("git is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitError(f"git command timed out after {timeout}s") from exc

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise GitError(f"git {args[0]} failed (rc={result.returncode}): {stderr}")

    return result.stdout


# ---------------------------------------------------------------------------
# Porcelain commands
# ---------------------------------------------------------------------------


def status(cwd: str | Path = ".") -> str:
    """Return ``git status`` output."""
    return _run_git(["status", "--short", "--branch"], cwd=cwd)


def diff(cwd: str | Path = ".", *, staged: bool = False, path: str = "") -> str:
    """Return ``git diff`` output."""
    args = ["diff"]
    if staged:
        args.append("--cached")
    if path:
        args.extend(["--", path])
    return _run_git(args, cwd=cwd)


def log(cwd: str | Path = ".", *, count: int = 10, oneline: bool = True) -> str:
    """Return ``git log`` output."""
    args = ["log", f"-{count}"]
    if oneline:
        args.append("--oneline")
    return _run_git(args, cwd=cwd)


def branch(cwd: str | Path = ".") -> str:
    """Return ``git branch`` output."""
    return _run_git(["branch", "-a"], cwd=cwd)


def checkout(target: str, *, cwd: str | Path = ".", create: bool = False) -> str:
    """Checkout a branch or commit."""
    args = ["checkout"]
    if create:
        args.append("-b")
    args.append(target)
    return _run_git(args, cwd=cwd)


def add(files: list[str] | str = ".", *, cwd: str | Path = ".") -> str:
    """Stage files for commit."""
    if isinstance(files, str):
        files = [files]
    return _run_git(["add", *files], cwd=cwd)


def commit(message: str, *, cwd: str | Path = ".") -> str:
    """Create a commit with the given message."""
    return _run_git(["commit", "-m", message], cwd=cwd)


def push(
    remote: str = "origin",
    branch_name: str = "",
    *,
    cwd: str | Path = ".",
) -> str:
    """Push to remote."""
    args = ["push", remote]
    if branch_name:
        args.append(branch_name)
    return _run_git(args, cwd=cwd)


def pull(
    remote: str = "origin",
    branch_name: str = "",
    *,
    cwd: str | Path = ".",
) -> str:
    """Pull from remote."""
    args = ["pull", remote]
    if branch_name:
        args.append(branch_name)
    return _run_git(args, cwd=cwd)


def stash(action: str = "push", *, cwd: str | Path = ".") -> str:
    """Git stash operations (push, pop, list, show)."""
    return _run_git(["stash", action], cwd=cwd)


def init(cwd: str | Path = ".") -> str:
    """Initialize a new git repository."""
    return _run_git(["init"], cwd=cwd)


_SAFE_CLONE_SCHEMES = ("https://", "http://", "git://", "ssh://", "git@")


def clone(url: str, dest: str = "", *, cwd: str | Path = ".") -> str:
    """Clone a repository.

    Only allows https://, http://, git://, ssh://, and git@ URLs.
    Blocks file:// and other local schemes to prevent local file exfiltration.
    """
    if not any(url.startswith(scheme) for scheme in _SAFE_CLONE_SCHEMES):
        raise GitError(
            f"Blocked clone URL scheme: {url[:30]}... "
            f"(allowed: {', '.join(_SAFE_CLONE_SCHEMES)})"
        )
    args = ["clone", url]
    if dest:
        args.append(dest)
    return _run_git(args, cwd=cwd)


def remote_list(cwd: str | Path = ".") -> str:
    """List remotes."""
    return _run_git(["remote", "-v"], cwd=cwd)


def show(ref: str = "HEAD", *, cwd: str | Path = ".") -> str:
    """Show a commit."""
    return _run_git(["show", "--stat", ref], cwd=cwd)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def is_git_repo(cwd: str | Path = ".") -> bool:
    """Check if the given directory is inside a git repository."""
    try:
        _run_git(["rev-parse", "--is-inside-work-tree"], cwd=cwd)
        return True
    except GitError:
        return False


def current_branch(cwd: str | Path = ".") -> str:
    """Return the current branch name."""
    return _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd).strip()


def summary(cwd: str | Path = ".") -> str:
    """Return a short summary: branch, status, recent commits."""
    parts: list[str] = []
    try:
        parts.append(f"Branch: {current_branch(cwd)}")
    except GitError:
        return "[not a git repository]"

    try:
        st = status(cwd)
        parts.append(f"Status:\n{st}")
    except GitError:
        pass

    try:
        lg = log(cwd, count=5)
        parts.append(f"Recent commits:\n{lg}")
    except GitError:
        pass

    return "\n".join(parts)
