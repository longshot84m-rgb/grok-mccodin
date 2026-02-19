"""Package manager integration — pip and npm operations from the chat loop."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 120  # Package installs can be slow


class PackageError(Exception):
    """Raised when a package manager operation fails."""


def _run_cmd(cmd: list[str], *, cwd: str | Path = ".", timeout: int = DEFAULT_TIMEOUT) -> str:
    """Run a command and return stdout. Raises PackageError on failure."""
    logger.debug("pkg: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(cwd),
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise PackageError(f"Command not found: {cmd[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise PackageError(f"Command timed out after {timeout}s") from exc

    if result.returncode != 0:
        raise PackageError(f"{cmd[0]} failed (rc={result.returncode}): {result.stderr.strip()}")

    return result.stdout


# ---------------------------------------------------------------------------
# pip
# ---------------------------------------------------------------------------


def pip_install(
    packages: str | list[str],
    *,
    cwd: str | Path = ".",
    upgrade: bool = False,
) -> str:
    """Install Python packages via pip."""
    if isinstance(packages, str):
        packages = packages.split()
    cmd = ["pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.extend(packages)
    return _run_cmd(cmd, cwd=cwd)


def pip_uninstall(packages: str | list[str], *, cwd: str | Path = ".") -> str:
    """Uninstall Python packages via pip."""
    if isinstance(packages, str):
        packages = packages.split()
    cmd = ["pip", "uninstall", "-y", *packages]
    return _run_cmd(cmd, cwd=cwd)


def pip_list(*, cwd: str | Path = ".") -> list[dict[str, str]]:
    """List installed Python packages. Returns [{"name": ..., "version": ...}]."""
    output = _run_cmd(["pip", "list", "--format=json"], cwd=cwd)
    try:
        data: list[dict[str, str]] = json.loads(output)
        return data
    except json.JSONDecodeError:
        return []


def pip_show(package: str, *, cwd: str | Path = ".") -> str:
    """Show details about an installed pip package."""
    return _run_cmd(["pip", "show", package], cwd=cwd)


def pip_freeze(*, cwd: str | Path = ".") -> str:
    """Return pip freeze output (pinned requirements)."""
    return _run_cmd(["pip", "freeze"], cwd=cwd)


# ---------------------------------------------------------------------------
# npm
# ---------------------------------------------------------------------------


def npm_install(
    packages: str | list[str] | None = None,
    *,
    cwd: str | Path = ".",
    dev: bool = False,
    global_: bool = False,
) -> str:
    """Install npm packages (or all deps if no packages specified)."""
    cmd = ["npm", "install"]
    if global_:
        cmd.append("-g")
    if dev:
        cmd.append("--save-dev")
    if packages:
        if isinstance(packages, str):
            packages = packages.split()
        cmd.extend(packages)
    return _run_cmd(cmd, cwd=cwd)


def npm_uninstall(packages: str | list[str], *, cwd: str | Path = ".") -> str:
    """Uninstall npm packages."""
    if isinstance(packages, str):
        packages = packages.split()
    cmd = ["npm", "uninstall", *packages]
    return _run_cmd(cmd, cwd=cwd)


def npm_list(*, cwd: str | Path = ".", depth: int = 0) -> str:
    """List installed npm packages."""
    return _run_cmd(["npm", "list", f"--depth={depth}"], cwd=cwd)


def npm_run(script: str, *, cwd: str | Path = ".") -> str:
    """Run an npm script."""
    return _run_cmd(["npm", "run", script], cwd=cwd)


def npm_init(*, cwd: str | Path = ".") -> str:
    """Initialize a new package.json."""
    return _run_cmd(["npm", "init", "-y"], cwd=cwd)


# ---------------------------------------------------------------------------
# Detect package manager type
# ---------------------------------------------------------------------------


def detect_package_manager(cwd: str | Path = ".") -> str:
    """Detect whether the project uses pip, npm, both, or unknown."""
    cwd = Path(cwd)
    managers: list[str] = []

    if (cwd / "requirements.txt").exists() or (cwd / "pyproject.toml").exists():
        managers.append("pip")
    if (cwd / "setup.py").exists() or (cwd / "setup.cfg").exists():
        managers.append("pip")
    if (cwd / "package.json").exists():
        managers.append("npm")
    if (cwd / "yarn.lock").exists():
        managers.append("yarn")

    if not managers:
        return "unknown"
    return "+".join(sorted(set(managers)))
