"""Docker container management — ps, run, stop, logs, images, build."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60


class DockerError(Exception):
    """Raised when a Docker operation fails."""


def _run_docker(args: list[str], *, cwd: str | Path = ".", timeout: int = DEFAULT_TIMEOUT) -> str:
    """Run a docker subcommand and return stdout."""
    cmd = ["docker", *args]
    logger.debug("docker: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(cwd),
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise DockerError("Docker is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise DockerError(f"Docker command timed out after {timeout}s") from exc

    if result.returncode != 0:
        raise DockerError(
            f"docker {args[0]} failed (rc={result.returncode}): {result.stderr.strip()}"
        )

    return result.stdout


# ---------------------------------------------------------------------------
# Container operations
# ---------------------------------------------------------------------------


def ps(*, all_: bool = False, cwd: str | Path = ".") -> str:
    """List running containers (or all with all_=True)."""
    args = ["ps", "--format", "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}"]
    if all_:
        args.insert(1, "-a")
    return _run_docker(args, cwd=cwd)


def ps_json(*, all_: bool = False, cwd: str | Path = ".") -> list[dict]:
    """List containers as structured data."""
    args = ["ps", "--format", "json"]
    if all_:
        args.insert(1, "-a")
    output = _run_docker(args, cwd=cwd)
    results: list[dict] = []
    for line in output.strip().split("\n"):
        if line.strip():
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def run(
    image: str,
    *,
    name: str = "",
    ports: list[str] | None = None,
    env: dict[str, str] | None = None,
    volumes: list[str] | None = None,
    detach: bool = True,
    command: str = "",
    cwd: str | Path = ".",
) -> str:
    """Run a container."""
    args = ["run"]
    if detach:
        args.append("-d")
    if name:
        args.extend(["--name", name])
    for p in ports or []:
        args.extend(["-p", p])
    for k, v in (env or {}).items():
        args.extend(["-e", f"{k}={v}"])
    for vol in volumes or []:
        args.extend(["-v", vol])
    args.append(image)
    if command:
        args.extend(command.split())
    return _run_docker(args, cwd=cwd).strip()


def stop(container: str, *, timeout: int = 10, cwd: str | Path = ".") -> str:
    """Stop a running container."""
    return _run_docker(["stop", "-t", str(timeout), container], cwd=cwd)


def rm(container: str, *, force: bool = False, cwd: str | Path = ".") -> str:
    """Remove a container."""
    args = ["rm"]
    if force:
        args.append("-f")
    args.append(container)
    return _run_docker(args, cwd=cwd)


def logs(container: str, *, tail: int = 100, cwd: str | Path = ".") -> str:
    """Fetch container logs."""
    return _run_docker(["logs", "--tail", str(tail), container], cwd=cwd)


def exec_(container: str, command: str, *, cwd: str | Path = ".") -> str:
    """Execute a command inside a running container."""
    args = ["exec", container, *command.split()]
    return _run_docker(args, cwd=cwd)


# ---------------------------------------------------------------------------
# Image operations
# ---------------------------------------------------------------------------


def images(*, cwd: str | Path = ".") -> str:
    """List Docker images."""
    return _run_docker(
        ["images", "--format", "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}"],
        cwd=cwd,
    )


def build(
    path: str = ".",
    *,
    tag: str = "",
    dockerfile: str = "",
    cwd: str | Path = ".",
) -> str:
    """Build a Docker image."""
    args = ["build"]
    if tag:
        args.extend(["-t", tag])
    if dockerfile:
        args.extend(["-f", dockerfile])
    args.append(path)
    return _run_docker(args, cwd=cwd, timeout=300)


def pull(image: str, *, cwd: str | Path = ".") -> str:
    """Pull a Docker image."""
    return _run_docker(["pull", image], cwd=cwd, timeout=300)


def push(image: str, *, cwd: str | Path = ".") -> str:
    """Push a Docker image."""
    return _run_docker(["push", image], cwd=cwd, timeout=300)


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


def compose_up(
    *,
    detach: bool = True,
    file: str = "",
    cwd: str | Path = ".",
) -> str:
    """Run docker compose up."""
    args = ["compose"]
    if file:
        args.extend(["-f", file])
    args.append("up")
    if detach:
        args.append("-d")
    return _run_docker(args, cwd=cwd, timeout=300)


def compose_down(*, file: str = "", cwd: str | Path = ".") -> str:
    """Run docker compose down."""
    args = ["compose"]
    if file:
        args.extend(["-f", file])
    args.append("down")
    return _run_docker(args, cwd=cwd)


def compose_ps(*, file: str = "", cwd: str | Path = ".") -> str:
    """List compose services."""
    args = ["compose"]
    if file:
        args.extend(["-f", file])
    args.append("ps")
    return _run_docker(args, cwd=cwd)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def is_docker_available() -> bool:
    """Check if Docker is installed and the daemon is running."""
    try:
        _run_docker(["info"], timeout=10)
        return True
    except DockerError:
        return False


def summary(cwd: str | Path = ".") -> str:
    """Return a quick Docker status summary."""
    parts: list[str] = []
    try:
        containers = ps(cwd=cwd)
        parts.append(f"Running containers:\n{containers}")
    except DockerError:
        return "[Docker not available]"

    try:
        imgs = images(cwd=cwd)
        parts.append(f"\nImages:\n{imgs}")
    except DockerError:
        pass

    return "\n".join(parts) if parts else "[no Docker info]"
