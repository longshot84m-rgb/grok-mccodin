"""Code and shell execution — sandboxed subprocess runner."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

logger = logging.getLogger(__name__)
console = Console()

# Commands that are always blocked
BLOCKED_COMMANDS = {
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=/dev/zero",
    ":(){:|:&};:",
}

# Max seconds a subprocess is allowed to run
DEFAULT_TIMEOUT = 30


def is_safe(command: str) -> bool:
    """Return False if the command matches a known-dangerous pattern."""
    normalized = command.strip().lower()
    for blocked in BLOCKED_COMMANDS:
        if blocked in normalized:
            return False
    return True


def run_shell(
    command: str,
    *,
    cwd: str | Path = ".",
    timeout: int = DEFAULT_TIMEOUT,
    safe_lock: bool = False,
) -> dict[str, str | int]:
    """Execute a shell command and return its output.

    Returns:
        {"stdout": ..., "stderr": ..., "returncode": ...}
    """
    if safe_lock:
        return {"stdout": "", "stderr": "[Safe Lock ON] Execution blocked.", "returncode": -1}

    if not is_safe(command):
        return {"stdout": "", "stderr": f"[BLOCKED] Dangerous command: {command}", "returncode": -1}

    logger.info("Executing: %s", command)
    console.print(Panel(command, title="Running", border_style="cyan"))

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=str(cwd),
            timeout=timeout,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"[TIMEOUT] Command exceeded {timeout}s", "returncode": -1}
    except OSError as exc:
        return {"stdout": "", "stderr": f"[ERROR] {exc}", "returncode": -1}


def run_python(
    code: str, *, cwd: str | Path = ".", timeout: int = DEFAULT_TIMEOUT
) -> dict[str, str | int]:
    """Write *code* to a temp file and execute it with ``python``."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=str(cwd), delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    logger.debug("Running temp script: %s", tmp_path)
    result = run_shell(f"python {tmp_path}", cwd=cwd, timeout=timeout)

    # Clean up temp file
    try:
        Path(tmp_path).unlink()
    except OSError:
        pass

    return result


def spawn_agent(task: str, *, cwd: str | Path = ".") -> dict[str, str | int]:
    """Spawn a background sub-task (runs as a detached subprocess)."""
    logger.info("Spawning agent for: %s", task)
    return run_shell(task, cwd=cwd, timeout=60)
