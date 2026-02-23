"""Tests for grok_mccodin.executor."""

from __future__ import annotations

from grok_mccodin.executor import is_safe, run_shell


class TestIsSafe:
    def test_allows_normal_commands(self):
        assert is_safe("ls -la")
        assert is_safe("python script.py")
        assert is_safe("pytest -v")

    def test_blocks_dangerous(self):
        assert not is_safe("rm -rf /")
        assert not is_safe("rm -rf /*")
        assert not is_safe("mkfs.ext4 /dev/sda")


class TestRunShell:
    def test_runs_echo(self):
        result = run_shell("echo hello", confirm=False)
        assert result["returncode"] == 0
        assert "hello" in result["stdout"]

    def test_safe_lock_blocks(self):
        result = run_shell("echo hi", safe_lock=True, confirm=False)
        assert result["returncode"] == -1
        assert "Safe Lock" in result["stderr"]

    def test_timeout(self):
        # Use python -c for cross-platform sleep
        result = run_shell(
            'python -c "import time; time.sleep(10)"',
            timeout=1,
            confirm=False,
        )
        assert result["returncode"] == -1
        assert "TIMEOUT" in result["stderr"]

    def test_nonexistent_command(self):
        result = run_shell("this_command_does_not_exist_12345", confirm=False)
        assert result["returncode"] != 0

    def test_blocked_command(self):
        result = run_shell("rm -rf /", confirm=False)
        assert result["returncode"] == -1
        assert "BLOCKED" in result["stderr"]
