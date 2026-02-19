"""Tests for grok_mccodin.git."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from grok_mccodin.git import (
    GitError,
    _run_git,
    current_branch,
    is_git_repo,
    status,
    summary,
)


class TestRunGit:
    @patch("grok_mccodin.git.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok\n", stderr="")
        result = _run_git(["status"])
        assert result == "ok\n"

    @patch("grok_mccodin.git.subprocess.run")
    def test_nonzero_exit(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="fatal: not a repo")
        with pytest.raises(GitError, match="not a repo"):
            _run_git(["status"])

    @patch("grok_mccodin.git.subprocess.run")
    def test_git_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        with pytest.raises(GitError, match="not installed"):
            _run_git(["status"])

    @patch("grok_mccodin.git.subprocess.run")
    def test_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=30)
        with pytest.raises(GitError, match="timed out"):
            _run_git(["log"])


class TestStatus:
    @patch("grok_mccodin.git.subprocess.run")
    def test_status(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="## main\n M file.py\n", stderr="")
        result = status("/tmp")
        assert "file.py" in result


class TestIsGitRepo:
    @patch("grok_mccodin.git.subprocess.run")
    def test_is_repo(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="true\n", stderr="")
        assert is_git_repo("/tmp") is True

    @patch("grok_mccodin.git.subprocess.run")
    def test_not_repo(self, mock_run):
        mock_run.return_value = MagicMock(returncode=128, stdout="", stderr="fatal")
        assert is_git_repo("/tmp") is False


class TestCurrentBranch:
    @patch("grok_mccodin.git.subprocess.run")
    def test_branch_name(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="feature-xyz\n", stderr="")
        assert current_branch("/tmp") == "feature-xyz"


class TestSummary:
    @patch("grok_mccodin.git.subprocess.run")
    def test_summary_not_repo(self, mock_run):
        mock_run.return_value = MagicMock(returncode=128, stdout="", stderr="fatal")
        result = summary("/tmp")
        assert "not a git repository" in result

    @patch("grok_mccodin.git.subprocess.run")
    def test_summary_with_data(self, mock_run):
        # current_branch call, then status, then log
        mock_run.return_value = MagicMock(returncode=0, stdout="main\n", stderr="")
        result = summary("/tmp")
        assert "Branch: main" in result
