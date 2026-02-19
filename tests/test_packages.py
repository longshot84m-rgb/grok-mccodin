"""Tests for grok_mccodin.packages."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from grok_mccodin.packages import (
    PackageError,
    _run_cmd,
    detect_package_manager,
    pip_list,
)


class TestRunCmd:
    @patch("grok_mccodin.packages.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        assert _run_cmd(["echo", "ok"]) == "ok"

    @patch("grok_mccodin.packages.subprocess.run")
    def test_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        with pytest.raises(PackageError, match="failed"):
            _run_cmd(["bad_cmd"])

    @patch("grok_mccodin.packages.subprocess.run")
    def test_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        with pytest.raises(PackageError, match="not found"):
            _run_cmd(["nonexistent"])

    @patch("grok_mccodin.packages.subprocess.run")
    def test_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="pip", timeout=120)
        with pytest.raises(PackageError, match="timed out"):
            _run_cmd(["pip", "install", "big-package"])


class TestPipList:
    @patch("grok_mccodin.packages.subprocess.run")
    def test_returns_packages(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='[{"name": "requests", "version": "2.31.0"}]',
            stderr="",
        )
        pkgs = pip_list()
        assert len(pkgs) == 1
        assert pkgs[0]["name"] == "requests"

    @patch("grok_mccodin.packages.subprocess.run")
    def test_empty_list(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="[]", stderr="")
        assert pip_list() == []


class TestDetectPackageManager:
    def test_pip_from_pyproject(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[tool.poetry]\nname = 'test'\n")
        assert detect_package_manager(tmp_path) == "pip"

    def test_npm_from_package_json(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "test"}')
        assert detect_package_manager(tmp_path) == "npm"

    def test_both(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[tool.poetry]\n")
        (tmp_path / "package.json").write_text('{"name": "test"}')
        result = detect_package_manager(tmp_path)
        assert "npm" in result
        assert "pip" in result

    def test_unknown(self, tmp_path):
        assert detect_package_manager(tmp_path) == "unknown"
