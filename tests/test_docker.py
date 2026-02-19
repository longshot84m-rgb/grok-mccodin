"""Tests for grok_mccodin.docker."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from grok_mccodin.docker import (
    DockerError,
    _run_docker,
    is_docker_available,
    ps,
    ps_json,
)


class TestRunDocker:
    @patch("grok_mccodin.docker.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="container list", stderr="")
        result = _run_docker(["ps"])
        assert result == "container list"

    @patch("grok_mccodin.docker.subprocess.run")
    def test_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="daemon not running")
        with pytest.raises(DockerError, match="daemon not running"):
            _run_docker(["ps"])

    @patch("grok_mccodin.docker.subprocess.run")
    def test_not_installed(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        with pytest.raises(DockerError, match="not installed"):
            _run_docker(["ps"])

    @patch("grok_mccodin.docker.subprocess.run")
    def test_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker", timeout=60)
        with pytest.raises(DockerError, match="timed out"):
            _run_docker(["build", "."])


class TestPs:
    @patch("grok_mccodin.docker.subprocess.run")
    def test_ps(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="CONTAINER ID\tIMAGE\tSTATUS\tNAMES\nabc123\tnginx\tUp\tweb\n",
            stderr="",
        )
        result = ps()
        assert "nginx" in result


class TestPsJson:
    @patch("grok_mccodin.docker.subprocess.run")
    def test_ps_json(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"ID":"abc123","Image":"nginx","Status":"Up","Names":"web"}\n',
            stderr="",
        )
        result = ps_json()
        assert len(result) == 1
        assert result[0]["Image"] == "nginx"

    @patch("grok_mccodin.docker.subprocess.run")
    def test_ps_json_empty(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="\n", stderr="")
        result = ps_json()
        assert result == []


class TestIsDockerAvailable:
    @patch("grok_mccodin.docker.subprocess.run")
    def test_available(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        assert is_docker_available() is True

    @patch("grok_mccodin.docker.subprocess.run")
    def test_not_available(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        assert is_docker_available() is False
