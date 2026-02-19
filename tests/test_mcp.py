"""Tests for grok_mccodin.mcp."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from grok_mccodin.mcp import MCPClient, MCPError, MCPRegistry


class TestMCPClient:
    def test_init(self):
        client = MCPClient("npx", ["-y", "some-server"])
        assert client.command == "npx"
        assert client.args == ["-y", "some-server"]
        assert not client.is_running

    def test_not_running(self):
        client = MCPClient("echo")
        with pytest.raises(MCPError, match="not running"):
            client.list_tools()

    @patch("grok_mccodin.mcp.subprocess.Popen")
    def test_start_command_not_found(self, mock_popen):
        mock_popen.side_effect = FileNotFoundError()
        client = MCPClient("nonexistent-server")
        with pytest.raises(MCPError, match="not found"):
            client.start()

    def test_stop_when_not_started(self):
        client = MCPClient("echo")
        # Should not raise
        client.stop()
        assert not client.is_running


class TestMCPRegistry:
    def test_empty_registry(self):
        reg = MCPRegistry()
        assert reg.server_names == []

    def test_load_dict(self):
        reg = MCPRegistry()
        reg.load_dict(
            {
                "fs": {"command": "npx", "args": ["-y", "server-fs"]},
                "github": {"command": "npx", "args": ["-y", "server-github"]},
            }
        )
        assert "fs" in reg.server_names
        assert "github" in reg.server_names

    def test_connect_unknown(self):
        reg = MCPRegistry()
        with pytest.raises(MCPError, match="Unknown MCP server"):
            reg.connect("nonexistent")

    def test_load_config_missing_file(self, tmp_path):
        reg = MCPRegistry()
        reg.load_config(tmp_path / "does_not_exist.json")
        assert reg.server_names == []

    def test_load_config_valid(self, tmp_path):
        config = {"test": {"command": "echo", "args": ["hello"]}}
        config_path = tmp_path / "mcp_servers.json"
        config_path.write_text(json.dumps(config))

        reg = MCPRegistry()
        reg.load_config(config_path)
        assert "test" in reg.server_names

    def test_load_config_invalid_json(self, tmp_path):
        config_path = tmp_path / "mcp_servers.json"
        config_path.write_text("not valid json {{{")

        reg = MCPRegistry()
        reg.load_config(config_path)
        assert reg.server_names == []

    def test_get_client_not_connected(self):
        reg = MCPRegistry()
        assert reg.get_client("anything") is None

    def test_disconnect_nonexistent(self):
        reg = MCPRegistry()
        # Should not raise
        reg.disconnect("nonexistent")

    def test_list_all_tools_empty(self):
        reg = MCPRegistry()
        assert reg.list_all_tools() == {}

    def test_disconnect_all_empty(self):
        reg = MCPRegistry()
        # Should not raise
        reg.disconnect_all()
