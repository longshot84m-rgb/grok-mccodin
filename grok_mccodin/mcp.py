"""MCP (Model Context Protocol) client — connect to any MCP server via stdio."""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MCPError(Exception):
    """Raised when an MCP operation fails."""


class MCPClient:
    """Lightweight MCP client using JSON-RPC 2.0 over stdio transport.

    Usage::

        client = MCPClient("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
        client.start()
        tools = client.list_tools()
        result = client.call_tool("read_file", {"path": "/tmp/hello.txt"})
        client.stop()
    """

    def __init__(self, command: str, args: list[str] | None = None) -> None:
        self.command = command
        self.args = args or []
        self._proc: subprocess.Popen[bytes] | None = None
        self._request_id = 0
        self._lock = threading.Lock()
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, timeout: int = 15) -> dict[str, Any]:
        """Launch the MCP server process and perform the initialize handshake."""
        cmd = [self.command, *self.args]
        logger.info("Starting MCP server: %s", " ".join(cmd))
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise MCPError(f"MCP server command not found: {self.command}") from exc
        except OSError as exc:
            raise MCPError(f"Failed to start MCP server: {exc}") from exc

        # Send initialize request
        result = self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "grok-mccodin", "version": "0.1.0"},
            },
            timeout=timeout,
        )

        # Send initialized notification (no response expected)
        self._notify("notifications/initialized", {})
        self._initialized = True
        logger.info("MCP server initialized: %s", result.get("serverInfo", {}))
        return result

    def stop(self) -> None:
        """Shut down the MCP server process."""
        if self._proc and self._proc.poll() is None:
            try:
                self._notify("notifications/cancelled", {"reason": "client shutdown"})
            except MCPError:
                pass
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        self._initialized = False
        logger.info("MCP server stopped")

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ------------------------------------------------------------------
    # MCP Protocol Methods
    # ------------------------------------------------------------------

    def list_tools(self) -> list[dict[str, Any]]:
        """Discover available tools from the server."""
        self._ensure_running()
        result = self._request("tools/list", {})
        tools: list[dict[str, Any]] = result.get("tools", [])
        return tools

    def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None, timeout: int = 30
    ) -> list[dict[str, Any]]:
        """Invoke a tool on the MCP server.

        Returns a list of content blocks (text, image, resource).
        """
        self._ensure_running()
        result = self._request(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
            timeout=timeout,
        )
        content: list[dict[str, Any]] = result.get("content", [])
        return content

    def list_resources(self) -> list[dict[str, Any]]:
        """List available resources from the server."""
        self._ensure_running()
        result = self._request("resources/list", {})
        resources: list[dict[str, Any]] = result.get("resources", [])
        return resources

    def read_resource(self, uri: str) -> list[dict[str, Any]]:
        """Read a specific resource by URI."""
        self._ensure_running()
        result = self._request("resources/read", {"uri": uri})
        contents: list[dict[str, Any]] = result.get("contents", [])
        return contents

    def list_prompts(self) -> list[dict[str, Any]]:
        """List available prompt templates."""
        self._ensure_running()
        result = self._request("prompts/list", {})
        prompts: list[dict[str, Any]] = result.get("prompts", [])
        return prompts

    def get_prompt(self, name: str, arguments: dict[str, str] | None = None) -> dict[str, Any]:
        """Retrieve a prompt template with optional arguments."""
        self._ensure_running()
        prompt: dict[str, Any] = self._request(
            "prompts/get", {"name": name, "arguments": arguments or {}}
        )
        return prompt

    # ------------------------------------------------------------------
    # JSON-RPC 2.0 transport
    # ------------------------------------------------------------------

    def _ensure_running(self) -> None:
        if not self.is_running:
            raise MCPError("MCP server is not running. Call start() first.")

    def _next_id(self) -> int:
        with self._lock:
            self._request_id += 1
            return self._request_id

    def _send(self, message: dict[str, Any]) -> None:
        """Write a JSON-RPC message to the server's stdin."""
        if not self._proc or not self._proc.stdin:
            raise MCPError("MCP server stdin not available")
        raw = json.dumps(message) + "\n"
        self._proc.stdin.write(raw.encode("utf-8"))
        self._proc.stdin.flush()
        logger.debug("MCP send: %s", message.get("method", "response"))

    def _recv(self, timeout: int = 15) -> dict[str, Any]:
        """Read a JSON-RPC response from the server's stdout.

        Skips notification messages and returns the first result/error.
        """
        if not self._proc or not self._proc.stdout:
            raise MCPError("MCP server stdout not available")

        deadline_lines = 100  # Safety: don't read forever
        for _ in range(deadline_lines):
            line = self._proc.stdout.readline()
            if not line:
                stderr_out = ""
                if self._proc.stderr:
                    stderr_out = self._proc.stderr.read().decode("utf-8", errors="replace")[:500]
                raise MCPError(f"MCP server closed stdout. stderr: {stderr_out}")

            line_str = line.decode("utf-8", errors="replace").strip()
            if not line_str:
                continue

            try:
                msg: dict[str, Any] = json.loads(line_str)
            except json.JSONDecodeError:
                logger.debug("MCP non-JSON line: %s", line_str[:200])
                continue

            # Skip notifications (no "id" field)
            if "id" not in msg:
                logger.debug("MCP notification: %s", msg.get("method", "unknown"))
                continue

            return msg

        raise MCPError("MCP: exceeded max read attempts without a response")

    def _request(self, method: str, params: dict[str, Any], timeout: int = 15) -> dict[str, Any]:
        """Send a JSON-RPC request and wait for the response."""
        req_id = self._next_id()
        message = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        self._send(message)
        response = self._recv(timeout=timeout)

        if "error" in response:
            err = response["error"]
            raise MCPError(f"MCP error {err.get('code', '?')}: {err.get('message', 'unknown')}")

        result: dict[str, Any] = response.get("result", {})
        return result

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        self._send(message)


# ---------------------------------------------------------------------------
# MCP Server Registry — manages multiple named servers
# ---------------------------------------------------------------------------


class MCPRegistry:
    """Manages a collection of named MCP server connections.

    Configured via a JSON file or dict::

        {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
            },
            "github": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {"GITHUB_TOKEN": "..."}
            }
        }
    """

    def __init__(self) -> None:
        self._servers: dict[str, dict[str, Any]] = {}
        self._clients: dict[str, MCPClient] = {}

    def load_config(self, config_path: str | Path) -> None:
        """Load MCP server configurations from a JSON file."""
        path = Path(config_path)
        if not path.is_file():
            logger.info("No MCP config found at %s", path)
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._servers = data
                logger.info("Loaded %d MCP server configs", len(data))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load MCP config: %s", exc)

    def load_dict(self, servers: dict[str, dict[str, Any]]) -> None:
        """Load MCP server configurations from a dict."""
        self._servers = servers

    @property
    def server_names(self) -> list[str]:
        return list(self._servers.keys())

    def connect(self, name: str, timeout: int = 15) -> MCPClient:
        """Start and connect to a named MCP server."""
        if name in self._clients and self._clients[name].is_running:
            return self._clients[name]

        if name not in self._servers:
            raise MCPError(f"Unknown MCP server: {name}. Available: {self.server_names}")

        cfg = self._servers[name]
        client = MCPClient(cfg["command"], cfg.get("args", []))
        client.start(timeout=timeout)
        self._clients[name] = client
        return client

    def get_client(self, name: str) -> MCPClient | None:
        """Get an already-connected client, or None."""
        client = self._clients.get(name)
        if client and client.is_running:
            return client
        return None

    def disconnect(self, name: str) -> None:
        """Stop a specific MCP server."""
        client = self._clients.pop(name, None)
        if client:
            client.stop()

    def disconnect_all(self) -> None:
        """Stop all MCP servers."""
        for name in list(self._clients.keys()):
            self.disconnect(name)

    def list_all_tools(self) -> dict[str, list[dict[str, Any]]]:
        """List tools from all connected servers."""
        result: dict[str, list[dict[str, Any]]] = {}
        for name, client in self._clients.items():
            if client.is_running:
                try:
                    result[name] = client.list_tools()
                except MCPError as exc:
                    logger.error("Failed to list tools from %s: %s", name, exc)
                    result[name] = []
        return result
