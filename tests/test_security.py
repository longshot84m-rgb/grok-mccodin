"""Tests for security fixes — path traversal, SSRF, SQL injection, etc."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from grok_mccodin.database import DatabaseError, SQLiteDB
from grok_mccodin.docker import DockerError, _validate_volume
from grok_mccodin.editor import _safe_resolve
from grok_mccodin.git import GitError, clone
from grok_mccodin.mcp import MCPRegistry, _validate_mcp_configs
from grok_mccodin.rag import TFIDFIndex
from grok_mccodin.web import _is_safe_url, web_fetch, web_fetch_raw

# ---------------------------------------------------------------------------
# 1. Path traversal: _safe_resolve()
# ---------------------------------------------------------------------------


class TestSafeResolve:
    def test_normal_path(self, tmp_path):
        (tmp_path / "file.txt").write_text("ok")
        result = _safe_resolve("file.txt", tmp_path)
        assert result is not None
        assert result == tmp_path / "file.txt"

    def test_subdirectory(self, tmp_path):
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "main.py").write_text("ok")
        result = _safe_resolve("src/main.py", tmp_path)
        assert result is not None

    def test_traversal_blocked(self, tmp_path):
        result = _safe_resolve("../../etc/passwd", tmp_path)
        assert result is None

    def test_double_traversal_blocked(self, tmp_path):
        result = _safe_resolve("../../../windows/system32/config", tmp_path)
        assert result is None

    def test_absolute_path_outside_blocked(self, tmp_path):
        result = _safe_resolve("/etc/passwd", tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# 2. SSRF protection: _is_safe_url()
# ---------------------------------------------------------------------------


class TestSSRFProtection:
    def test_https_allowed(self):
        assert _is_safe_url("https://example.com") == ""

    def test_http_allowed(self):
        assert _is_safe_url("http://example.com") == ""

    def test_file_scheme_blocked(self):
        err = _is_safe_url("file:///etc/passwd")
        assert "Blocked URL scheme" in err

    def test_ftp_scheme_blocked(self):
        err = _is_safe_url("ftp://evil.com/data")
        assert "Blocked URL scheme" in err

    def test_localhost_blocked(self):
        err = _is_safe_url("http://localhost:8080/admin")
        assert "Blocked hostname" in err

    def test_127_0_0_1_blocked(self):
        err = _is_safe_url("http://127.0.0.1:9200/_cat/indices")
        assert "private" in err.lower() or "Blocked" in err

    def test_10_x_blocked(self):
        err = _is_safe_url("http://10.0.0.1/internal")
        assert "private" in err.lower() or "Blocked" in err

    def test_172_16_blocked(self):
        err = _is_safe_url("http://172.16.0.1/secret")
        assert "private" in err.lower() or "Blocked" in err

    def test_192_168_blocked(self):
        err = _is_safe_url("http://192.168.1.1/router")
        assert "private" in err.lower() or "Blocked" in err

    def test_169_254_blocked(self):
        err = _is_safe_url("http://169.254.169.254/latest/meta-data/")
        assert "private" in err.lower() or "Blocked" in err

    def test_0_0_0_0_blocked(self):
        err = _is_safe_url("http://0.0.0.0:8080/")
        assert "Blocked" in err

    def test_web_fetch_blocks_ssrf(self):
        result = web_fetch("file:///etc/passwd")
        assert result["error"] != ""
        assert result["text"] == ""

    def test_web_fetch_blocks_private_ip(self):
        result = web_fetch("http://127.0.0.1:9200/")
        assert result["error"] != ""

    def test_web_fetch_raw_blocks_ssrf(self):
        with pytest.raises(Exception):
            web_fetch_raw("file:///etc/passwd")


# ---------------------------------------------------------------------------
# 3. SQL injection: table_info()
# ---------------------------------------------------------------------------


class TestTableInfoInjection:
    def test_normal_table_name(self, tmp_path):
        db_path = tmp_path / "test.db"
        with SQLiteDB(db_path) as db:
            db.execute("CREATE TABLE users (id INTEGER, name TEXT)")
            info = db.table_info("users")
            assert len(info) == 2

    def test_injection_attempt_blocked(self, tmp_path):
        db_path = tmp_path / "test.db"
        with SQLiteDB(db_path) as db:
            db.execute("CREATE TABLE users (id INTEGER)")
            with pytest.raises(DatabaseError):
                db.table_info("users'); DROP TABLE users;--")

    def test_nonexistent_table(self, tmp_path):
        db_path = tmp_path / "test.db"
        with SQLiteDB(db_path) as db:
            db.execute("CREATE TABLE real (id INTEGER)")
            with pytest.raises(DatabaseError, match="not found"):
                db.table_info("fake")


# ---------------------------------------------------------------------------
# 4. Symlink protection in RAG walker
# ---------------------------------------------------------------------------


class TestRAGSymlinkProtection:
    @pytest.mark.skipif(os.name == "nt", reason="symlinks need admin on Windows")
    def test_symlink_skipped(self, tmp_path):
        # Create a real directory with a code file
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        (real_dir / "code.py").write_text("def hello(): pass\n")

        # Create an external directory with sensitive file
        external = tmp_path / "external"
        external.mkdir()
        (external / "secret.py").write_text("SECRET_KEY = 'password123'\n")

        # Create project dir with symlink to external
        project = tmp_path / "project"
        project.mkdir()
        (project / "app.py").write_text("def main(): pass\n")
        (project / "linked").symlink_to(external)

        index = TFIDFIndex()
        index.index_folder(project)

        # Should find app.py but NOT secret.py
        paths = [doc["path"] for doc in index._documents]
        assert any("app.py" in p for p in paths)
        assert not any("secret.py" in p for p in paths)


# ---------------------------------------------------------------------------
# 5. Git clone URL validation
# ---------------------------------------------------------------------------


class TestGitCloneValidation:
    def test_file_scheme_blocked(self):
        with pytest.raises(GitError, match="Blocked clone URL"):
            clone("file:///tmp/evil-repo")

    def test_data_scheme_blocked(self):
        with pytest.raises(GitError, match="Blocked clone URL"):
            clone("data:text/plain,evil")

    def test_javascript_scheme_blocked(self):
        with pytest.raises(GitError, match="Blocked clone URL"):
            clone("javascript:alert(1)")

    @patch("grok_mccodin.git.subprocess.run")
    def test_https_allowed(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        clone("https://github.com/user/repo.git")
        mock_run.assert_called_once()

    @patch("grok_mccodin.git.subprocess.run")
    def test_ssh_allowed(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        clone("ssh://git@github.com/user/repo.git")
        mock_run.assert_called_once()

    @patch("grok_mccodin.git.subprocess.run")
    def test_git_at_allowed(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        clone("git@github.com:user/repo.git")
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# 6. Docker volume mount validation
# ---------------------------------------------------------------------------


class TestDockerVolumeValidation:
    def test_named_volume_allowed(self):
        # Should not raise
        _validate_volume("mydata:/app/data")

    def test_root_mount_blocked(self):
        with pytest.raises(DockerError, match="Blocked"):
            _validate_volume("/:/container")

    def test_etc_mount_blocked(self):
        with pytest.raises(DockerError, match="Blocked"):
            _validate_volume("/etc:/container/etc")

    def test_deep_path_allowed(self):
        # Paths 3+ levels deep are allowed
        _validate_volume("/home/user/project:/app")

    def test_var_mount_blocked(self):
        with pytest.raises(DockerError, match="Blocked"):
            _validate_volume("/var:/container/var")


# ---------------------------------------------------------------------------
# 7. MCP config validation
# ---------------------------------------------------------------------------


class TestMCPConfigValidation:
    def test_valid_config(self):
        data = {"test": {"command": "npx", "args": ["-y", "server"]}}
        result = _validate_mcp_configs(data)
        assert "test" in result

    def test_missing_command(self):
        data = {"bad": {"args": ["-y"]}}
        result = _validate_mcp_configs(data)
        assert "bad" not in result

    def test_non_string_command(self):
        data = {"bad": {"command": 123, "args": []}}
        result = _validate_mcp_configs(data)
        assert "bad" not in result

    def test_non_list_args(self):
        data = {"bad": {"command": "npx", "args": "not-a-list"}}
        result = _validate_mcp_configs(data)
        assert "bad" not in result

    def test_non_dict_entry(self):
        data = {"bad": "not-a-dict"}
        result = _validate_mcp_configs(data)
        assert "bad" not in result

    def test_traversal_command_blocked(self):
        data = {"bad": {"command": "../../malicious", "args": []}}
        result = _validate_mcp_configs(data)
        assert "bad" not in result

    def test_load_config_validates(self, tmp_path):
        config = {
            "good": {"command": "npx", "args": ["-y", "server"]},
            "bad": {"args": ["missing-command"]},
        }
        config_path = tmp_path / "mcp_servers.json"
        config_path.write_text(json.dumps(config))

        reg = MCPRegistry()
        reg.load_config(config_path)
        assert "good" in reg.server_names
        assert "bad" not in reg.server_names

    def test_load_dict_validates(self):
        reg = MCPRegistry()
        reg.load_dict(
            {
                "valid": {"command": "echo", "args": ["hello"]},
                "invalid": {"not-command": True},
            }
        )
        assert "valid" in reg.server_names
        assert "invalid" not in reg.server_names


# ---------------------------------------------------------------------------
# 8. Dispatch table existence (regression test)
# ---------------------------------------------------------------------------


class TestDispatchTable:
    def test_dispatch_has_all_commands(self):
        from grok_mccodin.main import _SLASH_DISPATCH, SLASH_COMMANDS

        for cmd_key in SLASH_COMMANDS:
            # Extract the base command (e.g. "/git" from "/git [cmd]")
            base_cmd = cmd_key.split()[0]
            assert base_cmd in _SLASH_DISPATCH, f"Missing dispatch for {base_cmd}"

    def test_dispatch_returns_callable(self):
        from grok_mccodin.main import _SLASH_DISPATCH

        for cmd, handler in _SLASH_DISPATCH.items():
            assert callable(handler), f"{cmd} handler is not callable"
