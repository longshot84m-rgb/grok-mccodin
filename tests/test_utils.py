"""Tests for grok_mccodin.utils."""

from __future__ import annotations

import json
from pathlib import Path

from grok_mccodin.utils import file_hash, index_folder, log_receipt, read_file_safe


class TestIndexFolder:
    def test_indexes_files(self, tmp_project):
        idx = index_folder(tmp_project)
        assert "main.py" in idx
        assert "utils.py" in idx
        assert "src/" in idx

    def test_respects_max_depth(self, tmp_project):
        idx = index_folder(tmp_project, max_depth=0)
        # At depth 0 we should still see top-level files
        assert "main.py" in idx

    def test_nonexistent_dir(self):
        idx = index_folder("/nonexistent/path")
        assert "not a directory" in idx


class TestReadFileSafe:
    def test_reads_existing(self, tmp_project):
        content = read_file_safe(tmp_project / "main.py")
        assert 'print("hello")' in content

    def test_missing_file(self):
        content = read_file_safe("/does/not/exist.py")
        assert "file not found" in content

    def test_truncation(self, tmp_path):
        big = tmp_path / "big.txt"
        big.write_text("\n".join(f"line {i}" for i in range(1000)))
        content = read_file_safe(big, max_lines=10)
        assert "truncated" in content


class TestLogReceipt:
    def test_creates_log(self, tmp_path):
        log_file = tmp_path / "test_log.json"
        log_receipt(log_file, action="test_action", detail="some detail")
        data = json.loads(log_file.read_text())
        assert len(data) == 1
        assert data[0]["action"] == "test_action"

    def test_appends_to_log(self, tmp_path):
        log_file = tmp_path / "test_log.json"
        log_receipt(log_file, action="first")
        log_receipt(log_file, action="second")
        data = json.loads(log_file.read_text())
        assert len(data) == 2


class TestFileHash:
    def test_deterministic(self, tmp_project):
        h1 = file_hash(tmp_project / "main.py")
        h2 = file_hash(tmp_project / "main.py")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_different_files(self, tmp_project):
        assert file_hash(tmp_project / "main.py") != file_hash(tmp_project / "utils.py")
