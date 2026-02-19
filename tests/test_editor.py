"""Tests for grok_mccodin.editor."""

from __future__ import annotations

from grok_mccodin.editor import (
    _safe_resolve,
    apply_create,
    apply_delete,
    apply_edit,
    extract_code_blocks,
    extract_commands,
    extract_creates,
    extract_deletes,
    show_diff,
)


class TestSafeResolve:
    def test_normal_path(self, tmp_path):
        assert _safe_resolve("foo.py", tmp_path) == (tmp_path / "foo.py").resolve()

    def test_nested_path(self, tmp_path):
        assert _safe_resolve("src/bar.py", tmp_path) is not None

    def test_blocks_traversal(self, tmp_path):
        assert _safe_resolve("../../etc/passwd", tmp_path) is None

    def test_blocks_absolute_escape(self, tmp_path):
        assert _safe_resolve("/etc/passwd", tmp_path) is None


class TestExtractCodeBlocks:
    def test_basic_block(self):
        text = "Here is code:\n```python\nprint('hi')\n```"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0]["lang"] == "python"
        assert "print('hi')" in blocks[0]["code"]

    def test_block_with_filename(self):
        text = "```python:src/app.py\nprint('hi')\n```"
        blocks = extract_code_blocks(text)
        assert blocks[0]["filename"] == "src/app.py"

    def test_multiple_blocks(self):
        text = "```python\na\n```\ntext\n```js\nb\n```"
        assert len(extract_code_blocks(text)) == 2

    def test_no_blocks(self):
        assert extract_code_blocks("no code here") == []


class TestExtractCommands:
    def test_single_command(self):
        text = "I'll run:\nRUN: pytest -v\nDone."
        cmds = extract_commands(text)
        assert cmds == ["pytest -v"]

    def test_multiple_commands(self):
        text = "RUN: ls\nRUN: pwd"
        assert len(extract_commands(text)) == 2


class TestExtractCreates:
    def test_create_directive(self):
        text = 'CREATE: src/new.py\n```python\nprint("new")\n```'
        creates = extract_creates(text)
        assert len(creates) == 1
        assert creates[0]["path"] == "src/new.py"


class TestExtractDeletes:
    def test_delete_directive(self):
        text = "DELETE: old_file.py\nDone."
        deletes = extract_deletes(text)
        assert deletes == ["old_file.py"]


class TestShowDiff:
    def test_shows_changes(self):
        diff = show_diff("line1\nline2\n", "line1\nline2_changed\n", "test.py")
        assert "---" in diff
        assert "+++" in diff

    def test_no_changes(self):
        diff = show_diff("same\n", "same\n")
        assert diff == ""


class TestApplyEdit:
    def test_creates_new_file(self, tmp_path):
        apply_edit("new.py", "content", base_dir=tmp_path)
        assert (tmp_path / "new.py").read_text() == "content"

    def test_overwrites_existing(self, tmp_path):
        (tmp_path / "f.py").write_text("old")
        apply_edit("f.py", "new", base_dir=tmp_path)
        assert (tmp_path / "f.py").read_text() == "new"

    def test_blocks_path_traversal(self, tmp_path):
        result = apply_edit("../../evil.py", "pwned", base_dir=tmp_path)
        assert "blocked" in result.lower()

    def test_no_changes_returns_message(self, tmp_path):
        (tmp_path / "same.py").write_text("content")
        result = apply_edit("same.py", "content", base_dir=tmp_path)
        assert "No changes" in result


class TestApplyCreate:
    def test_creates_file(self, tmp_path):
        apply_create("dir/f.py", "data", base_dir=tmp_path)
        assert (tmp_path / "dir" / "f.py").read_text() == "data"

    def test_skip_existing(self, tmp_path):
        (tmp_path / "f.py").write_text("old")
        result = apply_create("f.py", "new", base_dir=tmp_path)
        assert "skip" in result
        assert (tmp_path / "f.py").read_text() == "old"

    def test_blocks_path_traversal(self, tmp_path):
        result = apply_create("../../evil.py", "pwned", base_dir=tmp_path)
        assert "blocked" in result.lower()


class TestApplyDelete:
    def test_deletes_to_trash(self, tmp_path):
        (tmp_path / "doomed.py").write_text("bye")
        apply_delete("doomed.py", base_dir=tmp_path)
        assert not (tmp_path / "doomed.py").exists()
        # Trash file has timestamp prefix now
        trash_files = list((tmp_path / ".trash").iterdir())
        assert len(trash_files) == 1
        assert "doomed.py" in trash_files[0].name

    def test_skip_missing(self, tmp_path):
        result = apply_delete("nope.py", base_dir=tmp_path)
        assert "skip" in result

    def test_blocks_path_traversal(self, tmp_path):
        result = apply_delete("../../etc/passwd", base_dir=tmp_path)
        assert "blocked" in result.lower()
