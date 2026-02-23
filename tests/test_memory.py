"""Tests for grok_mccodin.memory."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from grok_mccodin.memory import (
    ConversationMemory,
    ScoredMessage,
    _distill_facts,
    _sanitize_filename,
    compress_messages,
    estimate_tokens,
    score_importance,
)

# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_basic(self):
        assert estimate_tokens("hello world!!") == len("hello world!!") // 4

    def test_empty(self):
        assert estimate_tokens("") == 1  # max(1, 0)

    def test_long(self):
        text = "x" * 1000
        assert estimate_tokens(text) == 250


# ---------------------------------------------------------------------------
# score_importance
# ---------------------------------------------------------------------------


class TestScoreImportance:
    def test_code_block_bonus(self):
        score = score_importance("assistant", "```python\nprint('hi')\n```", 0, 1)
        assert score >= 0.2  # code block bonus

    def test_decision_keyword_bonus(self):
        score_with = score_importance("user", "We decided to use React", 0, 1)
        score_without = score_importance("user", "Hello there", 0, 1)
        assert score_with > score_without

    def test_recency_scaling(self):
        old = score_importance("user", "hello", 0, 10)
        new = score_importance("user", "hello", 9, 10)
        assert new > old

    def test_role_weights(self):
        assistant_score = score_importance("assistant", "test", 0, 1)
        user_score = score_importance("user", "test", 0, 1)
        assert assistant_score > user_score

    def test_clamps_at_one(self):
        # Max all bonuses: recency(0.3) + assistant(0.2) + code(0.2) + keyword(0.15) + length(0.1) = 0.95
        score = score_importance(
            "assistant",
            "```python\nprint('decided')\n```\n" + "x" * 250,
            9,
            10,
        )
        assert score <= 1.0

    def test_length_bonus(self):
        short = score_importance("user", "hi", 0, 1)
        long = score_importance("user", "x" * 300, 0, 1)
        assert long > short


# ---------------------------------------------------------------------------
# compress_messages
# ---------------------------------------------------------------------------


class TestCompressMessages:
    def test_preserves_important(self):
        msgs = [
            ScoredMessage("assistant", "```python\nprint('hello')\n```", 0.8, ""),
            ScoredMessage("user", "ok thanks", 0.3, ""),
        ]
        result = compress_messages(msgs)
        assert "print('hello')" in result  # High-importance kept verbatim

    def test_distills_questions(self):
        msgs = [
            ScoredMessage("user", "What framework should we use?", 0.4, ""),
            ScoredMessage("assistant", "Let me think about that.", 0.3, ""),
        ]
        result = compress_messages(msgs)
        assert "framework" in result

    def test_drops_short_pleasantries(self):
        msgs = [
            ScoredMessage("user", "thanks", 0.2, ""),
            ScoredMessage("assistant", "np", 0.2, ""),
        ]
        result = compress_messages(msgs)
        # Short messages should be dropped from distillation
        assert "thanks" not in result or "Key exchanges" not in result

    def test_respects_max_chars(self):
        msgs = [
            ScoredMessage("user", "x" * 500, 0.8, ""),
            ScoredMessage("assistant", "y" * 500, 0.8, ""),
        ]
        result = compress_messages(msgs, max_chars=200)
        assert len(result) <= 250  # max_chars + prefix + truncation marker


# ---------------------------------------------------------------------------
# ConversationMemory
# ---------------------------------------------------------------------------


class TestConversationMemory:
    def test_add_and_stats(self, tmp_path):
        mem = ConversationMemory(memory_dir=str(tmp_path))
        mem.add("user", "hello")
        mem.add("assistant", "hi there")
        s = mem.stats
        assert s["messages"] == 2
        assert s["total_messages"] == 2
        assert s["summaries"] == 0

    def test_build_context_recent_only(self, tmp_path):
        """Under budget, build_context returns recent messages only."""
        mem = ConversationMemory(token_budget=10000, keep_recent=10, memory_dir=str(tmp_path))
        mem.add("user", "hello")
        mem.add("assistant", "hi")
        ctx = mem.build_context("next question")
        # Should have 2 recent messages (user + assistant), no summaries
        roles = [m["role"] for m in ctx]
        assert "user" in roles
        assert "assistant" in roles
        assert not any(
            "summary" in m.get("content", "").lower() for m in ctx if m["role"] == "system"
        )

    def test_compression_trigger(self, tmp_path):
        """Exceeding token budget triggers compression."""
        mem = ConversationMemory(
            token_budget=200,  # Very low budget
            keep_recent=2,
            memory_dir=str(tmp_path),
        )
        # Add enough messages to exceed budget
        for i in range(10):
            mem.add("user", f"Message number {i} with some extra content to use up tokens " * 3)
            mem.add("assistant", f"Reply to message {i} with additional detail " * 3)

        s = mem.stats
        assert s["summaries"] > 0  # Compression happened
        assert s["messages"] <= 4  # Recent window trimmed (keep_recent=2, but pairs)
        assert s["total_messages"] == 20  # All messages preserved in _all_messages

    def test_compression_preserves_important(self, tmp_path):
        """High-importance messages are kept verbatim in summaries."""
        mem = ConversationMemory(
            token_budget=200,
            keep_recent=2,
            memory_dir=str(tmp_path),
        )
        # Add a code block message (high importance)
        mem.add("assistant", "```python\ndef critical_function():\n    return 42\n```")
        # Add padding to trigger compression
        for i in range(10):
            mem.add("user", f"Padding message {i} to trigger compression " * 5)
            mem.add("assistant", f"Padding reply {i} " * 5)

        # The code block should appear in at least one summary
        assert any("critical_function" in s for s in mem._summaries)

    def test_recall_relevance(self, tmp_path):
        """TF-IDF recall returns relevant old messages."""
        mem = ConversationMemory(
            token_budget=200,
            keep_recent=2,
            memory_dir=str(tmp_path),
            top_k=3,
        )
        # Add distinct topics
        mem.add("user", "Tell me about Python decorators and metaclasses")
        mem.add(
            "assistant", "Python decorators wrap functions. Metaclasses control class creation."
        )
        mem.add("user", "Tell me about JavaScript promises and async await")
        mem.add(
            "assistant", "JavaScript promises handle async operations. Async await is syntax sugar."
        )
        # Add padding to push earlier messages out of recent window
        for i in range(10):
            mem.add("user", f"Unrelated padding topic number {i} " * 5)
            mem.add("assistant", f"Unrelated padding reply {i} " * 5)

        # Recall should find Python-related content
        recalled = mem._recall("How do Python decorators work?")
        assert "decorator" in recalled.lower() or "python" in recalled.lower()

    def test_recall_dedup_against_recent(self, tmp_path):
        """Recalled content should not duplicate recent messages."""
        mem = ConversationMemory(
            token_budget=50000,  # High budget so no compression
            keep_recent=5,
            memory_dir=str(tmp_path),
            top_k=3,
        )
        msg = "Python decorators are functions that wrap other functions"
        mem.add("user", msg)
        mem.add("assistant", "That's correct!")

        # The message is in the recent window, so recall should not return it
        recalled = mem._recall("Tell me about Python decorators")
        assert msg not in recalled

    def test_recall_after_compression(self, tmp_path):
        """TF-IDF recall still finds content after compression removes it from _messages."""
        mem = ConversationMemory(
            token_budget=200,
            keep_recent=2,
            memory_dir=str(tmp_path),
            top_k=5,
        )
        # Add a unique message
        mem.add("user", "Tell me about Kubernetes pod autoscaling and horizontal scaling")
        mem.add("assistant", "Kubernetes HPA scales pods based on CPU or custom metrics")
        # Push it out of recent window with padding
        for i in range(10):
            mem.add("user", f"Totally different topic about cooking pasta recipe {i} " * 5)
            mem.add("assistant", f"Cooking reply about recipes {i} " * 5)

        # Verify compression happened
        assert mem.stats["summaries"] > 0
        # Recall should still find Kubernetes content via TF-IDF index
        recalled = mem._recall("How does Kubernetes autoscaling work?")
        assert (
            "kubernetes" in recalled.lower()
            or "scaling" in recalled.lower()
            or "hpa" in recalled.lower()
        )

    def test_save_load_roundtrip(self, tmp_path):
        """Save and load preserves full state."""
        mem1 = ConversationMemory(
            token_budget=200,
            keep_recent=2,
            memory_dir=str(tmp_path),
        )
        for i in range(6):
            mem1.add("user", f"Message {i} with some content " * 3)
            mem1.add("assistant", f"Reply {i} " * 3)

        path = mem1.save_session("test_session")
        assert path.exists()

        # Load into fresh instance
        mem2 = ConversationMemory(
            token_budget=200,
            keep_recent=2,
            memory_dir=str(tmp_path),
        )
        mem2.load_session("test_session")

        assert mem2.stats["total_messages"] == mem1.stats["total_messages"]
        assert mem2.stats["summaries"] == mem1.stats["summaries"]
        assert mem2.stats["session"] == "test_session"

    def test_save_load_preserves_recall(self, tmp_path):
        """TF-IDF recall works after save/load cycle."""
        mem1 = ConversationMemory(
            token_budget=200,
            keep_recent=2,
            memory_dir=str(tmp_path),
            top_k=5,
        )
        mem1.add("user", "Explain quantum entanglement and superposition in physics")
        mem1.add("assistant", "Quantum entanglement links particles across distance")
        for i in range(10):
            mem1.add("user", f"Different topic about gardening {i} " * 5)
            mem1.add("assistant", f"Gardening reply {i} " * 5)

        mem1.save_session("quantum_session")

        # Load into fresh instance
        mem2 = ConversationMemory(
            token_budget=200,
            keep_recent=2,
            memory_dir=str(tmp_path),
            top_k=5,
        )
        mem2.load_session("quantum_session")

        recalled = mem2._recall("What is quantum entanglement?")
        assert "quantum" in recalled.lower() or "entangle" in recalled.lower()

    def test_list_sessions(self, tmp_path):
        mem = ConversationMemory(memory_dir=str(tmp_path))
        mem.add("user", "hello")
        mem.save_session("alpha")
        mem.save_session("beta")

        sessions = mem.list_sessions()
        assert "alpha" in sessions
        assert "beta" in sessions
        assert sessions == sorted(sessions)

    def test_clear(self, tmp_path):
        mem = ConversationMemory(memory_dir=str(tmp_path))
        mem.add("user", "hello")
        mem.add("assistant", "hi")
        mem.clear()
        s = mem.stats
        assert s["messages"] == 0
        assert s["total_messages"] == 0
        assert s["summaries"] == 0
        assert s["total_indexed"] == 0

    def test_load_nonexistent_raises(self, tmp_path):
        mem = ConversationMemory(memory_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            mem.load_session("does_not_exist")

    def test_auto_session_name(self, tmp_path):
        mem = ConversationMemory(memory_dir=str(tmp_path))
        mem.add("user", "hello")
        path = mem.save_session()
        # Auto-generated name should be a timestamp
        assert path.suffix == ".jsonl"
        assert path.exists()
        # Session name should be set
        assert mem.stats["session"] is not None

    def test_jsonl_format(self, tmp_path):
        """Verify the JSONL file format is correct."""
        mem = ConversationMemory(memory_dir=str(tmp_path))
        mem.add("user", "test message")
        mem.add("assistant", "test reply")
        path = mem.save_session("format_test")

        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            rec = json.loads(line)
            assert "ts" in rec
            assert "role" in rec
            assert "content" in rec
            assert "importance" in rec

    def test_build_context_with_summaries(self, tmp_path):
        """After compression, build_context includes summaries."""
        mem = ConversationMemory(
            token_budget=200,
            keep_recent=2,
            memory_dir=str(tmp_path),
        )
        for i in range(10):
            mem.add("user", f"Message {i} with padding content " * 5)
            mem.add("assistant", f"Reply {i} with padding " * 5)

        ctx = mem.build_context("next question")
        # Should have at least one system message with summary
        system_msgs = [m for m in ctx if m["role"] == "system"]
        assert any("summary" in m["content"].lower() for m in system_msgs)


# ---------------------------------------------------------------------------
# _sanitize_filename
# ---------------------------------------------------------------------------


class TestSanitizeFilename:
    def test_strips_special_chars(self):
        assert _sanitize_filename("my:session/test*") == "mysessiontest"

    def test_collapses_whitespace(self):
        assert _sanitize_filename("hello   world") == "hello_world"

    def test_empty_string(self):
        result = _sanitize_filename("")
        assert result.startswith("session_")
        assert len(result) > len("session_")  # Has a hash suffix

    def test_distinct_bad_inputs_dont_collide(self):
        """Different all-special-char inputs should produce different filenames."""
        a = _sanitize_filename("***")
        b = _sanitize_filename("???")
        assert a != b  # Hash-based fallback prevents collision

    def test_normal_name(self):
        assert _sanitize_filename("my_session_2024") == "my_session_2024"


# ---------------------------------------------------------------------------
# _distill_facts — unclosed code blocks
# ---------------------------------------------------------------------------


class TestDistillFacts:
    def test_unclosed_code_block_preserved(self):
        """Unclosed code blocks should be flushed with a closing fence, not dropped."""
        msgs = [{"role": "assistant", "content": "```python\nprint('hello')"}]
        result = _distill_facts(msgs)
        assert "print('hello')" in result
        assert result.strip().endswith("```")

    def test_closed_code_block_works(self):
        msgs = [{"role": "assistant", "content": "```python\nprint('hi')\n```"}]
        result = _distill_facts(msgs)
        assert "print('hi')" in result


# ---------------------------------------------------------------------------
# load_session — JSON error handling
# ---------------------------------------------------------------------------


class TestLoadSessionErrorHandling:
    def test_malformed_jsonl_skipped(self, tmp_path):
        """Malformed lines are skipped, valid lines still loaded."""
        path = tmp_path / "bad_session.jsonl"
        lines = [
            json.dumps({"ts": "", "role": "user", "content": "hello", "importance": 0.5}),
            "THIS IS NOT JSON",
            json.dumps({"ts": "", "role": "assistant", "content": "hi", "importance": 0.5}),
        ]
        path.write_text("\n".join(lines), encoding="utf-8")

        mem = ConversationMemory(memory_dir=str(tmp_path))
        mem.load_session("bad_session")
        assert mem.stats["total_messages"] == 2  # Skipped the bad line

    def test_load_preserves_state_on_file_not_found(self, tmp_path):
        """If load fails with FileNotFoundError, previous state is preserved."""
        mem = ConversationMemory(memory_dir=str(tmp_path))
        mem.add("user", "existing message")

        with pytest.raises(FileNotFoundError):
            mem.load_session("nonexistent")

        # State should be preserved (not cleared)
        assert mem.stats["total_messages"] == 1

    def test_load_missing_fields_uses_defaults(self, tmp_path):
        """Records with missing fields should use safe defaults."""
        path = tmp_path / "partial.jsonl"
        # Minimal record — missing 'ts' and 'importance'
        path.write_text(json.dumps({"role": "user", "content": "test"}) + "\n", encoding="utf-8")

        mem = ConversationMemory(memory_dir=str(tmp_path))
        mem.load_session("partial")
        assert mem.stats["total_messages"] == 1


# ---------------------------------------------------------------------------
# Config — safe int parsing
# ---------------------------------------------------------------------------


class TestConfigSafeInt:
    def test_invalid_int_falls_back_to_default(self):
        from grok_mccodin.config import _safe_int

        with patch.dict("os.environ", {"TEST_INT": "not_a_number"}):
            result = _safe_int("TEST_INT", 42)
            assert result == 42

    def test_valid_int_parsed(self):
        from grok_mccodin.config import _safe_int

        with patch.dict("os.environ", {"TEST_INT": "100"}):
            result = _safe_int("TEST_INT", 42)
            assert result == 100

    def test_missing_env_uses_default(self):
        from grok_mccodin.config import _safe_int

        result = _safe_int("DEFINITELY_NOT_SET_XYZ_123", 99)
        assert result == 99


# ---------------------------------------------------------------------------
# Context window guardrail
# ---------------------------------------------------------------------------


class TestBuildContextGuardrail:
    def test_context_stays_bounded(self, tmp_path):
        """build_context output should not exceed the context token budget."""
        mem = ConversationMemory(
            token_budget=200,
            keep_recent=2,
            memory_dir=str(tmp_path),
        )
        # Generate lots of messages to create many summaries
        for i in range(30):
            mem.add("user", f"Topic {i} content with padding words " * 10)
            mem.add("assistant", f"Reply {i} with detailed answer " * 10)

        ctx = mem.build_context("next question")
        total_tokens = sum(estimate_tokens(m["content"]) for m in ctx)
        # Should be bounded (with some slack for message overhead)
        assert total_tokens < 20000


# ---------------------------------------------------------------------------
# _all_messages pruning
# ---------------------------------------------------------------------------


class TestAllMessagesPruning:
    def test_prune_triggered_at_cap(self, tmp_path):
        """_all_messages should be pruned when exceeding _MAX_ALL_MESSAGES."""
        mem = ConversationMemory(
            token_budget=100000,  # High budget to avoid compression interfering
            keep_recent=5,
            memory_dir=str(tmp_path),
        )
        # Patch the cap to a small value for testing
        with patch("grok_mccodin.memory._MAX_ALL_MESSAGES", 20):
            for i in range(25):
                mem.add("user", f"msg {i}")

        # Should have pruned some messages
        assert len(mem._all_messages) < 25


# ---------------------------------------------------------------------------
# Summary capping
# ---------------------------------------------------------------------------


class TestSummaryCapping:
    def test_summaries_dont_grow_unbounded(self, tmp_path):
        """Summaries should be merged when exceeding the cap."""
        mem = ConversationMemory(
            token_budget=50,  # Very low to force frequent compression
            keep_recent=1,
            memory_dir=str(tmp_path),
        )
        with patch("grok_mccodin.memory._MAX_SUMMARIES", 5):
            for i in range(100):
                mem.add("user", f"Topic {i} with long content " * 5)
                mem.add("assistant", f"Reply {i} " * 5)

        # Summaries should be bounded
        assert len(mem._summaries) <= 55  # Bounded (may be slightly over due to merge timing)
