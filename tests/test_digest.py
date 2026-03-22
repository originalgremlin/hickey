"""Digest tests — transcript parsing, watermarks, memory extraction."""

import json
import numpy as np
from pathlib import Path
from unittest.mock import patch

import pytest

from hickey.store import MemoryStore, EMBED_DIM
from hickey import api
from hickey import digest


@pytest.fixture(autouse=True)
def _patch_store(tmp_path):
    """Point the global api.store at a temp database for every test."""
    original = api.store
    store = MemoryStore(base_dir=tmp_path)
    store._embed = lambda text: np.random.default_rng(hash(text) % 2**32).random(EMBED_DIM).astype(np.float32).tobytes()
    api.store = store
    yield
    api.store = original


class TestWatermarks:
    def test_get_nonexistent(self):
        assert api.store.get_watermark("/no/such/path") == 0

    def test_set_and_get(self):
        api.store.set_watermark("/tmp/t.jsonl", 42)
        assert api.store.get_watermark("/tmp/t.jsonl") == 42

    def test_upsert(self):
        api.store.set_watermark("/tmp/t.jsonl", 0)
        api.store.set_watermark("/tmp/t.jsonl", 100)
        assert api.store.get_watermark("/tmp/t.jsonl") == 100


class TestParseTranscript:
    def _make_jsonl(self, entries: list[dict]) -> str:
        return "\n".join(json.dumps(e) for e in entries)

    def test_user_text(self):
        result = digest._parse_transcript(self._make_jsonl([
            {"type": "user", "message": {"role": "user", "content": "hello world"}},
        ]))
        assert "user: hello world" in result

    def test_assistant_text(self):
        result = digest._parse_transcript(self._make_jsonl([
            {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "text", "text": "here is my response"}
            ]}},
        ]))
        assert "assistant: here is my response" in result

    def test_tool_use(self):
        result = digest._parse_transcript(self._make_jsonl([
            {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "tool_use", "name": "Read", "input": {"file_path": "/foo.py"}}
            ]}},
        ]))
        assert "[tool: Read(" in result
        assert "/foo.py" in result

    def test_tool_result(self):
        result = digest._parse_transcript(self._make_jsonl([
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "content": "file contents here", "is_error": False}
            ]}},
        ]))
        assert "[tool_result: file contents here]" in result

    def test_skips_system_reminder(self):
        result = digest._parse_transcript(self._make_jsonl([
            {"type": "user", "message": {"role": "user", "content": "<system-reminder>secret</system-reminder>"}},
        ]))
        assert result == ""

    def test_skips_meta(self):
        result = digest._parse_transcript(self._make_jsonl([
            {"type": "user", "isMeta": True, "message": {"role": "user", "content": "should be skipped"}},
        ]))
        assert result == ""

    def test_skips_non_message_types(self):
        result = digest._parse_transcript(self._make_jsonl([
            {"type": "file-history-snapshot", "snapshot": {}},
            {"type": "progress", "content": "thinking..."},
            {"type": "user", "message": {"role": "user", "content": "real message"}},
        ]))
        assert "real message" in result
        assert "snapshot" not in result
        assert "thinking" not in result

    def test_truncates_long_tool_result(self):
        long_content = "x" * 3000
        result = digest._parse_transcript(self._make_jsonl([
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "content": long_content, "is_error": False}
            ]}},
        ]))
        assert "[truncated]" in result
        assert len(result) < 3000

    def test_skips_digest_prompt_echo(self):
        result = digest._parse_transcript(self._make_jsonl([
            {"type": "user", "message": {"role": "user", "content": "You are a memory extraction system. Given a chunk..."}},
            {"type": "user", "message": {"role": "user", "content": "real question here"}},
        ]))
        assert "memory extraction system" not in result
        assert "real question here" in result


class TestDigest:
    def _write_transcript(self, tmp_path: Path, entries: list[dict]) -> Path:
        p = tmp_path / "transcript.jsonl"
        p.write_text("\n".join(json.dumps(e) for e in entries))
        return p

    def test_empty_transcript(self, tmp_path):
        p = tmp_path / "transcript.jsonl"
        p.write_text("")
        assert digest.digest(str(p), "proj") == 0

    def test_nonexistent_file(self):
        assert digest.digest("/no/such/file.jsonl", "proj") == 0

    @patch("hickey.digest._extract")
    def test_calls_extract_and_stores(self, mock_extract, tmp_path):
        mock_extract.return_value = [
            {"type": "decision", "content": "Use SQLite for storage.", "confidence": 0.9},
        ]
        p = self._write_transcript(tmp_path, [
            {"type": "user", "message": {"role": "user", "content": "what database should we use?"}},
            {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "text", "text": "SQLite is the right choice here."}
            ]}},
        ])
        assert digest.digest(str(p), "testproj") == 1
        memories = api.store.list()
        assert len(memories) == 1
        assert memories[0].content == "Use SQLite for storage."
        assert memories[0].type.name == "DECISION"
        assert memories[0].project == "testproj"

    @patch("hickey.digest._extract")
    def test_extraction_failure_preserves_offset(self, mock_extract, tmp_path):
        mock_extract.return_value = None
        p = self._write_transcript(tmp_path, [
            {"type": "user", "message": {"role": "user", "content": "important stuff"}},
        ])
        digest.digest(str(p), "proj")
        assert api.store.get_watermark(str(p)) == 0

    @patch("hickey.digest._extract")
    def test_incremental_offset(self, mock_extract, tmp_path):
        mock_extract.return_value = []
        p = self._write_transcript(tmp_path, [
            {"type": "user", "message": {"role": "user", "content": "first chunk"}},
        ])
        digest.digest(str(p), "proj")
        offset1 = api.store.get_watermark(str(p))
        assert offset1 > 0
        with open(p, "a") as f:
            f.write("\n" + json.dumps({"type": "user", "message": {"role": "user", "content": "second chunk"}}))
        mock_extract.return_value = [{"type": "fact", "content": "from second chunk", "confidence": 0.7}]
        assert digest.digest(str(p), "proj") == 1
        assert api.store.get_watermark(str(p)) > offset1

    @patch("hickey.digest._extract")
    def test_advances_watermark_on_empty_conversation(self, mock_extract, tmp_path):
        p = self._write_transcript(tmp_path, [
            {"type": "progress", "content": "thinking..."},
        ])
        digest.digest(str(p), "proj")
        assert api.store.get_watermark(str(p)) > 0
        mock_extract.assert_not_called()


def _mock_result(stdout: str, returncode: int = 0, stderr: str = "") -> object:
    return type("R", (), {"stdout": stdout, "returncode": returncode, "stderr": stderr})()


class TestExtract:
    @patch("subprocess.run")
    def test_parses_json_array(self, mock_run):
        mock_run.return_value = _mock_result(
            '[{"type": "fact", "content": "SQLite is fast.", "confidence": 0.9}]'
        )
        result = digest._extract("some conversation", "proj")
        assert len(result) == 1
        assert result[0]["content"] == "SQLite is fast."

    @patch("subprocess.run")
    def test_parses_json_wrapped_in_prose(self, mock_run):
        mock_run.return_value = _mock_result(
            'Here are the memories:\n\n[{"type": "decision", "content": "Use RRF.", "confidence": 0.8}]\n\nThese are important.'
        )
        result = digest._extract("some conversation", "proj")
        assert len(result) == 1

    @patch("subprocess.run")
    def test_empty_array(self, mock_run):
        mock_run.return_value = _mock_result("[]")
        assert digest._extract("boring conversation", "proj") == []

    @patch("subprocess.run")
    def test_filters_invalid_types(self, mock_run):
        mock_run.return_value = _mock_result(
            '[{"type": "bogus", "content": "nope"}, {"type": "fact", "content": "yes", "confidence": 0.8}]'
        )
        result = digest._extract("conversation", "proj")
        assert len(result) == 1
        assert result[0]["type"] == "fact"

    @patch("subprocess.run")
    def test_handles_malformed_output(self, mock_run):
        mock_run.return_value = _mock_result("not json at all")
        assert digest._extract("conversation", "proj") is None

    @patch("subprocess.run")
    def test_handles_nonzero_exit(self, mock_run):
        mock_run.return_value = _mock_result("", returncode=1, stderr="rate limited")
        assert digest._extract("conversation", "proj") is None

    @patch("subprocess.run", side_effect=FileNotFoundError("claude not found"))
    def test_handles_missing_claude(self, mock_run):
        assert digest._extract("conversation", "proj") is None

    @patch("subprocess.run")
    def test_passes_project_to_prompt(self, mock_run):
        mock_run.return_value = _mock_result("[]")
        digest._extract("conversation", "myproject")
        call_args = mock_run.call_args
        system_prompt = call_args[0][0][call_args[0][0].index("--system-prompt") + 1]
        assert "myproject" in system_prompt
