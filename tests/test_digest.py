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
        assert api.store.get_watermark("/no/such/path") is None

    def test_set_and_get(self):
        api.store.set_watermark("/tmp/t.jsonl", 42, "myproject")
        wm = api.store.get_watermark("/tmp/t.jsonl")
        assert wm == (42, "myproject")

    def test_upsert(self):
        api.store.set_watermark("/tmp/t.jsonl", 0, "proj")
        api.store.set_watermark("/tmp/t.jsonl", 100, "proj")
        assert api.store.get_watermark("/tmp/t.jsonl") == (100, "proj")

    def test_all_watermarks(self):
        api.store.set_watermark("/a.jsonl", 10, "alpha")
        api.store.set_watermark("/b.jsonl", 20, "beta")
        wms = api.store.all_watermarks()
        assert len(wms) == 2
        paths = {w[0] for w in wms}
        assert paths == {"/a.jsonl", "/b.jsonl"}


class TestRegister:
    def test_register_creates_watermark(self):
        digest.register("/tmp/t.jsonl", "proj")
        assert api.store.get_watermark("/tmp/t.jsonl") == (0, "proj")

    def test_register_idempotent(self):
        digest.register("/tmp/t.jsonl", "proj")
        api.store.set_watermark("/tmp/t.jsonl", 99, "proj")
        digest.register("/tmp/t.jsonl", "proj")
        assert api.store.get_watermark("/tmp/t.jsonl") == (99, "proj")


class TestParseTranscript:
    def _make_jsonl(self, entries: list[dict]) -> str:
        return "\n".join(json.dumps(e) for e in entries)

    def _joined(self, entries: list[dict]) -> str:
        return "\n\n".join(digest._parse_transcript(self._make_jsonl(entries)))

    def test_user_text(self):
        result = self._joined([
            {"type": "user", "message": {"role": "user", "content": "hello world"}},
        ])
        assert "user: hello world" in result

    def test_assistant_text(self):
        result = self._joined([
            {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "text", "text": "here is my response"}
            ]}},
        ])
        assert "assistant: here is my response" in result

    def test_tool_use(self):
        result = self._joined([
            {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "tool_use", "name": "Read", "input": {"file_path": "/foo.py"}}
            ]}},
        ])
        assert "[tool: Read(" in result
        assert "/foo.py" in result

    def test_tool_result(self):
        result = self._joined([
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "content": "file contents here", "is_error": False}
            ]}},
        ])
        assert "[tool_result: file contents here]" in result

    def test_skips_system_reminder(self):
        result = digest._parse_transcript(self._make_jsonl([
            {"type": "user", "message": {"role": "user", "content": "<system-reminder>secret</system-reminder>"}},
        ]))
        assert result == []

    def test_skips_meta(self):
        result = digest._parse_transcript(self._make_jsonl([
            {"type": "user", "isMeta": True, "message": {"role": "user", "content": "should be skipped"}},
        ]))
        assert result == []

    def test_skips_non_message_types(self):
        result = self._joined([
            {"type": "file-history-snapshot", "snapshot": {}},
            {"type": "progress", "content": "thinking..."},
            {"type": "user", "message": {"role": "user", "content": "real message"}},
        ])
        assert "real message" in result
        assert "snapshot" not in result
        assert "thinking" not in result

    def test_truncates_long_tool_result(self):
        long_content = "x" * 3000
        result = self._joined([
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "content": long_content, "is_error": False}
            ]}},
        ])
        assert "[truncated]" in result
        assert len(result) < 3000


class TestDigestOne:
    def _write_transcript(self, tmp_path: Path, entries: list[dict]) -> Path:
        p = tmp_path / "transcript.jsonl"
        p.write_text("\n".join(json.dumps(e) for e in entries))
        return p

    def test_empty_transcript(self, tmp_path):
        p = tmp_path / "transcript.jsonl"
        p.write_text("")
        count, offset = digest._digest_one(str(p), "proj", 0)
        assert count == 0

    def test_nonexistent_file(self):
        count, offset = digest._digest_one("/no/such/file.jsonl", "proj", 0)
        assert count == 0
        assert offset == 0

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
        count, offset = digest._digest_one(str(p), "testproj", 0)
        assert count == 1
        assert offset > 0
        memories = api.store.list()
        assert len(memories) == 1
        assert memories[0].content == "Use SQLite for storage."
        assert memories[0].type.name == "DECISION"
        assert memories[0].project == "testproj"

    @patch("hickey.digest._extract")
    def test_extraction_failure_preserves_offset(self, mock_extract, tmp_path):
        mock_extract.return_value = None  # extraction failed
        p = self._write_transcript(tmp_path, [
            {"type": "user", "message": {"role": "user", "content": "important stuff"}},
        ])
        count, offset = digest._digest_one(str(p), "proj", 0)
        assert count == 0
        assert offset == 0  # watermark should NOT advance

    @patch("hickey.digest._extract")
    def test_partial_failure_stores_nothing(self, mock_extract, tmp_path):
        """If any chunk fails, no memories are stored (all-or-nothing)."""
        # First chunk succeeds, second fails
        mock_extract.side_effect = [
            [{"type": "fact", "content": "from chunk 1", "confidence": 0.9}],
            None,  # second chunk fails
        ]
        # Create a transcript large enough to require multiple chunks
        entries = []
        for i in range(200):
            entries.append({"type": "user", "message": {"role": "user", "content": f"message {i} " + "x" * 200}})
            entries.append({"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "text", "text": f"response {i} " + "y" * 200}
            ]}})
        p = self._write_transcript(tmp_path, entries)
        count, offset = digest._digest_one(str(p), "proj", 0)
        assert count == 0
        assert offset == 0  # watermark stays put
        assert api.store.list() == []  # nothing stored

    @patch("hickey.digest._extract")
    def test_incremental_offset(self, mock_extract, tmp_path):
        mock_extract.return_value = []
        p = self._write_transcript(tmp_path, [
            {"type": "user", "message": {"role": "user", "content": "first chunk"}},
        ])
        _, offset1 = digest._digest_one(str(p), "proj", 0)
        # append more
        with open(p, "a") as f:
            f.write("\n" + json.dumps({"type": "user", "message": {"role": "user", "content": "second chunk"}}))
        mock_extract.return_value = [{"type": "fact", "content": "from second chunk", "confidence": 0.7}]
        count, offset2 = digest._digest_one(str(p), "proj", offset1)
        assert offset2 > offset1
        assert count == 1


def _mock_result(stdout: str, returncode: int = 0, stderr: str = "") -> object:
    return type("R", (), {"stdout": stdout, "returncode": returncode, "stderr": stderr})()


class TestExtract:
    @patch("subprocess.run")
    def test_parses_json_array(self, mock_run):
        mock_run.return_value = _mock_result(
            '[{"type": "fact", "content": "SQLite is fast.", "confidence": 0.9}]'
        )
        result = digest._extract("some conversation")
        assert len(result) == 1
        assert result[0]["content"] == "SQLite is fast."

    @patch("subprocess.run")
    def test_parses_fenced_json(self, mock_run):
        mock_run.return_value = _mock_result(
            '```json\n[{"type": "decision", "content": "Use RRF.", "confidence": 0.8}]\n```'
        )
        result = digest._extract("some conversation")
        assert len(result) == 1

    @patch("subprocess.run")
    def test_empty_array(self, mock_run):
        mock_run.return_value = _mock_result("[]")
        result = digest._extract("boring conversation")
        assert result == []

    @patch("subprocess.run")
    def test_filters_invalid_types(self, mock_run):
        mock_run.return_value = _mock_result(
            '[{"type": "bogus", "content": "nope"}, {"type": "fact", "content": "yes", "confidence": 0.8}]'
        )
        result = digest._extract("conversation")
        assert len(result) == 1
        assert result[0]["type"] == "fact"

    @patch("subprocess.run")
    def test_handles_malformed_output(self, mock_run):
        mock_run.return_value = _mock_result("not json at all")
        result = digest._extract("conversation")
        assert result is None

    @patch("subprocess.run")
    def test_handles_nonzero_exit(self, mock_run):
        mock_run.return_value = _mock_result("", returncode=1, stderr="rate limited")
        result = digest._extract("conversation")
        assert result is None

    @patch("subprocess.run", side_effect=FileNotFoundError("claude not found"))
    def test_handles_missing_claude(self, mock_run):
        result = digest._extract("conversation")
        assert result is None
