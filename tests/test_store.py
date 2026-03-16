"""Core store tests — markdown round-trip, CRUD, FTS5 search, rebuild."""

import tempfile
from pathlib import Path

import pytest

from hickey.model import Memory, MemoryType
from hickey.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(base_dir=tmp_path)
    yield s
    s.close()


class TestMarkdownRoundTrip:
    def test_simple(self, store):
        m = Memory(content="Use SQLite for everything.", type=MemoryType.DECISION)
        md = store._to_markdown(m)
        assert "---" in md
        assert "Use SQLite for everything." in md

        parsed = store._from_markdown(md)
        assert parsed.id == m.id
        assert parsed.content == m.content
        assert parsed.type == m.type
        assert parsed.confidence == m.confidence

    def test_with_tags_and_project(self, store):
        m = Memory(
            content="Corrections decay slower.",
            type=MemoryType.CORRECTION,
            tags=["design", "decay"],
            project="hickey",
            confidence=0.95,
        )
        parsed = store._from_markdown(store._to_markdown(m))
        assert parsed.tags == ["design", "decay"]
        assert parsed.project == "hickey"
        assert parsed.confidence == 0.95


class TestCRUD:
    def test_save_and_get(self, store):
        m = Memory(content="Test memory.", type=MemoryType.FACT)
        saved = store.save(m)
        assert (store.memories_dir / f"{saved.id}.md").exists()

        loaded = store.get(saved.id)
        assert loaded is not None
        assert loaded.content == "Test memory."

    def test_delete(self, store):
        m = store.save(Memory(content="Ephemeral.", type=MemoryType.FACT))
        assert store.delete(m.id)
        assert store.get(m.id) is None
        assert not (store.memories_dir / f"{m.id}.md").exists()

    def test_delete_nonexistent(self, store):
        assert not store.delete("doesnotexist")

    def test_get_nonexistent(self, store):
        assert store.get("doesnotexist") is None


class TestSearch:
    def test_fts_basic(self, store):
        store.save(Memory(content="SQLite is the best database for embedded use.", type=MemoryType.FACT))
        store.save(Memory(content="Python decorators simplify code.", type=MemoryType.FACT))
        store.save(Memory(content="Never use MongoDB for this project.", type=MemoryType.CORRECTION))

        results = store.search("SQLite database")
        assert len(results) >= 1
        assert "SQLite" in results[0][0].content

    def test_type_filter(self, store):
        store.save(Memory(content="Always validate input.", type=MemoryType.CORRECTION))
        store.save(Memory(content="Input validation is important.", type=MemoryType.FACT))

        results = store.search("validate input", type_filter=MemoryType.CORRECTION)
        assert all(m.type == MemoryType.CORRECTION for m, _ in results)

    def test_empty_query(self, store):
        store.save(Memory(content="Something.", type=MemoryType.FACT))
        results = store.search("")
        assert results == []

    def test_no_results(self, store):
        store.save(Memory(content="Apples and oranges.", type=MemoryType.FACT))
        results = store.search("quantum entanglement")
        assert results == []


class TestListAndAudit:
    def test_list_all(self, store):
        store.save(Memory(content="One.", type=MemoryType.FACT))
        store.save(Memory(content="Two.", type=MemoryType.DECISION))
        assert len(store.list_memories()) == 2

    def test_list_by_type(self, store):
        store.save(Memory(content="A fact.", type=MemoryType.FACT))
        store.save(Memory(content="A decision.", type=MemoryType.DECISION))
        facts = store.list_memories(type_filter=MemoryType.FACT)
        assert len(facts) == 1
        assert facts[0].type == MemoryType.FACT

    def test_audit_stats(self, store):
        store.save(Memory(content="Fact one.", type=MemoryType.FACT))
        store.save(Memory(content="Correction one.", type=MemoryType.CORRECTION))
        result = store.audit()
        assert result["stats"]["total"] == 2
        assert result["stats"]["by_type"]["fact"] == 1
        assert result["stats"]["by_type"]["correction"] == 1


class TestRebuild:
    def test_rebuild_recovers_index(self, store):
        store.save(Memory(content="Indexed memory.", type=MemoryType.FACT))
        store.save(Memory(content="Another one.", type=MemoryType.DECISION))

        # Nuke the index
        count = store.rebuild_index()
        assert count == 2

        # Search still works
        results = store.search("Indexed")
        assert len(results) >= 1
