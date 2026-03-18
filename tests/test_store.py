"""Core store tests — CRUD, FTS5 search, vector search, list."""

from pathlib import Path

import pytest

from hickey.store import Memory, MemoryType, MemoryStore, SearchResult


@pytest.fixture
def store(tmp_path):
    return MemoryStore(base_dir=tmp_path)


class TestCRUD:
    def test_save_and_list(self, store):
        m = Memory(content="Test memory.", type=MemoryType.FACT)
        saved = store.save(m)
        assert saved.id == m.id

        memories = store.list()
        assert len(memories) == 1
        assert memories[0].content == "Test memory."
        assert memories[0].type == MemoryType.FACT

    def test_save_preserves_fields(self, store):
        m = Memory(
            content="Corrections decay slower.",
            type=MemoryType.CORRECTION,
            tags=["design", "decay"],
            project="hickey",
            confidence=0.95,
            auto=True,
        )
        store.save(m)
        loaded = store.list()[0]
        assert loaded.id == m.id
        assert loaded.content == m.content
        assert loaded.type == MemoryType.CORRECTION
        assert loaded.tags == ["design", "decay"]
        assert loaded.project == "hickey"
        assert loaded.confidence == 0.95
        assert loaded.auto is True

    def test_save_updates_existing(self, store):
        m = store.save(Memory(content="Original.", type=MemoryType.FACT))
        m.content = "Updated."
        m.type = MemoryType.DECISION
        store.save(m)
        memories = store.list()
        assert len(memories) == 1
        assert memories[0].content == "Updated."
        assert memories[0].type == MemoryType.DECISION

    def test_delete(self, store):
        m = store.save(Memory(content="Ephemeral.", type=MemoryType.FACT))
        store.delete(m.id)
        assert store.list() == []

    def test_delete_nonexistent(self, store):
        store.delete("doesnotexist")  # should not raise


class TestList:
    def test_list_all(self, store):
        store.save(Memory(content="One.", type=MemoryType.FACT))
        store.save(Memory(content="Two.", type=MemoryType.DECISION))
        assert len(store.list()) == 2

    def test_list_by_type(self, store):
        store.save(Memory(content="A fact.", type=MemoryType.FACT))
        store.save(Memory(content="A decision.", type=MemoryType.DECISION))
        facts = store.list(type=MemoryType.FACT)
        assert len(facts) == 1
        assert facts[0].type == MemoryType.FACT

    def test_list_by_project(self, store):
        store.save(Memory(content="Alpha.", type=MemoryType.FACT, project="alpha"))
        store.save(Memory(content="Beta.", type=MemoryType.FACT, project="beta"))
        results = store.list(project="alpha")
        assert len(results) == 1
        assert results[0].project == "alpha"

    def test_list_limit(self, store):
        for i in range(10):
            store.save(Memory(content=f"Memory {i}.", type=MemoryType.FACT))
        assert len(store.list(limit=3)) == 3

    def test_list_ordered_by_updated(self, store):
        a = store.save(Memory(content="First.", type=MemoryType.FACT))
        b = store.save(Memory(content="Second.", type=MemoryType.FACT))
        memories = store.list()
        assert memories[0].id == b.id
        assert memories[1].id == a.id


class TestSearch:
    def test_fts_basic(self, store):
        store.save(Memory(content="SQLite is the best database for embedded use.", type=MemoryType.FACT))
        store.save(Memory(content="Python decorators simplify code.", type=MemoryType.FACT))
        store.save(Memory(content="Never use MongoDB for this project.", type=MemoryType.CORRECTION))

        results = store.search("SQLite database")
        assert len(results) >= 1
        assert "SQLite" in results[0].memory.content

    def test_type_filter(self, store):
        store.save(Memory(content="Always validate input.", type=MemoryType.CORRECTION))
        store.save(Memory(content="Input validation is important.", type=MemoryType.FACT))

        results = store.search("validate input", type=MemoryType.CORRECTION)
        assert all(r.memory.type == MemoryType.CORRECTION for r in results)

    def test_project_filter(self, store):
        store.save(Memory(content="Alpha database choice.", type=MemoryType.DECISION, project="alpha"))
        store.save(Memory(content="Beta database choice.", type=MemoryType.DECISION, project="beta"))

        results = store.search("database", project="alpha")
        assert all(r.memory.project == "alpha" for r in results)

    def test_empty_query(self, store):
        store.save(Memory(content="Something.", type=MemoryType.FACT))
        results = store.search("")
        assert results == []

    def test_returns_search_results(self, store):
        store.save(Memory(content="Apples and oranges.", type=MemoryType.FACT))
        results = store.search("apples")
        assert len(results) >= 1
        assert isinstance(results[0], SearchResult)
        assert results[0].score > 0

    def test_correction_ranks_higher(self, store):
        store.save(Memory(content="Database connections should be pooled.", type=MemoryType.FACT))
        store.save(Memory(content="Database connections must be pooled.", type=MemoryType.CORRECTION))

        results = store.search("database connections pooled")
        assert len(results) == 2
        assert results[0].memory.type == MemoryType.CORRECTION
