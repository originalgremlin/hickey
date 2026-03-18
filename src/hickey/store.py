import os
import sqlite3
import uuid
import sqlite_vec
import typing as T
from dataclasses import dataclass, field
from functools import cached_property
from datetime import datetime, timezone, MAXYEAR
from enum import Enum
from fastembed import TextEmbedding
from pathlib import Path


EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"
EMBED_DIM: int = 384


class MemoryType(Enum):
    CORRECTION    = (1.5, 90)
    DECISION      = (1.2, 60)
    FACT          = (1.0, 30)
    PREFERENCE    = (1.1, 60)
    INVESTIGATION = (0.8, 21)
    AUTO          = (0.7, 45)

    def __init__(self, weight: float, halflife: int):
        self.weight = weight
        self.halflife = halflife


@dataclass
class Memory:
    content: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    type: MemoryType = MemoryType.FACT
    project: str = field(default_factory=lambda: os.path.basename(os.getcwd()))
    tags: list[str] = field(default_factory=list)
    auto: bool = False
    confidence: float = 1.0
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires: datetime = field(default_factory=lambda: datetime(MAXYEAR, 12, 31, 23, 59, 59, 999999))

    def __repr__(self) -> str:
        age: int = (datetime.now(timezone.utc) - self.created).days
        line: str = f"[{self.type.name.lower()}] {self.content[:200]}"
        meta: str = f"  id={self.id}  confidence={self.confidence}  age={age}days  project={self.project}  tags={','.join(self.tags)}"
        return f"{line}\n{meta}"


class SearchResult(T.NamedTuple):
    memory: Memory
    score: float

    def __repr__(self) -> str:
        return f"({self.score:.4f}) {self.memory}"


class MemoryStore:
    def __init__(self, base_dir: T.Optional[Path] = None):
        self.base_dir: Path = Path(base_dir) if base_dir else Path.home() / ".hickey"
        self.db_path: Path = self.base_dir / "hickey.db"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @cached_property
    def embedder(self) -> TextEmbedding:
        return TextEmbedding(EMBED_MODEL)

    def _init_db(self) -> None:
        self._db: sqlite3.Connection = sqlite3.connect(str(self.db_path))
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.enable_load_extension(True)
        sqlite_vec.load(self._db)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id         TEXT PRIMARY KEY,
                content    TEXT NOT NULL,
                type       TEXT NOT NULL,
                project    TEXT NOT NULL,
                tags       TEXT NOT NULL DEFAULT '',
                auto       INTEGER NOT NULL DEFAULT 0,
                confidence REAL NOT NULL DEFAULT 1.0,
                created    TEXT NOT NULL,
                updated    TEXT NOT NULL,
                expires    TEXT NOT NULL
            )
        """)
        self._db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                id, content, type, tags, project,
                tokenize='porter unicode61'
            )
        """)
        self._db.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                id   TEXT PRIMARY KEY,
                embedding float[{EMBED_DIM}]
            )
        """)
        self._db.commit()

    def _embed(self, text: str) -> bytes:
        return list(self.embedder.embed([text]))[0].tobytes()

    def _row_to_memory(self, row: tuple) -> Memory:
        return Memory(
            id=row[0],
            content=row[1],
            type=MemoryType[row[2].upper()],
            project=row[3],
            tags=row[4].split(",") if row[4] else [],
            auto=bool(row[5]),
            confidence=row[6],
            created=datetime.fromisoformat(row[7]),
            updated=datetime.fromisoformat(row[8]),
            expires=datetime.fromisoformat(row[9]),
        )

    _MEMORY_COLS: str = "id, content, type, project, tags, auto, confidence, created, updated, expires"

    def save(
        self,
        memory: Memory,
    ) -> Memory:
        """Insert or replace a memory."""
        memory.updated = datetime.now(timezone.utc)
        self._db.execute(
            f"INSERT OR REPLACE INTO memories ({self._MEMORY_COLS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (memory.id, memory.content, memory.type.name.lower(), memory.project,
             ",".join(memory.tags), int(memory.auto), memory.confidence,
             memory.created.isoformat(), memory.updated.isoformat(), memory.expires.isoformat()),
        )
        self._db.execute("DELETE FROM memories_fts WHERE id = ?", (memory.id,))
        self._db.execute(
            "INSERT INTO memories_fts (id, content, type, tags, project) VALUES (?, ?, ?, ?, ?)",
            (memory.id, memory.content, memory.type.name.lower(), " ".join(memory.tags), memory.project),
        )
        self._db.execute("DELETE FROM memories_vec WHERE id = ?", (memory.id,))
        self._db.execute(
            "INSERT INTO memories_vec (id, embedding) VALUES (?, ?)",
            (memory.id, self._embed(memory.content)),
        )
        self._db.commit()
        return memory

    def delete(
        self,
        id: str,
    ) -> None:
        """Delete a memory by ID."""
        self._db.execute("DELETE FROM memories WHERE id = ?", (id,))
        self._db.execute("DELETE FROM memories_fts WHERE id = ?", (id,))
        self._db.execute("DELETE FROM memories_vec WHERE id = ?", (id,))
        self._db.commit()

    def list(
        self,
        type: T.Optional[MemoryType] = None,
        project: T.Optional[str] = None,
        limit: int = 20,
    ) -> T.List[Memory]:
        """Browse memories, newest first."""
        sql: str = f"SELECT {self._MEMORY_COLS} FROM memories WHERE 1=1"
        params: list = []
        if type:
            sql += " AND type = ?"
            params.append(type.name.lower())
        if project:
            sql += " AND project = ?"
            params.append(project)
        sql += " ORDER BY updated DESC LIMIT ?"
        params.append(limit)
        rows: list = self._db.execute(sql, params).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def search(
        self,
        query: str,
        type: T.Optional[MemoryType] = None,
        project: T.Optional[str] = None,
        limit: int = 5,
    ) -> T.List[SearchResult]:
        """Hybrid search: FTS5 BM25 + vector similarity, fused with RRF.
        Results are boosted by type weight, confidence, and freshness decay."""
        if not query.strip():
            return []
        overfetch: int = limit * 3
        now_iso: str = datetime.now(timezone.utc).isoformat()

        # Build filter clause
        where: str = "m.expires > ?"
        where_params: list = [now_iso]
        if type:
            where += " AND m.type = ?"
            where_params.append(type.name.lower())
        if project:
            where += " AND m.project = ?"
            where_params.append(project)

        # FTS5 ranked list (joined with memories to pre-filter)
        fts_ranked: T.List[str] = []
        try:
            fts_query: str = " OR ".join(f'"{t}"' for t in query.split() if t.strip())
            if fts_query:
                rows = self._db.execute(
                    f"""SELECT f.id FROM memories_fts f
                        JOIN memories m ON f.id = m.id
                        WHERE memories_fts MATCH ? AND {where}
                        ORDER BY f.rank LIMIT ?""",
                    (fts_query, *where_params, overfetch),
                ).fetchall()
                fts_ranked = [r[0] for r in rows]
        except sqlite3.OperationalError:
            pass

        # Vector ranked list
        rows = self._db.execute(
            "SELECT id, distance FROM memories_vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (self._embed(query), overfetch),
        ).fetchall()
        vec_ranked: T.List[str] = [r[0] for r in rows]

        # RRF fusion (k=60)
        k: int = 60
        rrf_scores: T.Dict[str, float] = {}
        for rank, mid in enumerate(fts_ranked):
            rrf_scores[mid] = rrf_scores.get(mid, 0) + 1.0 / (k + rank + 1)
        for rank, mid in enumerate(vec_ranked):
            rrf_scores[mid] = rrf_scores.get(mid, 0) + 1.0 / (k + rank + 1)

        if not rrf_scores:
            return []

        # Filter vec-only candidates against memories table
        placeholders: str = ",".join("?" * len(rrf_scores))
        valid_rows = self._db.execute(
            f"SELECT id FROM memories WHERE id IN ({placeholders}) AND {where.replace('m.', '')}",
            (*rrf_scores.keys(), *where_params),
        ).fetchall()
        valid_ids: set = {r[0] for r in valid_rows}
        rrf_scores = {mid: s for mid, s in rrf_scores.items() if mid in valid_ids}

        # Load and boost
        results: T.List[SearchResult] = []
        now: datetime = datetime.now(timezone.utc)
        if rrf_scores:
            placeholders = ",".join("?" * len(rrf_scores))
            rows = self._db.execute(
                f"SELECT {self._MEMORY_COLS} FROM memories WHERE id IN ({placeholders})",
                tuple(rrf_scores.keys()),
            ).fetchall()
            for row in rows:
                memory: Memory = self._row_to_memory(row)
                rrf: float = rrf_scores[memory.id]
                age_days: float = max((now - memory.created).total_seconds() / 86400, 0)
                freshness: float = 0.5 ** (age_days / memory.type.halflife)
                score: float = rrf * memory.type.weight * memory.confidence * freshness
                results.append(SearchResult(memory=memory, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]
