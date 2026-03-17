import json
import os
import sqlite3
import struct
import uuid
import typing as T
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone, MAXYEAR
from enum import Enum
from pathlib import Path

try:
    import sqlite_vec  # type: ignore
    _HAS_VEC = True
except ImportError:
    _HAS_VEC = False

try:
    from fastembed import TextEmbedding  # type: ignore
    _HAS_EMBED = True
except ImportError:
    _HAS_EMBED = False

EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"
EMBED_DIM: int = 384


def _serialize_f32(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


class MemoryType(Enum):
    CORRECTION    = (1.5, 90)
    DECISION      = (1.2, 60)
    FACT          = (1.0, 30)
    PREFERENCE    = (1.1, 60)
    INVESTIGATION = (0.8, 21)

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

    def to_json(self) -> str:
        return json.dumps({
            "id": self.id,
            "content": self.content,
            "type": self.type.name.lower(),
            "project": self.project,
            "tags": self.tags,
            "auto": self.auto,
            "confidence": self.confidence,
            "created": self.created.isoformat(),
            "updated": self.updated.isoformat(),
            "expires": self.expires.isoformat(),
        }, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Memory":
        d: dict = json.loads(text)
        return cls(
            id=d["id"],
            content=d["content"],
            type=MemoryType[d["type"].upper()],
            project=d["project"],
            tags=d.get("tags", []),
            auto=d.get("auto", False),
            confidence=d.get("confidence", 1.0),
            created=datetime.fromisoformat(d["created"]),
            updated=datetime.fromisoformat(d["updated"]),
            expires=datetime.fromisoformat(d["expires"]),
        )

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
        self.memories_dir: Path = self.base_dir / "memories"
        self.db_path: Path = self.base_dir / "index.db"
        self.memories_dir.mkdir(parents=True, exist_ok=True)
        self._pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
        self._embedder: T.Optional[T.Any] = None
        self._init_db()

    def _init_db(self) -> None:
        self._db: sqlite3.Connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS memories_meta (
                id      TEXT PRIMARY KEY,
                type    TEXT NOT NULL,
                project TEXT NOT NULL,
                updated TEXT NOT NULL,
                expires TEXT NOT NULL
            )
        """)
        self._db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                id, content, type, tags, project,
                tokenize='porter unicode61'
            )
        """)
        if self.vectors_available:
            self._db.enable_load_extension(True)
            sqlite_vec.load(self._db)
            self._db.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                    id   TEXT PRIMARY KEY,
                    embedding float[{EMBED_DIM}]
                )
            """)
        self._db.commit()

    @property
    def vectors_available(self) -> bool:
        return _HAS_VEC and _HAS_EMBED

    def _embed(self, text: str) -> T.Optional[list[float]]:
        if not self.vectors_available:
            return None
        if self._embedder is None:
            self._embedder = TextEmbedding(EMBED_MODEL)
        return list(self._embedder.embed([text]))[0].tolist()

    def _save_index(self, memory: Memory) -> None:
        self._db.execute("DELETE FROM memories_meta WHERE id = ?", (memory.id,))
        self._db.execute(
            "INSERT INTO memories_meta (id, type, project, updated, expires) VALUES (?, ?, ?, ?, ?)",
            (memory.id, memory.type.name.lower(), memory.project, memory.updated.isoformat(), memory.expires.isoformat()),
        )
        self._db.execute("DELETE FROM memories_fts WHERE id = ?", (memory.id,))
        self._db.execute(
            "INSERT INTO memories_fts (id, content, type, tags, project) VALUES (?, ?, ?, ?, ?)",
            (memory.id, memory.content, memory.type.name.lower(), " ".join(memory.tags), memory.project),
        )
        if self.vectors_available:
            embedding: T.Optional[list[float]] = self._embed(memory.content)
            if embedding:
                self._db.execute("DELETE FROM memories_vec WHERE id = ?", (memory.id,))
                self._db.execute(
                    "INSERT INTO memories_vec (id, embedding) VALUES (?, ?)",
                    (memory.id, _serialize_f32(embedding)),
                )
        self._db.commit()

    def _delete_index(self, id: str) -> None:
        self._db.execute("DELETE FROM memories_meta WHERE id = ?", (id,))
        self._db.execute("DELETE FROM memories_fts WHERE id = ?", (id,))
        if self.vectors_available:
            self._db.execute("DELETE FROM memories_vec WHERE id = ?", (id,))
        self._db.commit()

    def _rebuild_index(self) -> None:
        """Drop and rebuild all index tables from JSON files on disk."""
        self._db.execute("DELETE FROM memories_meta")
        self._db.execute("DELETE FROM memories_fts")
        if self.vectors_available:
            self._db.execute("DELETE FROM memories_vec")
        for path in self.memories_dir.glob("*.json"):
            try:
                memory: Memory = Memory.from_json(path.read_text())
                self._save_index(memory)
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                print(f"warning: skipping {path.name}: {exc}")

    def index(self) -> None:
        """Rebuild the SQLite index from JSON files. Runs in a background thread."""
        self._pool.submit(self._rebuild_index)

    def save(
        self,
        memory: Memory
    ) -> Memory:
        """Write markdown file and update index. Indexing runs in a background thread."""
        memory.updated = datetime.now(timezone.utc)
        path: Path = self.memories_dir / f"{memory.id}.json"
        path.write_text(memory.to_json())
        self._pool.submit(self._save_index, memory)
        return memory

    def delete(
        self,
        id: str
    ) -> None:
        """Remove JSON file and delete from index."""
        path: Path = self.memories_dir / f"{id}.json"
        path.unlink(missing_ok=True)
        self._pool.submit(self._delete_index, id)

    def list(
        self,
        type: T.Optional[MemoryType] = None,
        project: T.Optional[str] = None,
        limit: int = 20,
    ) -> T.List[Memory]:
        """Browse memories, newest first. Queries index for IDs, loads files on demand."""
        # form sql query
        sql: str = "SELECT id FROM memories_meta WHERE 1=1"
        params: list = []
        if type:
            sql += " AND type = ?"
            params.append(type.name.lower())
        if project:
            sql += " AND project = ?"
            params.append(project)
        sql += " ORDER BY updated DESC LIMIT ?"
        params.append(limit)
        # get an ordered list of matching memory ids
        rows: list = self._db.execute(sql, params).fetchall()
        # load those memories from files
        memories: T.List[Memory] = []
        for (row_id,) in rows:
            path: Path = self.memories_dir / f"{row_id}.json"
            if path.exists():
                try:
                    memories.append(Memory.from_json(path.read_text()))
                except (json.JSONDecodeError, KeyError):
                    continue
        return memories

    def search(
        self,
        query: str,
        type: T.Optional[MemoryType] = None,
        project: T.Optional[str] = None,
        limit: int = 5,
    ) -> T.List[SearchResult]:
        """Hybrid search: FTS5 BM25 + optional vector similarity, fused with RRF.
        Results are boosted by type weight, confidence, and freshness decay."""
        overfetch: int = limit * 3
        now_iso: str = datetime.now(timezone.utc).isoformat()

        # Build meta filter clause
        meta_where: str = "m.expires > ?"
        meta_params: list = [now_iso]
        if type:
            meta_where += " AND m.type = ?"
            meta_params.append(type.name.lower())
        if project:
            meta_where += " AND m.project = ?"
            meta_params.append(project)

        # FTS5 ranked list (joined with meta to pre-filter)
        fts_ranked: T.List[str] = []
        try:
            fts_query: str = " OR ".join(f'"{t}"' for t in query.split() if t.strip())
            if fts_query:
                rows = self._db.execute(
                    f"""SELECT f.id FROM memories_fts f
                        JOIN memories_meta m ON f.id = m.id
                        WHERE memories_fts MATCH ? AND {meta_where}
                        ORDER BY f.rank LIMIT ?""",
                    (fts_query, *meta_params, overfetch),
                ).fetchall()
                fts_ranked = [r[0] for r in rows]
        except sqlite3.OperationalError:
            pass

        # Vector ranked list
        vec_ranked: T.List[str] = []
        if self.vectors_available:
            embedding: T.Optional[list[float]] = self._embed(query)
            if embedding:
                rows = self._db.execute(
                    "SELECT id, distance FROM memories_vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                    (_serialize_f32(embedding), overfetch),
                ).fetchall()
                vec_ranked = [r[0] for r in rows]

        # RRF fusion (k=60)
        k: int = 60
        rrf_scores: T.Dict[str, float] = {}
        for rank, mid in enumerate(fts_ranked):
            rrf_scores[mid] = rrf_scores.get(mid, 0) + 1.0 / (k + rank + 1)
        for rank, mid in enumerate(vec_ranked):
            rrf_scores[mid] = rrf_scores.get(mid, 0) + 1.0 / (k + rank + 1)

        if not rrf_scores:
            return []

        # Filter vec-only candidates against meta
        if vec_ranked:
            placeholders: str = ",".join("?" * len(rrf_scores))
            valid_rows = self._db.execute(
                f"SELECT id FROM memories_meta WHERE id IN ({placeholders}) AND {meta_where}",
                (*rrf_scores.keys(), *meta_params),
            ).fetchall()
            valid_ids: set = {r[0] for r in valid_rows}
            rrf_scores = {mid: s for mid, s in rrf_scores.items() if mid in valid_ids}

        # Load and boost
        results: T.List[SearchResult] = []
        now: datetime = datetime.now(timezone.utc)
        for mid, rrf in rrf_scores.items():
            path: Path = self.memories_dir / f"{mid}.json"
            if not path.exists():
                continue
            try:
                memory: Memory = Memory.from_json(path.read_text())
            except (json.JSONDecodeError, KeyError):
                continue
            age_days: float = max((now - memory.created).total_seconds() / 86400, 0)
            freshness: float = 0.5 ** (age_days / memory.type.halflife)
            score: float = rrf * memory.type.weight * memory.confidence * freshness
            results.append(SearchResult(memory=memory, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]
