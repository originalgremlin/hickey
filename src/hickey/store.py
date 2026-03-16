"""
Markdown-first memory storage backed by a SQLite index.

Source of truth: markdown files in memories_dir.
Index (FTS5 + optional sqlite-vec): derived cache, rebuildable via rebuild_index().
"""

import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from .model import (
    HALF_LIVES,
    TYPE_WEIGHTS,
    Memory,
    MemoryType,
    id,
    _now,
)

# ---------------------------------------------------------------------------
# Optional vector support — gracefully degrades to FTS5-only
# ---------------------------------------------------------------------------

_sqlite_vec = None
_TextEmbedding = None

try:
    import sqlite_vec as _sqlite_vec  # type: ignore[no-redef]
except ImportError:
    pass

try:
    from fastembed import TextEmbedding as _TextEmbedding  # type: ignore[no-redef]
except ImportError:
    pass

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384


def _serialize_f32(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class MemoryStore:
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir) if base_dir else Path.home() / ".hickey"
        self.memories_dir = self.base_dir / "memories"
        self.db_path = self.base_dir / "index.db"
        self.memories_dir.mkdir(parents=True, exist_ok=True)

        self._embedder = None
        self._db: sqlite3.Connection = None  # type: ignore[assignment]
        self._init_db()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vectors_available(self) -> bool:
        return _sqlite_vec is not None and _TextEmbedding is not None

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        self._db = sqlite3.connect(str(self.db_path))
        self._db.execute("PRAGMA journal_mode=WAL")

        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS memories_meta (
                id          TEXT PRIMARY KEY,
                type        TEXT NOT NULL,
                confidence  REAL NOT NULL DEFAULT 1.0,
                created     TEXT NOT NULL,
                updated     TEXT NOT NULL,
                expires     TEXT,
                source      TEXT DEFAULT 'manual',
                project     TEXT,
                tags        TEXT DEFAULT ''
            )
            """
        )

        self._db.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                id, content, type, tags, project,
                tokenize='porter unicode61'
            )
            """
        )

        if self.vectors_available:
            self._db.enable_load_extension(True)
            _sqlite_vec.load(self._db)  # type: ignore[union-attr]
            self._db.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                    id   TEXT PRIMARY KEY,
                    embedding float[{EMBED_DIM}]
                )
                """
            )

        self._db.commit()

    def close(self) -> None:
        if self._db:
            self._db.close()

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def _get_embedder(self):
        if self._embedder is None and _TextEmbedding is not None:
            self._embedder = _TextEmbedding(EMBED_MODEL)
        return self._embedder

    def _embed(self, text: str) -> Optional[list[float]]:
        embedder = self._get_embedder()
        if embedder is None:
            return None
        results = list(embedder.embed([text]))
        return results[0].tolist()

    # ------------------------------------------------------------------
    # Markdown I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _to_markdown(memory: Memory) -> str:
        fm: dict = {
            "id": memory.id,
            "type": memory.type.value,
            "confidence": memory.confidence,
            "created": memory.created.isoformat(),
            "updated": memory.updated.isoformat(),
            "source": memory.source,
        }
        if memory.expires:
            fm["expires"] = memory.expires.isoformat()
        if memory.tags:
            fm["tags"] = memory.tags
        if memory.project:
            fm["project"] = memory.project
        return f"---\n{yaml.dump(fm, default_flow_style=False, sort_keys=False)}---\n\n{memory.content}\n"

    @staticmethod
    def _from_markdown(text: str) -> Memory:
        if not text.startswith("---"):
            raise ValueError("No frontmatter")
        _, fm_raw, body = text.split("---", 2)
        meta = yaml.safe_load(fm_raw)

        def _parse_dt(val):
            if isinstance(val, datetime):
                return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
            return datetime.fromisoformat(str(val))

        return Memory(
            id=meta["id"],
            content=body.strip(),
            type=MemoryType(meta["type"]),
            confidence=meta.get("confidence", 1.0),
            created=_parse_dt(meta["created"]),
            updated=_parse_dt(meta["updated"]),
            expires=_parse_dt(meta["expires"]) if meta.get("expires") else None,
            tags=meta.get("tags", []),
            source=meta.get("source", "manual"),
            project=meta.get("project"),
        )

    def _file_path(self, memory_id: str) -> Path:
        return self.memories_dir / f"{memory_id}.md"

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    def _index_memory(self, memory: Memory) -> None:
        """Insert a memory into the SQLite index (FTS + meta + optional vec)."""
        mid = memory.id

        self._db.execute("DELETE FROM memories_fts WHERE id = ?", (mid,))
        self._db.execute("DELETE FROM memories_meta WHERE id = ?", (mid,))

        self._db.execute(
            "INSERT INTO memories_fts (id, content, type, tags, project) VALUES (?, ?, ?, ?, ?)",
            (mid, memory.content, memory.type.value, " ".join(memory.tags), memory.project or ""),
        )
        self._db.execute(
            """INSERT INTO memories_meta
               (id, type, confidence, created, updated, expires, source, project, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                mid,
                memory.type.value,
                memory.confidence,
                memory.created.isoformat(),
                memory.updated.isoformat(),
                memory.expires.isoformat() if memory.expires else None,
                memory.source,
                memory.project,
                ",".join(memory.tags),
            ),
        )

        if self.vectors_available:
            embedding = self._embed(memory.content)
            if embedding:
                self._db.execute("DELETE FROM memories_vec WHERE id = ?", (mid,))
                self._db.execute(
                    "INSERT INTO memories_vec (id, embedding) VALUES (?, ?)",
                    (mid, _serialize_f32(embedding)),
                )

    def _remove_from_index(self, memory_id: str) -> None:
        self._db.execute("DELETE FROM memories_fts WHERE id = ?", (memory_id,))
        self._db.execute("DELETE FROM memories_meta WHERE id = ?", (memory_id,))
        if self.vectors_available:
            self._db.execute("DELETE FROM memories_vec WHERE id = ?", (memory_id,))

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save(self, memory: Memory) -> Memory:
        memory.updated = _now()
        self._file_path(memory.id).write_text(self._to_markdown(memory))
        self._index_memory(memory)
        self._db.commit()
        return memory

    def get(self, memory_id: str) -> Optional[Memory]:
        path = self._file_path(memory_id)
        if not path.exists():
            return None
        return self._from_markdown(path.read_text())

    def delete(self, memory_id: str) -> bool:
        path = self._file_path(memory_id)
        if not path.exists():
            return False
        path.unlink()
        self._remove_from_index(memory_id)
        self._db.commit()
        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 5,
        type_filter: Optional[MemoryType] = None,
        project: Optional[str] = None,
    ) -> list[tuple[Memory, float]]:
        """Hybrid search: FTS5 BM25 + optional vector similarity, fused with RRF."""
        overfetch = limit * 3

        # FTS5 ranked list
        fts_ranked: list[str] = []
        try:
            fts_query = " OR ".join(
                f'"{token}"' for token in query.split() if token.strip()
            )
            if fts_query:
                rows = self._db.execute(
                    "SELECT id FROM memories_fts WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?",
                    (fts_query, overfetch),
                ).fetchall()
                fts_ranked = [r[0] for r in rows]
        except sqlite3.OperationalError:
            pass

        # Vector ranked list
        vec_ranked: list[str] = []
        if self.vectors_available:
            embedding = self._embed(query)
            if embedding:
                rows = self._db.execute(
                    "SELECT id, distance FROM memories_vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                    (_serialize_f32(embedding), overfetch),
                ).fetchall()
                vec_ranked = [r[0] for r in rows]

        # RRF fusion (k=60)
        k = 60
        rrf_scores: dict[str, float] = {}
        for rank, mid in enumerate(fts_ranked):
            rrf_scores[mid] = rrf_scores.get(mid, 0) + 1.0 / (k + rank + 1)
        for rank, mid in enumerate(vec_ranked):
            rrf_scores[mid] = rrf_scores.get(mid, 0) + 1.0 / (k + rank + 1)

        if not rrf_scores:
            return []

        # Load, filter, boost
        now = _now()
        scored: list[tuple[Memory, float]] = []
        for mid, rrf in rrf_scores.items():
            memory = self.get(mid)
            if memory is None:
                continue
            if type_filter and memory.type != type_filter:
                continue
            if project and memory.project != project:
                continue
            if memory.expires and memory.expires < now:
                continue

            type_weight = TYPE_WEIGHTS.get(memory.type, 1.0)
            age_days = max((now - memory.created).total_seconds() / 86400, 0)
            half_life = HALF_LIVES.get(memory.type, 30)
            freshness = 0.5 ** (age_days / half_life)

            scored.append((memory, rrf * type_weight * memory.confidence * freshness))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    # ------------------------------------------------------------------
    # List & audit
    # ------------------------------------------------------------------

    def list_memories(
        self,
        type_filter: Optional[MemoryType] = None,
        project: Optional[str] = None,
        limit: int = 20,
    ) -> list[Memory]:
        q = "SELECT id FROM memories_meta WHERE 1=1"
        params: list = []
        if type_filter:
            q += " AND type = ?"
            params.append(type_filter.value)
        if project:
            q += " AND project = ?"
            params.append(project)
        q += " ORDER BY updated DESC LIMIT ?"
        params.append(limit)

        rows = self._db.execute(q, params).fetchall()
        return [m for mid in rows if (m := self.get(mid[0])) is not None]

    def audit(
        self,
        topic: Optional[str] = None,
        project: Optional[str] = None,
    ) -> dict:
        """Introspect: what do I know? What's stale? Stats."""
        now = _now()

        # Stats
        total = self._db.execute("SELECT COUNT(*) FROM memories_meta").fetchone()[0]
        by_type = {}
        for row in self._db.execute(
            "SELECT type, COUNT(*) FROM memories_meta GROUP BY type"
        ).fetchall():
            by_type[row[0]] = row[1]

        # Stale: memories past 80% of their half-life
        stale: list[Memory] = []
        expired: list[Memory] = []
        for md_file in self.memories_dir.glob("*.md"):
            try:
                m = self._from_markdown(md_file.read_text())
            except Exception:
                continue
            if project and m.project != project:
                continue
            if m.expires and m.expires < now:
                expired.append(m)
                continue
            age_days = (now - m.created).total_seconds() / 86400
            hl = HALF_LIVES.get(m.type, 30)
            if age_days > hl * 0.8:
                stale.append(m)

        result: dict = {
            "stats": {"total": total, "by_type": by_type},
            "stale": stale,
            "expired": expired,
        }

        if topic:
            result["relevant"] = [
                mem for mem, _score in self.search(topic, limit=10, project=project)
            ]

        return result

    # ------------------------------------------------------------------
    # Rebuild
    # ------------------------------------------------------------------

    def rebuild_index(self) -> int:
        """Rebuild the SQLite index from markdown files on disk."""
        self._db.close()
        if self.db_path.exists():
            self.db_path.unlink()
        self._init_db()

        count = 0
        for md_file in self.memories_dir.glob("*.md"):
            try:
                memory = self._from_markdown(md_file.read_text())
                self._index_memory(memory)
                count += 1
            except Exception as exc:
                print(f"warning: skipping {md_file.name}: {exc}")

        self._db.commit()
        return count
