import hashlib
import sqlite3
import typing as T
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_TTL: int = 7 * 86400


@dataclass
class CacheEntry:
    url: str
    url_hash: str
    content_type: str = ""
    size_bytes: int = 0
    ttl_seconds: int = DEFAULT_TTL
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    accessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def hash_url(url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()

    @property
    def is_expired(self) -> bool:
        age = (datetime.now(timezone.utc) - self.accessed_at).total_seconds()
        return age > self.ttl_seconds


class WebfetchCache:
    def __init__(self, base_dir: Path = Path.home() / ".hickey"):
        base_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir = base_dir
        self.db_path: Path = base_dir / "webfetch.db"
        self._init_db()

    def _init_db(self) -> None:
        self._db: sqlite3.Connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS webcache (
                url_hash     TEXT PRIMARY KEY,
                url          TEXT NOT NULL,
                content_type TEXT NOT NULL DEFAULT '',
                size_bytes   INTEGER NOT NULL DEFAULT 0,
                ttl_seconds  INTEGER NOT NULL DEFAULT 604800,
                fetched_at   TEXT NOT NULL,
                accessed_at  TEXT NOT NULL
            )
        """)
        self._db.commit()

    def _file_path(self, url_hash: str) -> Path:
        return self.base_dir / "cache" / "webfetch" / url_hash[:2] / url_hash[2:4] / url_hash

    def get(self, url: str) -> T.Optional[tuple[CacheEntry, bytes]]:
        """Return (entry, content) if cached and not expired, else None. Touching resets accessed_at."""
        url_hash: str = CacheEntry.hash_url(url)
        row = self._db.execute(
            "SELECT url, url_hash, content_type, size_bytes, ttl_seconds, fetched_at, accessed_at FROM webcache WHERE url_hash = ?",
            (url_hash,),
        ).fetchone()
        if not row:
            return None
        entry = CacheEntry(
            url=row[0], url_hash=row[1], content_type=row[2],
            size_bytes=row[3], ttl_seconds=row[4],
            fetched_at=datetime.fromisoformat(row[5]),
            accessed_at=datetime.fromisoformat(row[6]),
        )
        if entry.is_expired:
            return None
        path: Path = self._file_path(entry.url_hash)
        if not path.exists():
            return None
        # touch accessed_at
        now: str = datetime.now(timezone.utc).isoformat()
        self._db.execute("UPDATE webcache SET accessed_at = ? WHERE url_hash = ?", (now, url_hash))
        self._db.commit()
        entry.accessed_at = datetime.fromisoformat(now)
        return entry, path.read_bytes()

    def put(self, url: str, content: bytes, content_type: str = "", ttl_seconds: int = DEFAULT_TTL) -> CacheEntry:
        """Store content on disk and index metadata in SQLite."""
        url_hash: str = CacheEntry.hash_url(url)
        path: Path = self._file_path(url_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        now: str = datetime.now(timezone.utc).isoformat()
        self._db.execute(
            "INSERT OR REPLACE INTO webcache (url_hash, url, content_type, size_bytes, ttl_seconds, fetched_at, accessed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (url_hash, url, content_type, len(content), ttl_seconds, now, now),
        )
        self._db.commit()
        return CacheEntry(
            url=url, url_hash=url_hash, content_type=content_type,
            size_bytes=len(content), ttl_seconds=ttl_seconds,
            fetched_at=datetime.fromisoformat(now), accessed_at=datetime.fromisoformat(now),
        )

    def prune(self) -> int:
        """Delete expired cache entries. Returns number pruned."""
        now: datetime = datetime.now(timezone.utc)
        rows = self._db.execute("SELECT url_hash, ttl_seconds, accessed_at FROM webcache").fetchall()
        pruned: int = 0
        for url_hash, ttl, accessed_at_str in rows:
            accessed_at: datetime = datetime.fromisoformat(accessed_at_str)
            if (now - accessed_at).total_seconds() > ttl:
                path: Path = self._file_path(url_hash)
                if path.exists():
                    path.unlink()
                    for parent in [path.parent, path.parent.parent]:
                        try:
                            parent.rmdir()
                        except OSError:
                            break
                self._db.execute("DELETE FROM webcache WHERE url_hash = ?", (url_hash,))
                pruned += 1
        if pruned:
            self._db.commit()
        return pruned
