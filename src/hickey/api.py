import typing as T
from hickey.store import Memory, MemoryType, MemoryStore, SearchResult


store: MemoryStore = MemoryStore()


def save(
    content: str,
    id: T.Optional[str] = None,
    type: str = "fact",
    project: T.Optional[str] = None,
    tags: T.Optional[list[str]] = None,
    confidence: float = 1.0,
) -> Memory:
    memory: Memory = Memory(content=content, type=MemoryType[type.upper()], tags=tags or [], confidence=confidence)
    memory.id = id or memory.id
    memory.project = project or memory.project
    return store.save(memory)


def delete(
    id: str
) -> None:
    store.delete(id)


def list(
    type: T.Optional[str] = None,
    project: T.Optional[str] = None,
    limit: int = 20,
) -> T.List[Memory]:
    return store.list(
        type=MemoryType[type.upper()] if type else None,
        project=project,
        limit=limit,
    )


def search(
    query: str,
    type: T.Optional[str] = None,
    project: T.Optional[str] = None,
    limit: int = 5,
) -> T.List[SearchResult]:
    return store.search(
        query,
        type=MemoryType[type.upper()] if type else None,
        project=project,
        limit=limit,
    )
