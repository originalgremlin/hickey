"""MCP server — 5 tools, no more."""

import json
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .model import Memory, MemoryType
from .store import MemoryStore

mcp = FastMCP("hickey")

_store: Optional[MemoryStore] = None


def _get_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store


def _detect_project() -> Optional[str]:
    """Best-effort project detection from cwd or env."""
    return os.environ.get("HICKEY_PROJECT") or os.path.basename(os.getcwd())


def _fmt_memory(m: Memory, score: Optional[float] = None) -> str:
    age_days = max(
        (m.updated.__class__.now(m.updated.tzinfo or __import__("datetime").timezone.utc) - m.created).days, 0
    )
    parts = [
        f"**[{m.type.value}]** {m.content[:200]}{'...' if len(m.content) > 200 else ''}",
        f"  id={m.id}  confidence={m.confidence}  age={age_days}d",
    ]
    if m.tags:
        parts[-1] += f"  tags={','.join(m.tags)}"
    if m.project:
        parts[-1] += f"  project={m.project}"
    if score is not None:
        parts[-1] += f"  score={score:.4f}"
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def remember(
    content: str,
    type: str = "fact",
    confidence: float = 1.0,
    tags: str = "",
    project: str = "",
    source: str = "manual",
) -> str:
    """Store a memory.

    Types: correction, decision, fact, preference, investigation.
    Tags: comma-separated. Project: auto-detected if empty.
    """
    store = _get_store()
    mem = Memory(
        content=content,
        type=MemoryType(type),
        confidence=confidence,
        tags=[t.strip() for t in tags.split(",") if t.strip()],
        project=project or _detect_project(),
        source=source,
    )
    saved = store.save(mem)
    return f"Stored {saved.id} ({saved.type.value}, confidence={saved.confidence})"


@mcp.tool()
def recall(
    query: str,
    limit: int = 5,
    type: str = "",
    project: str = "",
) -> str:
    """Search memories by keyword and semantic similarity.

    Returns ranked results with scores. Leave type/project empty for unfiltered.
    """
    store = _get_store()
    results = store.search(
        query,
        limit=limit,
        type_filter=MemoryType(type) if type else None,
        project=project or None,
    )
    if not results:
        return "No matching memories."
    return "\n\n".join(_fmt_memory(m, score=s) for m, s in results)


@mcp.tool()
def forget(id: str) -> str:
    """Delete a memory by ID."""
    store = _get_store()
    if store.delete(id):
        return f"Deleted {id}"
    return f"Not found: {id}"


@mcp.tool()
def audit(topic: str = "", project: str = "") -> str:
    """Introspect: what do I know? What's stale?

    Provide a topic for targeted search, or leave empty for overview.
    """
    store = _get_store()
    result = store.audit(
        topic=topic or None,
        project=project or None,
    )

    lines = []
    stats = result["stats"]
    lines.append(f"**Total memories:** {stats['total']}")
    if stats["by_type"]:
        lines.append("**By type:** " + ", ".join(f"{k}={v}" for k, v in stats["by_type"].items()))

    if result.get("relevant"):
        lines.append(f"\n**Relevant to '{topic}'** ({len(result['relevant'])} found):")
        for m in result["relevant"]:
            lines.append(f"- [{m.type.value}] {m.content[:120]}... (id={m.id})")

    if result["stale"]:
        lines.append(f"\n**Potentially stale** ({len(result['stale'])}):")
        for m in result["stale"]:
            lines.append(f"- [{m.type.value}] {m.content[:80]}... (id={m.id})")

    if result["expired"]:
        lines.append(f"\n**Expired** ({len(result['expired'])}):")
        for m in result["expired"]:
            lines.append(f"- [{m.type.value}] {m.content[:80]}... (id={m.id})")

    return "\n".join(lines)


@mcp.tool()
def list_memories(
    type: str = "",
    project: str = "",
    limit: int = 20,
) -> str:
    """Browse stored memories, newest first.

    Filter by type (correction/decision/fact/preference/investigation) or project.
    """
    store = _get_store()
    memories = store.list_memories(
        type_filter=MemoryType(type) if type else None,
        project=project or None,
        limit=limit,
    )
    if not memories:
        return "No memories found."
    return "\n\n".join(_fmt_memory(m) for m in memories)
