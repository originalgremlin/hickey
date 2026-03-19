# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Hickey is an agent memory system — SQLite-backed with hybrid FTS5 + vector search. Named after Rich Hickey: simplicity, earn each concept, remove what doesn't justify its cost.

## Build & Test

```bash
uv venv --python 3.12 .venv
uv pip install -e . pytest --python .venv/bin/python
.venv/bin/pytest tests/ -v
```

## Architecture

Five modules, same four verbs (save, delete, list, search):

- **store.py** — `Memory`, `MemoryType`, `SearchResult`, `MemoryStore`. SQLite is the single source of truth. Four tables: `memories` (primary), `memories_fts` (FTS5), `memories_vec` (sqlite-vec), `watermarks` (digest progress). Hybrid search with RRF fusion, scored by type weight × confidence × freshness decay.
- **api.py** — Four functions that bridge store to frontends. Handles string→MemoryType conversion. Shared by CLI and MCP.
- **mcp.py** — FastMCP server on port 8420 (streamable-http). Four MCP tools + `/hook` HTTP endpoint for SessionStart, PreCompact, and SessionEnd hooks.
- **cli.py** — Click CLI. Admin: `start`, `stop`. Memory: `save`, `delete`, `list`, `search`.
- **digest.py** — Reads Claude Code transcript JSONL, calls Haiku via `claude -p` to extract typed memories. Triggered by hooks on session boundaries. Tracks byte-offset watermarks per transcript to process incrementally.

## Design Constraints

- 4 MCP tools — fewer tools = less model confusion
- SQLite is the only storage — no files, no derived caches, no rebuild command
- Corrections > decisions > facts in search ranking and decay rate
- Digest uses Haiku to summarize — only stores what's worth remembering, not raw responses

## Code Style

- Start with the fewest types, fields, and functions that work. Don't add structure to handle cases that don't exist yet.
- If you're about to write a second data structure that mirrors the first, put the data on the first one instead.
- Default to concrete values, not Optional. Default to bool, not str. Default to inline, not extracted.
