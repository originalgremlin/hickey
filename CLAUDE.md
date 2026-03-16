# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Hickey is an agent memory system — markdown-first, with a SQLite-derived index (FTS5 + optional sqlite-vec). Named after Rich Hickey: simplicity, earn each concept, remove what doesn't justify its cost.

## Build & Test

```bash
uv venv --python 3.12 .venv
uv pip install -e . pytest --python .venv/bin/python      # core (FTS5 only)
uv pip install -e ".[vectors]" --python .venv/bin/python   # + vector search
.venv/bin/pytest tests/ -v                                  # run all tests
.venv/bin/pytest tests/test_store.py::TestSearch -v         # run one class
```

## Architecture

- **model.py** — `Memory` dataclass, `MemoryType` enum, type weights and half-lives for ranked decay
- **store.py** — `MemoryStore`: markdown file I/O, SQLite index (FTS5 + optional vec0), hybrid search with RRF fusion, rebuild-from-files
- **server.py** — MCP server (FastMCP) exposing 5 tools: `remember`, `recall`, `forget`, `audit`, `list_memories`
- **__main__.py** — CLI entry point; no-arg or `serve` starts MCP server, subcommands mirror the tools

Source of truth is always the markdown files in `~/.hickey/memories/`. The SQLite index is a derived cache rebuilt via `hickey rebuild`.

## Design Constraints

- 5-6 MCP tools max — fewer tools = less model confusion
- No external deps beyond Python + SQLite for core; vector search (sqlite-vec + fastembed) is optional
- Corrections > decisions > facts in search ranking and decay rate

## Code Style

- Start with the fewest types, fields, and functions that work. Don't add structure to handle cases that don't exist yet.
- If you're about to write a second data structure that mirrors the first, put the data on the first one instead.
- Default to concrete values, not Optional. Default to bool, not str. Default to inline, not extracted.
