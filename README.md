# hickey

A memory system that leaves a mark.

Persistent memory for Claude Code. On session boundaries, Sonnet reads the full conversation transcript and extracts what's worth remembering — decisions, corrections, facts, preferences, investigations. Memories are indexed with FTS5 + vector embeddings and searchable with hybrid BM25 + semantic similarity, ranked by type weight, confidence, and freshness decay.

## Install

```bash
# Install the package (requires Python 3.11+)
uv tool install git+https://github.com/originalgremlin/hickey.git

# Start the server
hickey start

# Install the Claude Code plugin (configures hooks and MCP connection)
/plugin marketplace add originalgremlin/hickey
/plugin install hickey
/reload-plugins
```

## Manual setup (without plugin)

```bash
# Start the server
hickey start

# Register MCP server with Claude Code
claude mcp add --transport http hickey http://localhost:8420/mcp

# Add hooks to .claude/settings.local.json
# "SessionStart": [{"matcher": "", "hooks": [{"type": "command", "command": "hickey start 2>/dev/null || true"}, {"type": "command", "command": "curl -s -X POST http://localhost:8420/hook -H 'Content-Type: application/json' -d @-", "timeout": 120000}]}]
# "PreCompact": [{"matcher": "", "hooks": [{"type": "command", "command": "curl -s -X POST http://localhost:8420/hook -H 'Content-Type: application/json' -d @-", "timeout": 120000}]}]
# "SessionEnd": [{"matcher": "", "hooks": [{"type": "command", "command": "curl -s -X POST http://localhost:8420/hook -H 'Content-Type: application/json' -d @-", "timeout": 120000}]}]
```

## CLI

```bash
hickey start                    # start server daemon
hickey stop                     # stop server daemon
hickey save "content" --type decision --confidence 0.9
hickey delete <id>
hickey list --type correction --project myproject --limit 10
hickey search "database architecture"
```

## MCP tools

- **save** — store a memory (pass `id` to revise an existing one)
- **delete** — remove a memory by ID
- **list** — browse memories, newest first (filter by type, project)
- **search** — hybrid keyword + semantic search with ranked results

## How it works

On context compaction and session end, a hook sends the transcript path to the hickey server. The digest system reads new transcript content since the last watermark, passes it to Sonnet via `claude -p --system-prompt`, and stores any memories Sonnet extracts. Only what's worth remembering gets stored — not raw responses.

Memories live in SQLite (`~/.hickey/memory.db`) with three indexes:

- **memories** — primary table with all fields
- **memories_fts** — FTS5 full-text search (porter stemming)
- **memories_vec** — vector embeddings (BAAI/bge-small-en-v1.5, 384-dim)

Search combines FTS5 BM25 scores with vector cosine similarity using Reciprocal Rank Fusion, then boosts by type weight, confidence, and freshness decay.

## Memory types

| Type | Weight | Half-life | Use |
|------|--------|-----------|-----|
| correction | 1.5 | 90 days | Mistakes caught, wrong approaches |
| decision | 1.2 | 60 days | Architectural/design choices |
| preference | 1.1 | 60 days | How the user likes things done |
| fact | 1.0 | 30 days | Verified information |
| investigation | 0.8 | 21 days | Research findings, comparisons |

## License

MIT
