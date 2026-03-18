# hickey

A memory system that leaves a mark.

Transparent memory capture for Claude Code. Every substantive response is stored, indexed with FTS5 + vector embeddings, and searchable with hybrid BM25 + semantic similarity. Memories are ranked by type, confidence, and freshness decay.

## Install

```bash
# Install the package
pip install git+https://github.com/originalgremlin/hickey.git

# Install the Claude Code plugin (auto-starts server, configures hooks)
/plugin marketplace add originalgremlin/hickey
/plugin install hickey
```

## Manual setup (without plugin)

```bash
# Start the server
hickey start

# Register MCP server with Claude Code
claude mcp add --transport http hickey http://localhost:8420/mcp

# Add stop hook to .claude/settings.local.json
# "Stop": [{"matcher": "", "hooks": [{"type": "command", "command": "curl -s -X POST http://localhost:8420/hook -H 'Content-Type: application/json' -d @-"}]}]
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

Every assistant response over 100 characters is automatically stored via a Claude Code stop hook. Memories are saved to SQLite with three indexes:

- **memories** — primary table with all fields
- **memories_fts** — FTS5 full-text search (porter stemming)
- **memories_vec** — vector embeddings (BAAI/bge-small-en-v1.5, 384-dim)

Search combines FTS5 BM25 scores with vector cosine similarity using Reciprocal Rank Fusion, then boosts by type weight, confidence, and freshness decay.

## Memory review agent

The plugin includes a `hickey` agent that periodically reviews, reclassifies, and prunes memories. Run it in the background while you work:

```
/agent hickey
```

It works through the full inventory oldest-first: reclassifies `auto` memories to specific types, adjusts confidence, deletes noise and duplicates, and reports a summary when done.

## Memory types

| Type | Weight | Half-life | Use |
|------|--------|-----------|-----|
| correction | 1.5 | 90 days | Mistakes caught, wrong approaches |
| decision | 1.2 | 60 days | Architectural/design choices |
| preference | 1.1 | 60 days | How the user likes things done |
| fact | 1.0 | 30 days | Verified information |
| investigation | 0.8 | 21 days | Research findings, comparisons |
| auto | 0.7 | 45 days | Unclassified (hook-captured) |

## License

MIT
