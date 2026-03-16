"""CLI entry point.

    hickey              → start MCP server (stdio)
    hickey serve        → start MCP server (stdio)
    hickey store ...    → store a memory
    hickey recall ...   → search memories
    hickey list         → list memories
    hickey audit        → audit memories
    hickey rebuild      → rebuild index from markdown files
"""

import argparse
import sys

from .model import Memory, MemoryType
from .store import MemoryStore


def _print_memory(m: Memory) -> None:
    age = max((m.updated.__class__.now(m.updated.tzinfo) - m.created).days, 0)
    print(f"[{m.type.value}] {m.content[:120]}")
    print(f"  id={m.id}  confidence={m.confidence}  age={age}d  source={m.source}")
    if m.tags:
        print(f"  tags={','.join(m.tags)}")
    if m.project:
        print(f"  project={m.project}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hickey",
        description="A memory system that leaves a mark.",
    )
    parser.add_argument(
        "--dir", help="Base directory (default: ~/.hickey)", default=None
    )
    sub = parser.add_subparsers(dest="command")

    # serve
    sub.add_parser("serve", help="Start MCP server (stdio)")

    # store
    p_store = sub.add_parser("store", help="Store a memory")
    p_store.add_argument("content")
    p_store.add_argument("--type", default="fact", choices=[t.value for t in MemoryType])
    p_store.add_argument("--confidence", type=float, default=1.0)
    p_store.add_argument("--tags", default="")
    p_store.add_argument("--project", default="")
    p_store.add_argument("--source", default="cli")

    # recall
    p_recall = sub.add_parser("recall", help="Search memories")
    p_recall.add_argument("query")
    p_recall.add_argument("--limit", type=int, default=5)
    p_recall.add_argument("--type", default="")
    p_recall.add_argument("--project", default="")

    # list
    p_list = sub.add_parser("list", help="List memories")
    p_list.add_argument("--type", default="")
    p_list.add_argument("--project", default="")
    p_list.add_argument("--limit", type=int, default=20)

    # audit
    p_audit = sub.add_parser("audit", help="Audit memories")
    p_audit.add_argument("--topic", default="")
    p_audit.add_argument("--project", default="")

    # rebuild
    sub.add_parser("rebuild", help="Rebuild index from markdown files")

    args = parser.parse_args()

    # Default (no command) or explicit "serve" → MCP server
    if args.command is None or args.command == "serve":
        from .server import mcp

        mcp.run()
        return

    store = MemoryStore(base_dir=args.dir)

    if args.command == "store":
        mem = Memory(
            content=args.content,
            type=MemoryType(args.type),
            confidence=args.confidence,
            tags=[t.strip() for t in args.tags.split(",") if t.strip()],
            project=args.project or None,
            source=args.source,
        )
        saved = store.save(mem)
        print(f"Stored {saved.id} ({saved.type.value})")

    elif args.command == "recall":
        results = store.search(
            args.query,
            limit=args.limit,
            type_filter=MemoryType(args.type) if args.type else None,
            project=args.project or None,
        )
        if not results:
            print("No matching memories.")
        for m, score in results:
            print(f"({score:.4f})", end=" ")
            _print_memory(m)

    elif args.command == "list":
        memories = store.list_memories(
            type_filter=MemoryType(args.type) if args.type else None,
            project=args.project or None,
            limit=args.limit,
        )
        if not memories:
            print("No memories found.")
        for m in memories:
            _print_memory(m)

    elif args.command == "audit":
        result = store.audit(
            topic=args.topic or None,
            project=args.project or None,
        )
        stats = result["stats"]
        print(f"Total: {stats['total']}")
        if stats["by_type"]:
            print("By type:", ", ".join(f"{k}={v}" for k, v in stats["by_type"].items()))
        if result["stale"]:
            print(f"\nPotentially stale ({len(result['stale'])}):")
            for m in result["stale"]:
                print(f"  [{m.type.value}] {m.content[:80]}... (id={m.id})")
        if result["expired"]:
            print(f"\nExpired ({len(result['expired'])}):")
            for m in result["expired"]:
                print(f"  [{m.type.value}] {m.content[:80]}... (id={m.id})")
        if result.get("relevant"):
            print(f"\nRelevant to '{args.topic}' ({len(result['relevant'])}):")
            for m in result["relevant"]:
                _print_memory(m)

    elif args.command == "rebuild":
        count = store.rebuild_index()
        print(f"Rebuilt index: {count} memories indexed")


if __name__ == "__main__":
    main()
