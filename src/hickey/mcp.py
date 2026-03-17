import hashlib
import json
import os
import subprocess
import typing as T
from hickey import api
from hickey.store import Memory, MemoryType, SearchResult
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse


EXTRACTION_PROMPT: str = """You are a memory extraction system. You ONLY output JSON lines. Never respond conversationally.

Extract ONLY information worth remembering across future coding sessions. Be extremely selective — most turns contain nothing worth storing.

Extract:
- Corrections: mistakes caught, wrong approaches identified
- Decisions: architectural or design choices with rationale
- Preferences: how the user likes things done
- Non-obvious findings: surprising API behavior, bug root causes

Do NOT extract:
- What code was written or changed (that's in git)
- Routine task completion or explanations of existing code
- Anything derivable from reading the codebase

For each memory, output one JSON object per line:
{"content": "...", "type": "correction|decision|fact|preference|investigation", "confidence": 0.8}

Output NOTHING if there's nothing worth keeping. Most turns produce nothing."""


mcp = FastMCP("hickey", port=8420)


@mcp.tool()
def index():
    """Rebuild index from markdown files."""
    api.store.index()
    return "Rebuilding index."


@mcp.tool()
def save(
    content: str,
    id: T.Optional[str] = None,
    type: str = "fact",
    project: T.Optional[str] = None,
    tags: T.Optional[list[str]] = None,
    confidence: float = 1.0,
) -> str:
    """Store a memory. Pass id to revise an existing one."""
    saved: Memory = api.save(content=content, id=id, type=type, project=project, tags=tags, confidence=confidence)
    return f"Stored {saved.id} ({saved.type.name.lower()})"


@mcp.tool()
def delete(
    id: str
) -> str:
    """Delete a memory by ID."""
    api.delete(id)
    return f"Deleted memory: {id}"


@mcp.tool()
def list(
    type: T.Optional[str] = None,
    project: T.Optional[str] = None,
    limit: int = 20,
) -> str:
    """Browse stored memories, newest first."""
    memories: T.List[Memory] = api.list(type=type, project=project, limit=limit)
    if memories:
        return "\n".join(map(str, memories))
    else:
        return "No matching memories found."


@mcp.tool()
def search(
    query: str,
    type: T.Optional[str] = None,
    project: T.Optional[str] = None,
    limit: int = 5,
) -> str:
    """Search memories by keyword and semantic similarity."""
    results: T.List[SearchResult] = api.search(query, type=type, project=project, limit=limit)
    if results:
        return "\n".join(map(str, results))
    else:
        return "No matching memories found."


def _extract_and_store(message: str, project: str) -> None:
    """Call Haiku to extract memories, store results. Runs in background thread."""
    print(f"[hook] extracting from {len(message)} chars, project={project}", flush=True)
    try:
        print("[hook] calling claude...", flush=True)
        result: subprocess.CompletedProcess = subprocess.run(
            ["/opt/homebrew/bin/claude", "-p", "--model", "haiku",
             f"{EXTRACTION_PROMPT}\n\n---\nCONVERSATION TURN TO ANALYZE:\n---\n\n{message}"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        print(f"[hook] claude returned rc={result.returncode}", flush=True)
    except subprocess.TimeoutExpired:
        print("[hook] claude timed out", flush=True)
        return
    except FileNotFoundError as exc:
        print(f"[hook] claude not found: {exc}", flush=True)
        return
    except Exception as exc:
        print(f"[hook] unexpected error: {exc}", flush=True)
        return
    if result.returncode != 0:
        print(f"[hook] claude stderr: {result.stderr[:500]}", flush=True)
        return
    if not result.stdout.strip():
        print("[hook] haiku returned empty", flush=True)
        return
    print(f"[hook] haiku output: {result.stdout[:500]}", flush=True)
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            data: dict = json.loads(line)
            api.save(
                content=data["content"],
                type=data.get("type", "fact"),
                confidence=float(data.get("confidence", 0.8)),
                project=project,
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            continue


_last_hash: str = ""


@mcp.custom_route("/hook", methods=["POST"])
async def hook(request: Request) -> PlainTextResponse:
    """Claude Code stop hook. Accepts raw JSON payload via HTTP POST."""
    global _last_hash
    body: bytes = await request.body()
    data: dict = json.loads(body)
    message: str = data.get("last_assistant_message", "")
    msg_hash: str = hashlib.md5(message.encode()).hexdigest()
    if msg_hash == _last_hash:
        print(f"[hook] duplicate, skipped", flush=True)
        return PlainTextResponse("skipped")
    _last_hash = msg_hash
    print(f"[hook] received {len(body)} bytes", flush=True)
    cwd: str = data.get("cwd", "")
    print(f"[hook] message={len(message)} chars, cwd={cwd}, stop_hook_active={data.get('stop_hook_active')}", flush=True)
    if len(message) < 100 or data.get("stop_hook_active"):
        print(f"[hook] skipped", flush=True)
        return PlainTextResponse("skipped")
    project: str = os.path.basename(cwd) if cwd else "unknown"
    print(f"[hook] submitting extraction for project={project}", flush=True)
    api.store._pool.submit(_extract_and_store, message, project)
    return PlainTextResponse("ok")
