import json
import os
import threading
import typing as T
from datetime import datetime, timezone
from hickey import api
from hickey import digest
from hickey.store import Memory, SearchResult
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse


mcp = FastMCP("hickey", port=8420)


# real-time interaction
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


# automation hooks
@mcp.custom_route("/hook", methods=["POST"])
async def hook(request: Request) -> JSONResponse:
    """Claude Code hook. Digests transcripts on session boundaries."""
    body: bytes = await request.body()
    data: dict = json.loads(body)
    # get transcript file
    transcript_path: str = data.get("transcript_path", "")
    if not transcript_path:
        return JSONResponse({})
    # register the project
    cwd: str = data.get("cwd", "")
    project: str = os.path.basename(cwd) if cwd else "unknown"
    digest.register(transcript_path, project)
    # summarize the transcript in background (avoid blocking the event loop)
    match data.get("hook_event_name", ""):
        case "SessionStart":
            threading.Thread(target=digest.digest_all, daemon=True).start()
            threading.Thread(target=api.cache.prune, daemon=True).start()
        case "PreCompact" | "SessionEnd":
            threading.Thread(target=digest.digest, args=(transcript_path,), daemon=True).start()
    return JSONResponse({})


@mcp.custom_route("/hook/pre-webfetch", methods=["POST"])
async def pre_webfetch(request: Request) -> JSONResponse:
    """PreToolUse hook for WebFetch. Returns cached content if available."""
    body: bytes = await request.body()
    data: dict = json.loads(body)
    url: str = data.get("tool_input", {}).get("url", "")
    if not url:
        return JSONResponse({})
    result = api.cache.get(url)
    if result is None:
        return JSONResponse({})
    entry, content = result
    text: str = content.decode("utf-8", errors="replace")
    age: int = int((datetime.now(timezone.utc) - entry.fetched_at).total_seconds())
    return JSONResponse({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": f"[hickey cache hit — fetched {age}s ago, {entry.size_bytes} bytes]\n\n{text}",
        }
    })


@mcp.custom_route("/hook/post-webfetch", methods=["POST"])
async def post_webfetch(request: Request) -> JSONResponse:
    """PostToolUse hook for WebFetch. Caches the fetched content."""
    body: bytes = await request.body()
    data: dict = json.loads(body)
    url: str = data.get("tool_input", {}).get("url", "")
    tool_output: str = data.get("tool_output", "")
    if url and tool_output:
        api.cache.put(url, tool_output.encode("utf-8"))
    return JSONResponse({})
