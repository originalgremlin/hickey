import hashlib
import json
import os
import typing as T
from hickey import api
from hickey.store import Memory, SearchResult
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse


mcp = FastMCP("hickey", port=8420)
last_hash: str = ""


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
    saved: Memory = api.save(content=content, id=id, type=type, project=project, tags=tags, auto=False, confidence=confidence)
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


@mcp.custom_route("/hook", methods=["POST"])
async def hook(request: Request) -> PlainTextResponse:
    """Claude Code stop hook. Stores assistant responses as memories."""
    global last_hash
    # parse request body
    body: bytes = await request.body()
    data: dict = json.loads(body)
    message: str = data.get("last_assistant_message", "")
    # filter immediate duplicates
    msg_hash: str = hashlib.md5(message.encode()).hexdigest()
    if msg_hash == last_hash:
        return PlainTextResponse("skipped")
    last_hash = msg_hash
    # filter short messages
    if len(message) < 100 or data.get("stop_hook_active"):
        return PlainTextResponse("skipped")
    # save the memory
    cwd: str = data.get("cwd", "")
    project: str = os.path.basename(cwd) if cwd else "unknown"
    api.save(content=message, project=project, type="auto", auto=True, confidence=0.8)
    print(f"[hook] stored {len(message)} chars for project={project}", flush=True)
    return PlainTextResponse("ok")
