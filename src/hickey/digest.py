import json
import subprocess
import threading
from pathlib import Path
from hickey import api


MAX_TOOL_OUTPUT: int = 2000
MAX_CHUNK: int = 20_000
_lock: threading.Lock = threading.Lock()
VALID_TYPES: set[str] = {"correction", "decision", "fact", "preference", "investigation"}


PROMPT: str = """You are a memory extraction system. Given a chunk of conversation between a user and an AI coding assistant, extract memories worth preserving for future sessions.

Memory types:
- correction: A mistake was identified. "Don't do X, do Y instead."
- decision: An architectural or design choice with rationale.
- fact: A verified piece of information. API behavior, library quirks, system constraints.
- preference: How the user likes things done. Code style, tool choices, workflow habits.
- investigation: Research findings, comparisons, analysis.

Rules:
- Only extract information useful in a FUTURE conversation with no access to this one.
- Skip: greetings, status updates, routine tool output, code already in files.
- Skip: anything derivable by reading the current codebase or git history.
- Each memory must be self-contained — understandable without this conversation.
- Fewer, higher-quality memories. When in doubt, don't store.
- Set confidence 0.5-1.0 based on how certain/validated the information is.

Return ONLY a JSON array:
[{"type": "decision", "content": "...", "confidence": 0.9}]
Or if nothing worth storing: []"""


def register(transcript_path: str, project: str) -> None:
    """Register a transcript for digesting. Idempotent."""
    if api.store.get_watermark(transcript_path) is None:
        api.store.set_watermark(transcript_path, 0, project)


def digest(transcript_path: str) -> int:
    """Digest new content from a specific transcript. Returns memories stored."""
    with _lock:
        wm: tuple[int, str] | None = api.store.get_watermark(transcript_path)
        if not wm:
            return 0
        offset, project = wm
        count, new_offset = _digest_one(transcript_path, project, offset)
        api.store.set_watermark(transcript_path, new_offset, project)
        return count


def digest_all() -> int:
    """Digest all registered transcripts. Returns total memories stored."""
    with _lock:
        total: int = 0
        for path, offset, project in api.store.all_watermarks():
            count, new_offset = _digest_one(path, project, offset)
            api.store.set_watermark(path, new_offset, project)
            total += count
        return total


def _digest_one(path: str, project: str, offset: int) -> tuple[int, int]:
    """Read new content from one transcript and extract memories.
    Chunks the conversation to avoid timeouts. All-or-nothing: if any chunk
    fails extraction, no memories are stored and the watermark stays put."""
    p: Path = Path(path).expanduser()
    if not p.exists():
        return 0, offset
    with open(p, "r") as f:
        f.seek(offset)
        raw: str = f.read()
        new_offset: int = f.tell()
    if not raw.strip():
        return 0, new_offset
    segments: list[str] = _parse_transcript(raw)
    if not segments:
        return 0, new_offset
    # Extract from all chunks first, store only if all succeed
    all_memories: list[dict] = []
    chunk: list[str] = []
    chunk_len: int = 0
    for seg in segments:
        if chunk and chunk_len + len(seg) > MAX_CHUNK:
            memories: list[dict] | None = _extract("\n\n".join(chunk))
            if memories is None:
                return 0, offset
            all_memories.extend(memories)
            chunk = []
            chunk_len = 0
        chunk.append(seg)
        chunk_len += len(seg)
    if chunk:
        memories = _extract("\n\n".join(chunk))
        if memories is None:
            return 0, offset
        all_memories.extend(memories)
    # All chunks succeeded — store everything
    for mem in all_memories:
        api.save(
            content=mem["content"],
            type=mem.get("type", "fact"),
            project=project,
            confidence=mem.get("confidence", 0.8),
        )
    if all_memories:
        print(f"[digest] stored {len(all_memories)} memories for project={project}", flush=True)
    return len(all_memories), new_offset


def _parse_transcript(raw: str) -> list[str]:
    """Parse JSONL transcript into conversation segments (one per turn)."""
    segments: list[str] = []
    for line in raw.strip().split("\n"):
        if not line.strip():
            continue
        try:
            entry: dict = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("isMeta"):
            continue
        t: str = entry.get("type", "")
        if t == "user":
            text: str = _parse_user(entry)
            if text:
                segments.append(f"user: {text}")
        elif t == "assistant":
            text = _parse_assistant(entry)
            if text:
                segments.append(f"assistant: {text}")
    return segments


def _parse_user(entry: dict) -> str:
    """Extract text from a user transcript entry."""
    content = entry.get("message", {}).get("content", "")
    if isinstance(content, str):
        if any(tag in content for tag in ("<system-reminder>", "<local-command-caveat>", "<command-name>", "<local-command-stdout>")):
            return ""
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text: str = block["text"]
                if "<system-reminder>" not in text:
                    parts.append(text)
            elif block.get("type") == "tool_result":
                result: str = str(block.get("content", ""))
                if len(result) > MAX_TOOL_OUTPUT:
                    result = result[:MAX_TOOL_OUTPUT] + "\n[truncated]"
                parts.append(f"[tool_result: {result}]")
        return "\n".join(parts)
    return ""


def _parse_assistant(entry: dict) -> str:
    """Extract text and tool calls from an assistant transcript entry."""
    content = entry.get("message", {}).get("content", [])
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append(block["text"])
        elif block.get("type") == "tool_use":
            name: str = block.get("name", "?")
            inp: str = json.dumps(block.get("input", {}))
            if len(inp) > 300:
                inp = inp[:300] + "..."
            parts.append(f"[tool: {name}({inp})]")
    return "\n".join(parts)


def _extract(conversation: str) -> list[dict] | None:
    """Call claude CLI to extract memories from conversation.
    Returns list on success (possibly empty), None on failure."""
    full_input: str = f"{PROMPT}\n\n---\n\n{conversation}"
    try:
        result: subprocess.CompletedProcess = subprocess.run(
            ["claude", "-p", "--model", "haiku"],
            input=full_input,
            capture_output=True,
            text=True,
            timeout=120,
        )
        text: str = result.stdout.strip()
        if result.returncode != 0 or not text:
            stderr: str = result.stderr.strip()[:200] if result.stderr else ""
            print(f"[digest] claude returned code={result.returncode} stderr={stderr}", flush=True)
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"[digest] claude call failed: {e}", flush=True)
        return None
    # strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        memories = json.loads(text)
        if isinstance(memories, list):
            return [
                m for m in memories
                if isinstance(m, dict)
                and "content" in m
                and m.get("type") in VALID_TYPES
            ]
    except json.JSONDecodeError:
        print(f"[digest] failed to parse: {text[:200]}", flush=True)
    return None
