import json
import subprocess
from pathlib import Path
from hickey import api


MAX_TOOL_OUTPUT: int = 2000
VALID_TYPES: set[str] = {"correction", "decision", "fact", "preference", "investigation"}

SKIP_MARKERS: tuple[str, ...] = (
    "<system-reminder>", "<local-command-caveat>", "<command-name>", "<local-command-stdout>",
    "You are a memory extraction system",
)

PROMPT: str = """You are a memory extraction system. Given a conversation between a user and an AI coding assistant, extract memories worth preserving for future sessions.

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
- Be detailed: include the problem, the root cause, the fix or decision, and the rationale. A future reader should understand not just WHAT but WHY.
- Each memory content MUST be 200-500 characters. If shorter, add more context. If longer, tighten.
- Fewer, higher-quality memories. When in doubt, don't store.
- Set confidence 0.5-1.0 based on how certain/validated the information is.

Return ONLY a JSON array:
[{"type": "decision", "content": "We chose X over Y because Z. The tradeoff was A vs B. This matters when...", "confidence": 0.9}]
Or if nothing worth storing: []"""


def digest(transcript_path: str, project: str) -> int:
    """Digest new content from a transcript. Returns memories stored."""
    offset: int = api.store.get_watermark(transcript_path) or 0
    p: Path = Path(transcript_path).expanduser()
    if not p.exists():
        return 0
    with open(p, "r") as f:
        f.seek(offset)
        raw: str = f.read()
        new_offset: int = f.tell()
    conversation: str = _parse_transcript(raw) if raw.strip() else ""
    if not conversation:
        api.store.set_watermark(transcript_path, new_offset)
        return 0
    memories: list[dict] | None = _extract(conversation)
    if memories is None:
        return 0
    for mem in memories:
        api.save(
            content=mem["content"],
            type=mem.get("type", "fact"),
            project=project,
            confidence=mem.get("confidence", 0.8),
        )
    api.store.set_watermark(transcript_path, new_offset)
    if memories:
        print(f"[digest] stored {len(memories)} memories for project={project}", flush=True)
    return len(memories)


def _parse_transcript(raw: str) -> str:
    """Parse JSONL transcript into a single conversation string."""
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
        elif t == "assistant":
            text = _parse_assistant(entry)
        else:
            continue
        if text and not any(m in text for m in SKIP_MARKERS):
            segments.append(f"{t}: {text}")
    return "\n\n".join(segments)


def _parse_user(entry: dict) -> str:
    """Extract text from a user transcript entry."""
    content = entry.get("message", {}).get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append(block["text"])
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
    try:
        result: subprocess.CompletedProcess = subprocess.run(
            ["claude", "-p", "--model", "sonnet", "--system-prompt", PROMPT],
            input=conversation,
            capture_output=True,
            text=True,
            timeout=300,
        )
        text: str = result.stdout.strip()
        if result.returncode != 0 or not text:
            stderr: str = result.stderr.strip()[:200] if result.stderr else ""
            print(f"[digest] claude returned code={result.returncode} stderr={stderr}", flush=True)
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"[digest] claude call failed: {e}", flush=True)
        return None
    # find JSON array in the response
    start: int = text.find("[")
    end: int = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        print(f"[digest] no JSON array found: {text[:200]}", flush=True)
        return None
    try:
        memories = json.loads(text[start:end + 1])
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
