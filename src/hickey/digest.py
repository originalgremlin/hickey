import json
import subprocess
from pathlib import Path
from hickey import api


MAX_TOOL_RESULT: int = 2000
MAX_TOOL_USE: int = 300
VALID_TYPES: set[str] = {"correction", "decision", "fact", "preference", "investigation"}
SKIP_MARKERS: tuple[str, ...] = (
    "<system-reminder>",
    "<local-command-caveat>",
    "<command-name>",
    "<local-command-stdout>",
    "You are a memory extraction system",
)

PROMPT: str = """You are a memory extraction system. Given a conversation between a user and an AI coding assistant working on the project "{project}", extract memories worth preserving for future sessions.

Memory types:
- correction: A mistake was caught and fixed. Include what went wrong, why, and the fix.
- decision: An architectural or design choice. Include what was chosen, what was rejected, and why.
- fact: A verified piece of information not obvious from code. API behavior, library quirks, system constraints.
- preference: How the user likes things done. Code style, tool choices, workflow habits.
- investigation: Research findings, comparisons, benchmarks, analysis results.

Rules:
- Only extract information useful in a FUTURE session with no access to this one.
- Each memory must be self-contained — a reader with no context should understand it.
- Prefer decisions and rationale over implementation details. Code changes are in git; the reasoning behind them is not.
- If a decision was made and then reversed in the same conversation, only store the final state.
- Skip: greetings, status updates, routine tool output, code that's already committed.
- Fewer, higher-quality memories. When in doubt, don't extract.
- Set confidence 0.5-1.0. Use 0.9+ only for things that were tested or verified.

Return ONLY a JSON array:
[{{"type": "decision", "content": "We chose X over Y because Z. The tradeoff was...", "confidence": 0.9}}]
Or if nothing worth storing: []"""


def digest(transcript_path: str, project: str) -> int:
    """Digest new content from a transcript. Returns memories stored."""
    offset: int = api.store.get_watermark(transcript_path)
    path: Path = Path(transcript_path).expanduser()
    if not path.exists():
        return 0
    with open(path, "r") as f:
        f.seek(offset)
        raw: str = f.read()
        end_offset: int = f.tell()
    conversation: str = _parse_transcript(raw) if raw.strip() else ""
    if not conversation:
        api.store.set_watermark(transcript_path, end_offset)
        return 0
    memories: list[dict] | None = _extract(conversation, project)
    if memories is None:
        return 0
    for mem in memories:
        api.save(
            content=mem["content"],
            type=mem.get("type", "fact"),
            project=project,
            confidence=mem.get("confidence", 0.8),
        )
    api.store.set_watermark(transcript_path, end_offset)
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
        match type := entry.get("type", ""):
            case "user" | "assistant":
                text = _parse_turn(entry)
            case _:
                continue
        if text and not any(marker in text for marker in SKIP_MARKERS):
            segments.append(f"{type}: {text}")
    return "\n\n".join(segments)


def _parse_turn(entry: dict) -> str:
    """Extract text and tool interactions from a transcript entry."""
    content = entry.get("message", {}).get("content", "")
    parts: list[str] = []
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    for block in content:
        if not isinstance(block, dict):
            continue
        match block.get("type"):
            case "text":
                parts.append(block["text"])
            case "tool_result":
                tool_result = str(block.get("content", ""))
                if len(tool_result) > MAX_TOOL_RESULT:
                    tool_result = tool_result[:MAX_TOOL_RESULT] + "\n[truncated]"
                parts.append(f"[tool_result: {tool_result}]")
            case "tool_use":
                name = block.get("name", "?")
                tool_use: str = json.dumps(block.get("input", {}))
                if len(tool_use) > MAX_TOOL_USE:
                    tool_use = tool_use[:MAX_TOOL_USE] + "..."
                parts.append(f"[tool: {name}({tool_use})]")
    return "\n".join(parts)


def _extract(conversation: str, project: str) -> list[dict] | None:
    """Call claude CLI to extract memories from conversation.
    Returns list on success (possibly empty), None on failure."""
    prompt: str = PROMPT.format(project=project)
    try:
        result: subprocess.CompletedProcess = subprocess.run(
            ["claude", "-p", "--model", "sonnet", "--system-prompt", prompt],
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
        return [
            m for m in memories
            if isinstance(m, dict)
            and "content" in m
            and m.get("type") in VALID_TYPES
        ]
    except json.JSONDecodeError:
        print(f"[digest] failed to parse: {text[:200]}", flush=True)
    return None
