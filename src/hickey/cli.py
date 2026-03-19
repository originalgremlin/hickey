import click
import os
import signal
import subprocess
import sys
import typing as T
from hickey import api
from hickey.store import Memory, MemoryType, SearchResult
from io import TextIOWrapper
from pathlib import Path


PIDFILE: Path = Path("/tmp/hickey.pid")
OUTFILE: Path = Path("/tmp/hickey.out")

def is_alive() -> tuple[bool, int]:
    try:
        pid: int = int(PIDFILE.read_text().strip())
        os.kill(pid, 0)
        return True, pid
    except FileNotFoundError:
        return False, 0
    except OSError:
        PIDFILE.unlink()
        return False, 0


@click.group(invoke_without_command=True)
def main():
    """A memory system that leaves a mark."""
    if click.get_current_context().invoked_subcommand is None:
        start()


# admin commands
@main.command()
def start():
    """Start MCP server as a background daemon."""
    alive, pid = is_alive()
    if alive:
        click.echo(f"Hickey server already running (pid {pid}).")
    else:
        logfile: TextIOWrapper = open(OUTFILE, "a")
        proc: subprocess.Popen = subprocess.Popen(
            [sys.executable, "-c", "from hickey.mcp import mcp; mcp.run(transport='streamable-http')"],
            stdout=logfile,
            stderr=logfile,
            start_new_session=True,
        )
        PIDFILE.write_text(str(proc.pid))
        click.echo(f"Hickey server started (pid {proc.pid}).")


@main.command()
def stop():
    """Stop the MCP server daemon."""
    alive, pid = is_alive()
    if alive:
        os.kill(pid, signal.SIGTERM)
        PIDFILE.unlink()
        click.echo(f"Hickey server stopped (pid {pid}).")
    else:
        click.echo("Hickey server is not running.")


# client commands
@main.command()
@click.argument("content")
@click.option("--id", required=False, default=None)
@click.option("--type", "type", type=click.Choice([t.name.lower() for t in MemoryType]), default="fact")
@click.option("--project", required=False, default=None)
@click.option("--tags", default="")
@click.option("--confidence", default=1.0)
def save(
    content: str,
    id: T.Optional[str],
    type: str,
    project: T.Optional[str],
    tags: str,
    confidence: float,
):
    """Store a memory. Pass --id to revise an existing one."""
    saved: Memory = api.save(content=content, id=id, type=type, project=project, tags=tags.split(",") if tags else [], confidence=confidence)
    click.echo(f"Stored {saved.id} ({saved.type.name.lower()})")


@main.command()
@click.argument("id")
def delete(
    id: str
):
    """Delete a memory by ID."""
    api.delete(id)
    click.echo(f"Deleted memory: {id}")


@main.command()
@click.option("--type", required=False, default=None)
@click.option("--project", required=False, default=None)
@click.option("--limit", default=20)
def list(
    type: T.Optional[str],
    project: T.Optional[str],
    limit: int,
):
    """Browse stored memories, newest first."""
    memories: T.List[Memory] = api.list(type=type, project=project, limit=limit)
    if memories:
        click.echo("\n".join(map(str, memories)))
    else:
        click.echo("No matching memories found.")


@main.command()
@click.argument("query")
@click.option("--type", required=False, default=None)
@click.option("--project", required=False, default=None)
@click.option("--limit", default=5)
def search(
    query: str,
    type: T.Optional[str],
    project: T.Optional[str],
    limit: int,
):
    """Search memories by keyword and semantic similarity."""
    results: T.List[SearchResult] = api.search(query, type=type, project=project, limit=limit)
    if results:
        click.echo("\n".join(map(str, results)))
    else:
        click.echo("No matching memories found.")


if __name__ == "__main__":
    main()
