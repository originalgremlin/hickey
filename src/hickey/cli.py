import click
import typing as T
from hickey import api
from hickey.store import Memory, MemoryType, SearchResult


@click.group(invoke_without_command=True)
def main():
    """A memory system that leaves a mark."""
    if click.get_current_context().invoked_subcommand is None:
        serve()


# admin commands
@main.command()
def serve():
    """Start MCP server (HTTP)."""
    from hickey.mcp import mcp
    mcp.run(transport="streamable-http")


@main.command()
def index():
    """Rebuild index from markdown files."""
    api.store.index()
    click.echo("Rebuilding index.")


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
    saved: Memory = api.save(content=content, id=id, type=type, project=project, tags=tags.split(","), confidence=confidence)
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
