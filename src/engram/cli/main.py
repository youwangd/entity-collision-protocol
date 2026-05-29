"""Engram CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from engram import __version__
from engram.core.config import Config
from engram.core.types import MemoryType, RecallContext
from engram.engine import Engram


def _get_engram(path: str | None = None) -> Engram:
    """Get an Engram instance from config."""
    if path:
        config = Config.minimal(path)
    else:
        config_path = Path("~/.engram/config.yaml").expanduser()
        if config_path.exists():
            config = Config.from_yaml(str(config_path))
        else:
            config = Config.minimal()
    return Engram(config, actor="cli")


@click.group()
@click.version_option(version=__version__, prog_name="engram")
def cli():
    """Engram — Neuroscience-inspired memory for AI agents."""
    pass


@cli.command()
@click.option("--path", "-p", default="~/.engram", help="Memory store path")
def init(path: str):
    """Initialize a new memory store."""
    config = Config.minimal(path)
    mem = Engram(config)
    mem.close()
    resolved = config.resolved_path
    click.echo(f"✓ Initialized engram at {resolved}")
    click.echo(f"  Events: {resolved / 'events.jsonl'}")
    click.echo(f"  Memory: {resolved / 'memory.db'}")
    click.echo(f"  Audit:  {resolved / 'audit.jsonl'}")


@cli.command()
@click.argument("content")
@click.option("--salience", "-s", type=float, default=0.0, help="Salience hint (0.0-1.0)")
@click.option("--type", "-t", "memory_type", type=click.Choice(["fact", "episode", "skill", "schema"]), default="fact")
@click.option("--path", "-p", default=None, help="Memory store path")
def remember(content: str, salience: float, memory_type: str, path: str | None):
    """Store a memory."""
    mem = _get_engram(path)
    try:
        mt = MemoryType(memory_type)
        event_id = mem.remember(content, salience=salience, memory_type=mt)
        click.echo(f"✓ Remembered ({event_id})")
    finally:
        mem.close()


@cli.command()
@click.argument("query")
@click.option("--limit", "-n", type=int, default=5, help="Max results")
@click.option("--depth", type=click.Choice(["L0", "L1", "L2"]), default="L1")
@click.option("--context", "-c", default=None, help="Context key=value pairs (e.g. task=deploy)")
@click.option("--include-faded", is_flag=True, help="Include faded memories")
@click.option("--include-suppressed", is_flag=True, help="Include suppressed memories")
@click.option("--path", "-p", default=None, help="Memory store path")
def recall(query: str, limit: int, depth: str, context: str | None, include_faded: bool, include_suppressed: bool, path: str | None):
    """Search memories."""
    mem = _get_engram(path)
    try:
        recall_ctx = None
        if context:
            parts = dict(p.split("=", 1) for p in context.split(",") if "=" in p)
            recall_ctx = RecallContext(task=parts.get("task"))
        results = mem.recall(
            query, limit=limit, depth=depth, context=recall_ctx,
            include_faded=include_faded, include_suppressed=include_suppressed,
        )
        if not results:
            click.echo("No memories found.")
            return
        for i, r in enumerate(results, 1):
            m = r.memory
            click.echo(f"\n{i}. [{m.type.value}] {m.content}")
            click.echo(f"   Score: {r.score:.3f} | Salience: {m.salience:.2f} | State: {m.state.value}")
            if depth in ("L1", "L2") and m.summary and m.summary != m.content:
                click.echo(f"   Summary: {m.summary}")
            if depth == "L2":
                click.echo(f"   ID: {m.id}")
                click.echo(f"   Created: {m.created_at.isoformat()}")
                click.echo(f"   Sources: {m.source_events}")
    finally:
        mem.close()


@cli.command()
@click.option("--id", "memory_id", default=None, help="Memory ID to forget")
@click.option("--query", "-q", default=None, help="Query to match memories to forget")
@click.option("--below", type=float, default=None, help="Forget memories with salience below threshold")
@click.option("--hard", is_flag=True, help="Hard delete (GDPR — permanent)")
@click.option("--path", "-p", default=None, help="Memory store path")
def forget(memory_id: str | None, query: str | None, below: float | None, hard: bool, path: str | None):
    """Suppress or delete memories."""
    if not memory_id and not query and below is None:
        click.echo("Error: provide --id, --query, or --below", err=True)
        sys.exit(1)
    mem = _get_engram(path)
    try:
        count = mem.forget(id=memory_id, query=query, hard=hard, below=below)
        action = "deleted" if hard else "suppressed"
        click.echo(f"✓ {count} memories {action}")
    finally:
        mem.close()


@cli.command()
@click.option("--path", "-p", default=None, help="Memory store path")
def status(path: str | None):
    """Show memory stats and health."""
    mem = _get_engram(path)
    try:
        s = mem.status()
        click.echo("Engram Status")
        click.echo("─" * 40)
        click.echo(f"  Total memories: {s['total_memories']}")
        click.echo(f"  Buffer events:  {s['buffer_events']}")
        click.echo(f"  Pins:           {s['pins']}")
        click.echo()
        click.echo("  By state:")
        for state, count in s["by_state"].items():
            if count > 0:
                click.echo(f"    {state}: {count}")
        click.echo()
        click.echo("  By type:")
        for mtype, count in s["by_type"].items():
            if count > 0:
                click.echo(f"    {mtype}: {count}")
    finally:
        mem.close()


@cli.command()
@click.option("--window", "-w", default=None, help="Consolidation window (e.g. 24h, 7d)")
@click.option("--path", "-p", default=None, help="Memory store path")
def consolidate(window: str | None, path: str | None):
    """Run the consolidation pipeline (process buffered events)."""
    mem = _get_engram(path)
    try:
        click.echo("Running consolidation...")
        report = mem.consolidate(window=window)
        click.echo("✓ Consolidation complete")
        click.echo(f"  Events processed: {report.events_processed}")
        click.echo(f"  Memories created: {report.memories_created}")
        click.echo(f"  Facts extracted:  {report.facts_extracted}")
        click.echo(f"  Duration:         {report.duration_ms}ms")
        if report.errors:
            click.echo(f"  Errors: {len(report.errors)}")
            for err in report.errors:
                click.echo(f"    ⚠ {err}")
    finally:
        mem.close()


@cli.command()
@click.option("--full", is_flag=True, help="Full rebuild (ignore snapshot)")
@click.option("--path", "-p", default=None, help="Memory store path")
def rebuild(full: bool, path: str | None):
    """Rebuild SQLite from event log (event sourcing recovery)."""
    mem = _get_engram(path)
    try:
        mode = "full" if full else "incremental"
        click.echo(f"Rebuilding from event log ({mode})...")
        count = mem.rebuild(incremental=not full)
        click.echo(f"✓ Rebuilt {count} memories from events")
    finally:
        mem.close()


@cli.command("context")
@click.option("--max-tokens", "-t", type=int, default=4096, help="Token budget")
@click.option("--path", "-p", default=None, help="Memory store path")
def active_context(max_tokens: int, path: str | None):
    """Show active context for prompt injection."""
    mem = _get_engram(path)
    try:
        ctx = mem.active_context(max_tokens=max_tokens)
        if ctx:
            click.echo(ctx)
        else:
            click.echo("(empty context)")
    finally:
        mem.close()


@cli.command()
@click.option("--path", "-p", default=None, help="Memory store path")
def export(path: str | None):
    """Export all memories as JSON."""
    mem = _get_engram(path)
    try:
        memories = mem.export_memories()
        click.echo(json.dumps(memories, indent=2))
    finally:
        mem.close()


@cli.command("import")
@click.argument("file", type=click.Path(exists=True))
@click.option("--path", "-p", default=None, help="Memory store path")
def import_data(file: str, path: str | None):
    """Import memories from a JSON backup."""
    mem = _get_engram(path)
    try:
        count = mem.import_from(file)
        click.echo(f"✓ Imported {count} memories")
    finally:
        mem.close()


@cli.command()
@click.option("--path", "-p", default=None, help="Memory store path")
def schemas(path: str | None):
    """List auto-generated schemas (patterns)."""
    mem = _get_engram(path)
    try:
        result = mem.schemas()
        if not result:
            click.echo("No schemas found. Run consolidation with enough facts to generate patterns.")
            return
        for i, s in enumerate(result, 1):
            click.echo(f"\n{i}. {s['content']}")
            click.echo(f"   Salience: {s['salience']:.2f} | Created: {s['created_at']}")
    finally:
        mem.close()


@cli.command("trace")
@click.argument("memory_id")
@click.option("--path", "-p", default=None, help="Memory store path")
def trace(memory_id: str, path: str | None):
    """Show full lineage trace for a memory."""
    mem = _get_engram(path)
    try:
        result = mem.trace(memory_id)
        if not result:
            click.echo(f"Memory {memory_id} not found.", err=True)
            return
        click.echo(json.dumps(result, indent=2, default=str))
    finally:
        mem.close()


@cli.command()
@click.option("--path", "-p", default=None, help="Memory store path")
def serve(path: str | None):
    """Start MCP server on stdin/stdout (JSON-RPC 2.0)."""
    from engram.mcp.server import MCPServer
    mem = _get_engram(path)
    try:
        server = MCPServer(mem)
        server.serve_stdio()
    except KeyboardInterrupt:
        pass
    finally:
        mem.close()


@cli.command()
@click.option("--trigger", "-t", default=None, help="Trigger emotion (joy, trust, fear, surprise, sadness, disgust, anger, anticipation)")
@click.option("--intensity", "-i", type=float, default=0.5, help="Emotion intensity 0.0-1.0")
@click.option("--mood", is_flag=True, help="Show only mood")
@click.option("--timeline", default=None, help="Show affect history (e.g. 7d, 30d)")
@click.option("--path", "-p", default=None, help="Memory store path")
def affect(trigger: str | None, intensity: float, mood: bool, timeline: str | None, path: str | None):
    """Show or trigger affect state (mood, emotions, temperament)."""
    mem = _get_engram(path)
    try:
        if trigger:
            mem.affect.trigger(trigger, intensity, trigger=f"cli:{trigger}")
            click.echo(f"✓ Triggered {trigger} (intensity={intensity})")

        if mood:
            m = mem.affect.mood()
            click.echo(f"Mood: {m['label']} (valence={m['valence']:.2f}, arousal={m['arousal']:.2f}, confidence={m['confidence']:.2f})")
            return

        if timeline:
            # Parse days from timeline string
            days_str = timeline.strip().lower().rstrip("d")
            try:
                days = int(days_str)
            except ValueError:
                days = 7
            history = mem.affect.history(limit=days * 10)
            if not history:
                click.echo("No affect history found.")
                return
            for entry in history[:20]:
                click.echo(f"  [{entry.get('ts', '?')}] {entry.get('type', '?')}: {json.dumps(entry.get('data', {}))}")
            return

        state = mem.affect.status()
        click.echo(f"\nMood: {state['mood_label']} (valence={state['mood_valence']:.2f}, arousal={state['mood_arousal']:.2f})")
        if state["active_emotions"]:
            click.echo("Active emotions:")
            for e in state["active_emotions"]:
                click.echo(f"  • {e['primary']} ({e['intensity']:.2f})")
        t = state["temperament"]
        click.echo("\nTemperament:")
        click.echo(f"  Novelty seeking:   {t['novelty_seeking']:.2f}")
        click.echo(f"  Harm avoidance:    {t['harm_avoidance']:.2f}")
        click.echo(f"  Reward dependence: {t['reward_dependence']:.2f}")
        click.echo(f"  Persistence:       {t['persistence']:.2f}")
    finally:
        mem.close()


if __name__ == "__main__":
    cli()
