"""Command-line interface: `research-radar run`, `research-radar ask`, `research-radar sources`."""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from .ask import ask as ask_papers
from .config import Config
from .graph import build_graph
from .llm import get_chat_model
from .sources import FETCHERS
from .store import Store

# Paper titles and digests contain non-ASCII text (≥, Greek letters, accents). Legacy Windows
# consoles default to cp1252 and would crash on them, so always write UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(help="An LLM agent that watches arXiv and PubMed for you and writes a digest.", no_args_is_help=True)
console = Console()

ConfigOpt = typer.Option("config.yaml", "--config", "-c", help="Path to config YAML")


def _llm(cfg: Config, provider: str | None, model: str | None, json_mode: bool = False):
    return get_chat_model(provider or cfg.llm.provider, model or cfg.llm.model, json_mode=json_mode)


@app.command()
def run(
    config: Path = ConfigOpt,
    topic: str = typer.Option(None, "--topic", "-t", help="Run a single topic (default: all)"),
    provider: str = typer.Option(None, help="Override LLM provider: anthropic | openai | ollama | ..."),
    model: str = typer.Option(None, help="Override model name"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Fetch new papers, score and summarize them, and write a Markdown digest per topic."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(message)s")
    cfg = Config.load(config)
    store = Store(cfg.store)
    graph = build_graph(_llm(cfg, provider, model, json_mode=True), store)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in [cfg.topic(topic)] if topic else cfg.topics:
        with console.status(f"Scanning [bold]{t.title}[/]..."):
            result = graph.invoke({"topic": t})
        path = out_dir / f"{date.today():%Y-%m-%d}_{t.name}.md"
        path.write_text(result["digest"], encoding="utf-8")
        console.print(Markdown(result["digest"]))
        console.print(f"[green]Saved[/] {path}")
    store.close()


@app.command()
def ask(
    question: str,
    config: Path = ConfigOpt,
    topic: str = typer.Option(None, "--topic", "-t"),
    min_relevance: int = typer.Option(4, help="Only search papers scored at least this high (0 = everything)"),
    provider: str = typer.Option(None),
    model: str = typer.Option(None),
):
    """Ask a question; answered from the papers the radar has already screened, with citations."""
    cfg = Config.load(config)
    store = Store(cfg.store)
    ans = ask_papers(question, store.all_scored(topic, min_relevance), _llm(cfg, provider, model))
    console.print(Markdown(ans.text))
    for i, s in enumerate(ans.sources, 1):
        console.print(f"[dim][{i}] {s.paper.title} — {s.paper.url}[/]")
    store.close()


@app.command()
def sources(config: Path = ConfigOpt):
    """Dry run: show how many papers each source query returns (no LLM calls)."""
    cfg = Config.load(config)
    table = Table("topic", "source", "query", "papers")
    for t in cfg.topics:
        for s in t.sources:
            n = len(FETCHERS[s.name](s.query, max_results=s.max_results))
            table.add_row(t.name, s.name, s.query, str(n))
    console.print(table)


def main():
    app()
