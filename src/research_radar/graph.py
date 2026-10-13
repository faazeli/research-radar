"""The research-radar agent as a LangGraph state machine:

    fetch -> dedupe -> assess -> (any relevant?) -> summarize -> compose -> END
                                       \\-- no --> compose (empty digest)

Each node is a plain function over a typed state, so the graph is easy to test with
a fake chat model and to extend (e.g. add a "critic" node or a new source).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph

from .config import Topic
from .llm import invoke_structured
from .models import Assessment, Paper, ScoredPaper, Summary
from .sources import FETCHERS
from .store import Store

log = logging.getLogger(__name__)

ASSESS_SYSTEM = (
    "You are a meticulous research analyst screening new publications for a reader. "
    "Score how relevant the paper is to the reader's brief from 0 (unrelated) to 10 (must read). "
    "Be strict: methods papers in adjacent fields score low unless the brief asks for them. "
    "Base your judgement only on the title and abstract."
)
SUMMARY_SYSTEM = (
    "You write concise, accurate research briefs. Use only facts stated in the abstract; "
    "never invent numbers, datasets, or conclusions. If the abstract is vague, say so."
)


class RadarState(TypedDict, total=False):
    topic: Topic
    fetched: list[Paper]
    new: list[Paper]
    scored: list[ScoredPaper]
    relevant: list[ScoredPaper]
    digest: str


def _paper_block(p: Paper) -> str:
    return f"Title: {p.title}\nVenue: {p.venue}\nPublished: {p.published}\nAbstract: {p.abstract}"


def build_graph(llm: BaseChatModel, store: Store, fetchers: dict[str, Callable] | None = None, workers: int = 4):
    fetchers = fetchers or FETCHERS

    def fetch(state: RadarState) -> RadarState:
        t = state["topic"]
        papers: list[Paper] = []
        for src in t.sources:
            try:
                got = fetchers[src.name](src.query, max_results=src.max_results)
                log.info("fetched %d from %s", len(got), src.name)
                papers += got
            except Exception as err:  # noqa: BLE001 — one flaky source should not kill the run
                log.warning("source %s failed: %s", src.name, err)
        return {"fetched": papers}

    def dedupe(state: RadarState) -> RadarState:
        new = store.unseen(state["topic"].name, state["fetched"])
        log.info("%d new of %d fetched", len(new), len(state["fetched"]))
        return {"new": new}

    def assess(state: RadarState) -> RadarState:
        brief = state["topic"].brief

        def one(p: Paper) -> ScoredPaper:
            a = invoke_structured(
                llm, ASSESS_SYSTEM, f"Reader's brief:\n{brief}\n\nPaper:\n{_paper_block(p)}", Assessment
            )
            return ScoredPaper(paper=p, assessment=a)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            scored = list(pool.map(one, state["new"]))
        threshold = state["topic"].min_relevance
        relevant = sorted(
            (s for s in scored if s.assessment.relevance >= threshold),
            key=lambda s: s.assessment.relevance,
            reverse=True,
        )[: state["topic"].max_in_digest]
        return {"scored": scored, "relevant": relevant}

    def summarize(state: RadarState) -> RadarState:
        brief = state["topic"].brief

        def one(s: ScoredPaper) -> ScoredPaper:
            summ = invoke_structured(
                llm, SUMMARY_SYSTEM, f"Reader's brief:\n{brief}\n\nPaper:\n{_paper_block(s.paper)}", Summary
            )
            return s.model_copy(update={"summary": summ})

        with ThreadPoolExecutor(max_workers=workers) as pool:
            relevant = list(pool.map(one, state["relevant"]))
        by_uid = {s.paper.uid: s for s in relevant}
        scored = [by_uid.get(s.paper.uid, s) for s in state["scored"]]
        return {"relevant": relevant, "scored": scored}

    def compose(state: RadarState) -> RadarState:
        store.save(state["topic"].name, state.get("scored", []))
        return {"digest": render_digest(state["topic"], state.get("relevant", []), len(state.get("new", [])))}

    g = StateGraph(RadarState)
    for name, fn in [
        ("fetch", fetch),
        ("dedupe", dedupe),
        ("assess", assess),
        ("summarize", summarize),
        ("compose", compose),
    ]:
        g.add_node(name, fn)
    g.set_entry_point("fetch")
    g.add_edge("fetch", "dedupe")
    g.add_edge("dedupe", "assess")
    g.add_conditional_edges("assess", lambda s: "summarize" if s["relevant"] else "compose")
    g.add_edge("summarize", "compose")
    g.add_edge("compose", END)
    return g.compile()


def render_digest(topic: Topic, relevant: list[ScoredPaper], n_screened: int) -> str:
    lines = [
        f"# Research Radar: {topic.title}",
        (
            f"*{date.today():%B %d, %Y}* · {n_screened} new papers screened · {len(relevant)} made the cut "
            f"(relevance ≥ {topic.min_relevance}/10)"
        ),
        "",
    ]
    if not relevant:
        lines.append("_Nothing new cleared the relevance bar this time._")
        return "\n".join(lines)
    for i, s in enumerate(relevant, 1):
        p, a, sm = s.paper, s.assessment, s.summary
        lines += [
            f"## {i}. [{p.title}]({p.url})",
            (
                f"{p.short_authors()} · {p.venue or p.source} · {p.published or 'n.d.'} · "
                f"**relevance {a.relevance}/10** · {' '.join('`' + t + '`' for t in a.tags)}"
            ),
            "",
        ]
        if sm:
            lines += [f"**TL;DR:** {sm.tldr}", ""]
            lines += [f"- {k}" for k in sm.key_findings]
            lines += ["", f"**Why it matters:** {sm.why_it_matters}", ""]
        else:
            lines += [f"_{a.rationale}_", ""]
    return "\n".join(lines)
