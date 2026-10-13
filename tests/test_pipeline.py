"""Offline tests: real arXiv/PubMed responses saved as fixtures, and a fake chat model
that returns scripted JSON, so the whole LangGraph pipeline runs without network or keys."""

import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from research_radar.ask import ask
from research_radar.config import SourceQuery, Topic
from research_radar.graph import build_graph
from research_radar.llm import extract_json, invoke_structured
from research_radar.models import Assessment
from research_radar.sources import parse_arxiv, parse_pubmed
from research_radar.store import Store

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def arxiv_papers():
    return parse_arxiv((FIX / "arxiv.xml").read_text(encoding="utf-8"))


@pytest.fixture
def pubmed_papers():
    return parse_pubmed((FIX / "pubmed.xml").read_text(encoding="utf-8"))


def test_parse_arxiv(arxiv_papers):
    assert len(arxiv_papers) == 3
    p = arxiv_papers[0]
    assert p.uid.startswith("arxiv:") and "v" not in p.uid.split(".")[-1]
    assert p.title and p.abstract and p.authors
    assert p.url.startswith("https://arxiv.org/abs/")


def test_parse_pubmed(pubmed_papers):
    assert len(pubmed_papers) == 3
    for p in pubmed_papers:
        assert p.uid.startswith("pubmed:")
        assert p.title
        assert p.url.startswith("https://pubmed.ncbi.nlm.nih.gov/")


def test_extract_json_handles_fences_and_chatter():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"a": {"b": 2}} hope that helps') == {"a": {"b": 2}}


def test_invoke_structured_retries_on_bad_json():
    llm = FakeListChatModel(responses=["not json at all", '{"relevance": 7, "rationale": "ok", "tags": ["x"]}'])
    a = invoke_structured(llm, "sys", "user", Assessment)
    assert a.relevance == 7


def _topic():
    return Topic(
        name="t", title="Test topic", brief="anything", min_relevance=6, sources=[SourceQuery(name="arxiv", query="q")]
    )


def test_graph_end_to_end(tmp_path, arxiv_papers):
    scores = [9, 3, 7]  # one per paper; the 3 should be filtered out
    responses = [json.dumps({"relevance": s, "rationale": "r", "tags": ["pinn"]}) for s in scores]
    responses += [json.dumps({"tldr": "t", "key_findings": ["f1", "f2"], "why_it_matters": "w"})] * 2
    llm = FakeListChatModel(responses=responses)
    store = Store(tmp_path / "radar.sqlite")
    graph = build_graph(llm, store, fetchers={"arxiv": lambda q, max_results: arxiv_papers}, workers=1)

    out = graph.invoke({"topic": _topic()})
    assert [s.assessment.relevance for s in out["relevant"]] == [9, 7]
    assert all(s.summary for s in out["relevant"])
    assert "3 new papers screened" in out["digest"] and "2 made the cut" in out["digest"]

    # Second run: everything already seen, so nothing is re-scored and no LLM calls are made.
    out2 = build_graph(
        FakeListChatModel(responses=["unused"]), store, fetchers={"arxiv": lambda q, max_results: arxiv_papers}
    ).invoke({"topic": _topic()})
    assert out2["new"] == [] and "Nothing new" in out2["digest"]
    assert len(store.all_scored("t")) == 3


def test_failing_source_does_not_crash(tmp_path):
    def boom(q, max_results):
        raise ConnectionError("down")

    graph = build_graph(FakeListChatModel(responses=["unused"]), Store(tmp_path / "s.sqlite"), fetchers={"arxiv": boom})
    assert "Nothing new" in graph.invoke({"topic": _topic()})["digest"]


def test_ask_retrieves_and_cites(tmp_path, arxiv_papers, pubmed_papers):
    store = Store(tmp_path / "s.sqlite")
    from research_radar.models import ScoredPaper

    store.save(
        "t",
        [ScoredPaper(paper=p, assessment=Assessment(relevance=8, rationale="r")) for p in arxiv_papers + pubmed_papers],
    )
    llm = FakeListChatModel(responses=["Answer citing [1]."])
    ans = ask(arxiv_papers[0].title, store.all_scored("t"), llm, top_k=3)
    assert ans.text == "Answer citing [1]."
    assert ans.sources[0].paper.uid == arxiv_papers[0].uid  # BM25 finds the exact-title match first
