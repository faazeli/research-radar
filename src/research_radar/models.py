"""Core data models shared across the pipeline."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class Paper(BaseModel):
    """A single publication pulled from a source (arXiv, PubMed, ...)."""

    uid: str = Field(description="Globally unique id, e.g. 'arxiv:2401.01234' or 'pubmed:38123456'")
    source: str
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    venue: str | None = None
    published: date | None = None
    url: str

    def short_authors(self, n: int = 3) -> str:
        if not self.authors:
            return "Unknown authors"
        if len(self.authors) <= n:
            return ", ".join(self.authors)
        return ", ".join(self.authors[:n]) + " et al."


class Assessment(BaseModel):
    """LLM judgement of how relevant a paper is to the topic brief."""

    relevance: int = Field(ge=0, le=10, description="0 = unrelated, 10 = must read")
    rationale: str = Field(description="One or two sentences explaining the score")
    tags: list[str] = Field(default_factory=list, description="Up to 4 short topical tags")


class Summary(BaseModel):
    """Structured summary of a relevant paper."""

    tldr: str = Field(description="One-sentence plain-language takeaway")
    key_findings: list[str] = Field(description="2-4 concrete findings, with numbers when the abstract gives them")
    why_it_matters: str = Field(description="Why this matters for the reader's topic brief")


class ScoredPaper(BaseModel):
    paper: Paper
    assessment: Assessment
    summary: Summary | None = None
