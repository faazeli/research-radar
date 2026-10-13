"""YAML configuration: one or more topics, each with a brief and a set of source queries."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class SourceQuery(BaseModel):
    name: str = Field(description="'arxiv' or 'pubmed'")
    query: str
    max_results: int = 25


class Topic(BaseModel):
    name: str = Field(description="Short slug, used for storage and file names")
    title: str
    brief: str = Field(description="What the reader cares about; the LLM scores papers against this")
    sources: list[SourceQuery]
    min_relevance: int = 6
    max_in_digest: int = 10


class LLMConfig(BaseModel):
    provider: str = "anthropic"
    model: str | None = None


class Config(BaseModel):
    llm: LLMConfig = LLMConfig()
    store: str = "data/radar.sqlite"
    output_dir: str = "digests"
    topics: list[Topic]

    @classmethod
    def load(cls, path: str | Path) -> Config:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def topic(self, name: str) -> Topic:
        for t in self.topics:
            if t.name == name:
                return t
        raise KeyError(f"No topic named {name!r}; have {[t.name for t in self.topics]}")
