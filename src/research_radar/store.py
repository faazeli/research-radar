"""SQLite store: remembers which papers were already seen and keeps scored papers
so later runs only surface new work and `ask` can search the history."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from .models import Paper, ScoredPaper

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    uid        TEXT PRIMARY KEY,
    topic      TEXT NOT NULL,
    seen_at    TEXT NOT NULL,
    relevance  INTEGER,
    payload    TEXT NOT NULL      -- ScoredPaper (or bare Paper) as JSON
);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.executescript(SCHEMA)

    def unseen(self, topic: str, papers: list[Paper]) -> list[Paper]:
        seen = {row[0] for row in self.db.execute("SELECT uid FROM papers WHERE topic = ?", (topic,))}
        out, batch = [], set()
        for p in papers:
            if p.uid not in seen and p.uid not in batch:
                out.append(p)
                batch.add(p.uid)
        return out

    def save(self, topic: str, scored: list[ScoredPaper]) -> None:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        self.db.executemany(
            "INSERT OR REPLACE INTO papers (uid, topic, seen_at, relevance, payload) VALUES (?, ?, ?, ?, ?)",
            [(s.paper.uid, topic, now, s.assessment.relevance, s.model_dump_json()) for s in scored],
        )
        self.db.commit()

    def all_scored(self, topic: str | None = None, min_relevance: int = 0) -> list[ScoredPaper]:
        q, args = "SELECT payload FROM papers WHERE relevance >= ?", [min_relevance]
        if topic:
            q += " AND topic = ?"
            args.append(topic)
        return [ScoredPaper.model_validate(json.loads(r[0])) for r in self.db.execute(q, args)]

    def close(self) -> None:
        self.db.close()
