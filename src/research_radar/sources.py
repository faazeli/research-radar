"""Paper sources. Each source has a pure `parse_*` function (unit-tested on fixtures)
and a thin `fetch_*` wrapper that does the HTTP call."""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime

import feedparser
import httpx

from .models import Paper

USER_AGENT = "research-radar/0.1 (https://github.com/faazeli/research-radar)"
ARXIV_API = "https://export.arxiv.org/api/query"
PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0, follow_redirects=True)


# --------------------------------------------------------------------------- arXiv


def parse_arxiv(atom_xml: str) -> list[Paper]:
    feed = feedparser.parse(atom_xml)
    papers = []
    for e in feed.entries:
        arxiv_id = re.sub(r"v\d+$", "", e.id.rsplit("/abs/", 1)[-1])
        published = datetime.strptime(e.published[:10], "%Y-%m-%d").date() if "published" in e else None
        papers.append(
            Paper(
                uid=f"arxiv:{arxiv_id}",
                source="arxiv",
                title=" ".join(e.title.split()),
                abstract=" ".join(e.summary.split()),
                authors=[a.name for a in e.get("authors", [])],
                venue="arXiv " + e.get("arxiv_primary_category", {}).get("term", "").strip(),
                published=published,
                url=f"https://arxiv.org/abs/{arxiv_id}",
            )
        )
    return papers


def fetch_arxiv(query: str, max_results: int = 25) -> list[Paper]:
    params = {
        "search_query": query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": max_results,
    }
    with _client() as c:
        r = c.get(ARXIV_API, params=params)
        r.raise_for_status()
    time.sleep(3)  # arXiv API etiquette: max one request every 3 seconds
    return parse_arxiv(r.text)


# --------------------------------------------------------------------------- PubMed

_MONTHS = {
    m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)
}


def _pubmed_date(article: ET.Element) -> date | None:
    d = article.find(".//Article/ArticleDate")
    if d is None:
        d = article.find(".//JournalIssue/PubDate")
    if d is None or d.findtext("Year") is None:
        return None
    month = d.findtext("Month") or "1"
    month_num = int(month) if month.isdigit() else _MONTHS.get(month[:3], 1)
    return date(int(d.findtext("Year")), month_num, int(d.findtext("Day") or 1))


def parse_pubmed(efetch_xml: str) -> list[Paper]:
    root = ET.fromstring(efetch_xml)
    papers = []
    for art in root.findall("PubmedArticle"):
        pmid = art.findtext(".//MedlineCitation/PMID")
        title = "".join(art.find(".//Article/ArticleTitle").itertext()).strip()
        abstract_parts = []
        for node in art.findall(".//Abstract/AbstractText"):
            label = node.get("Label")
            text = "".join(node.itertext()).strip()
            abstract_parts.append(f"{label}: {text}" if label else text)
        authors = []
        for a in art.findall(".//AuthorList/Author"):
            last, init = a.findtext("LastName"), a.findtext("Initials")
            if last:
                authors.append(f"{last} {init}" if init else last)
            elif a.findtext("CollectiveName"):
                authors.append(a.findtext("CollectiveName"))
        papers.append(
            Paper(
                uid=f"pubmed:{pmid}",
                source="pubmed",
                title=title,
                abstract=" ".join(abstract_parts),
                authors=authors,
                venue=art.findtext(".//Journal/Title"),
                published=_pubmed_date(art),
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            )
        )
    return papers


def fetch_pubmed(query: str, max_results: int = 25, days: int = 30) -> list[Paper]:
    with _client() as c:
        r = c.get(
            PUBMED_ESEARCH,
            params={
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "sort": "pub_date",
                "retmode": "json",
                "datetype": "pdat",
                "reldate": days,
            },
        )
        r.raise_for_status()
        ids = r.json()["esearchresult"]["idlist"]
        if not ids:
            return []
        time.sleep(0.4)  # NCBI: max 3 requests/second without an API key
        r = c.get(PUBMED_EFETCH, params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"})
        r.raise_for_status()
    return parse_pubmed(r.text)


FETCHERS = {"arxiv": fetch_arxiv, "pubmed": fetch_pubmed}
