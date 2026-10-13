"""`ask`: retrieval-augmented Q&A over every paper the radar has screened.

Retrieval uses LlamaIndex's BM25 retriever (keyword-based, so no embedding model or
vector database is needed); generation uses the same LangChain chat model as the
agent. Answers cite papers as [1], [2], ... and list the sources."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever

from .models import ScoredPaper

ASK_SYSTEM = (
    "Answer the question using only the numbered papers provided. Cite them inline as [1], [2]. "
    "If the papers do not answer the question, say so plainly instead of guessing."
)


@dataclass
class Answer:
    text: str
    sources: list[ScoredPaper]


def build_retriever(papers: list[ScoredPaper], top_k: int = 5) -> BM25Retriever:
    nodes = [
        TextNode(text=f"{s.paper.title}\n\n{s.paper.abstract}", id_=s.paper.uid, metadata={"uid": s.paper.uid})
        for s in papers
    ]
    return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=min(top_k, len(nodes)))


def ask(question: str, papers: list[ScoredPaper], llm: BaseChatModel, top_k: int = 5) -> Answer:
    if not papers:
        return Answer("No papers in the store yet. Run `research-radar run` first.", [])
    by_uid = {s.paper.uid: s for s in papers}
    hits = build_retriever(papers, top_k).retrieve(question)
    sources = [by_uid[h.node.metadata["uid"]] for h in hits]
    context = "\n\n".join(
        f"[{i}] {s.paper.title} ({s.paper.venue}, {s.paper.published})\n{s.paper.abstract}"
        for i, s in enumerate(sources, 1)
    )
    reply = llm.invoke([SystemMessage(ASK_SYSTEM), HumanMessage(f"Papers:\n{context}\n\nQuestion: {question}")])
    return Answer(reply.content, sources)
