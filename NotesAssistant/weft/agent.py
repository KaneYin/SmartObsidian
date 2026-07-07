"""The Weft agent: retrieve relevant chunks, then reason over them with Claude,
citing sources. Wired as a minimal LangGraph StateGraph (retrieve -> reason),
per the locked architecture decision, while keeping each step a plain testable
function."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from weft.embeddings import Embedder
from weft.llm import LLMClient
from weft.store import SearchHit, VectorStore

SYSTEM = (
    "You are Weft, an assistant that answers strictly from the user's notes. "
    "Use only the numbered sources provided. Cite them inline as [n]. "
    "If the sources do not contain the answer, say so plainly."
)


@dataclass
class AskResult:
    answer: str
    sources: list[str] = field(default_factory=list)


class AgentState(TypedDict, total=False):
    question: str
    k: int
    hits: list[SearchHit]
    answer: str


def retrieve(question: str, embedder: Embedder, store: VectorStore, k: int = 5) -> list[SearchHit]:
    query_vec = embedder.embed([question])[0]
    return store.search(query_vec, k=k)


def build_prompt(question: str, hits: list[SearchHit]) -> str:
    lines = ["Sources:"]
    for i, h in enumerate(hits, start=1):
        m = h.metadata
        lines.append(f"[{i}] {m['rel_path']} — {m['heading']}\n{m['text']}")
    lines.append("")
    lines.append(f"Question: {question}")
    lines.append("Answer using only the sources above, citing them as [n].")
    return "\n".join(lines)


def reason(question: str, hits: list[SearchHit], llm: LLMClient) -> str:
    if not hits:
        return "I couldn't find anything in your notes about that."
    return llm.complete(system=SYSTEM, prompt=build_prompt(question, hits))


def build_graph(embedder: Embedder, store: VectorStore, llm: LLMClient):
    """Compile the retrieve -> reason graph. State carries question/k in,
    answer/hits out. Dependencies are captured in the node closures."""

    def _retrieve(state: AgentState) -> AgentState:
        hits = retrieve(state["question"], embedder, store, k=state.get("k", 5))
        return {"hits": hits}

    def _reason(state: AgentState) -> AgentState:
        return {"answer": reason(state["question"], state["hits"], llm)}

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("reason", _reason)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "reason")
    graph.add_edge("reason", END)
    return graph.compile()


def ask(question: str, embedder: Embedder, store: VectorStore, llm: LLMClient, k: int = 5) -> AskResult:
    app = build_graph(embedder, store, llm)
    final = app.invoke({"question": question, "k": k})
    hits = final.get("hits", [])
    sources: list[str] = []
    for h in hits:
        rp = h.metadata["rel_path"]
        if rp not in sources:
            sources.append(rp)
    return AskResult(answer=final["answer"], sources=sources)
