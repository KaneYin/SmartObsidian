"""The Weft agent: retrieve relevant chunks, then reason over them with Claude,
citing sources. Wired as a minimal LangGraph StateGraph (retrieve -> reason),
per the locked architecture decision, while keeping each step a plain testable
function."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from weft.embeddings import Embedder
from weft.graph import LinkGraph
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


def graph_aware_retrieve(
    question: str,
    embedder: Embedder,
    store: VectorStore,
    graph: LinkGraph,
    k: int = 5,
    neighbor_budget: int = 5,
) -> list[SearchHit]:
    """Vector top-k as seeds, then the single best-scoring chunk from each
    1-hop neighbor note (diversified, budgeted). Base top-k is always kept and
    stays first; expansion chunks follow in score order."""
    query_vec = embedder.embed([question])[0]
    # Full score-desc scan of the store: cheap at M0/M1 scale and needed to pick
    # the single best chunk per neighbor note below. Revisit if vaults grow large.
    scored = store.search(query_vec, k=len(store))  # every chunk, score-desc
    base = scored[:k]

    def key(h: SearchHit) -> tuple[str, int]:
        return (h.metadata["rel_path"], h.metadata["ordinal"])

    base_keys = {key(h) for h in base}
    seed_notes = {h.metadata["rel_path"] for h in base}

    neighbor_notes: set[str] = set()
    for note in seed_notes:
        neighbor_notes |= graph.neighbors(note)
    neighbor_notes -= seed_notes

    expansion: list[SearchHit] = []
    used_notes: set[str] = set()
    for h in scored:  # already score-desc, so first hit per note is its best
        if len(expansion) >= neighbor_budget:
            break
        rp = h.metadata["rel_path"]
        # `key(h) not in base_keys` is defensive: neighbor_notes already excludes
        # seed notes, so a base chunk can't reach here — kept to make the dedup explicit.
        if rp in neighbor_notes and rp not in used_notes and key(h) not in base_keys:
            expansion.append(h)
            used_notes.add(rp)

    return base + expansion


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


def build_graph(embedder: Embedder, store: VectorStore, llm: LLMClient, link_graph: LinkGraph | None = None):
    """Compile the retrieve -> reason graph. When link_graph is provided, the
    retrieve node uses graph-aware retrieval; otherwise pure vector."""

    def _retrieve(state: AgentState) -> AgentState:
        if link_graph is not None:
            hits = graph_aware_retrieve(
                state["question"], embedder, store, link_graph, k=state.get("k", 5)
            )
        else:
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


def ask(
    question: str,
    embedder: Embedder,
    store: VectorStore,
    llm: LLMClient,
    k: int = 5,
    graph: LinkGraph | None = None,
) -> AskResult:
    app = build_graph(embedder, store, llm, link_graph=graph)
    final = app.invoke({"question": question, "k": k})
    hits = final.get("hits", [])
    sources: list[str] = []
    for h in hits:
        rp = h.metadata["rel_path"]
        if rp not in sources:
            sources.append(rp)
    return AskResult(answer=final["answer"], sources=sources)
