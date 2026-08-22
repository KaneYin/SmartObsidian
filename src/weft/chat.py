"""Conversational session over the retrieval + memory stack. Keeps a rolling window
of recent turns, answers each turn from notes + memory + conversation context, logs
an episode, and supports explicit `/remember`. Composed from agent primitives."""

from __future__ import annotations

from dataclasses import dataclass, field

from weft.agent import SYSTEM, build_prompt, collect_memory, graph_aware_retrieve, retrieve
from weft.memory import SEMANTIC_TYPES

MAX_MEMORY_TEXT = 2000


@dataclass
class ChatTurn:
    answer: str
    sources: list[str] = field(default_factory=list)


class ChatSession:
    def __init__(self, embedder, store, llm, *, graph=None, memory=None,
                 k: int = 5, window: int = 6):
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._graph = graph
        self._memory = memory
        self._k = k
        self._window = window
        self.history: list[dict] = []
        self.last_sources: list[str] = []

    def _retrieve(self, question: str):
        if self._graph is not None:
            return graph_aware_retrieve(question, self._embedder, self._store,
                                        self._graph, k=self._k)
        return retrieve(question, self._embedder, self._store, k=self._k)

    def send(self, question: str) -> ChatTurn:
        hits = self._retrieve(question)
        mem = (collect_memory(self._memory, self._embedder, question, k=self._k)
               if self._memory is not None else None)
        prompt = build_prompt(question, hits, memory=mem, conversation=list(self.history))
        answer = self._llm.complete(system=SYSTEM, prompt=prompt)

        sources: list[str] = []
        for h in hits:
            rp = h.metadata["rel_path"]
            if rp not in sources:
                sources.append(rp)

        self.history.append({"role": "user", "text": question})
        self.history.append({"role": "assistant", "text": answer})
        maxlen = self._window * 2
        if len(self.history) > maxlen:
            self.history = self.history[-maxlen:]
        self.last_sources = sources
        if self._memory is not None:
            self._memory.log_episode(question, answer, sources)
        return ChatTurn(answer=answer, sources=sources)

    def remember(self, type: str, text: str):
        if self._memory is None:
            raise RuntimeError("memory is disabled")
        if type not in SEMANTIC_TYPES:
            raise ValueError(f"type must be one of: {', '.join(sorted(SEMANTIC_TYPES))}")
        if not text or len(text) > MAX_MEMORY_TEXT:
            raise ValueError(f"text must be 1-{MAX_MEMORY_TEXT} characters")
        return self._memory.remember(type, text)

    def reset(self) -> None:
        self.history = []
