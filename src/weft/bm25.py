"""A small hand-rolled Okapi BM25 lexical index, aligned to the vector store's rows
(row i corresponds to the i-th chunk added to the store). The lexical half of hybrid
retrieval; its ranking is fused with the vector ranking via reciprocal rank fusion."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from weft.security import secure_write_text

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    docs: list[list[str]]        # tokens per store row
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, texts: list[str]) -> "BM25Index":
        return cls(docs=[_tokenize(t) for t in texts])

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = _tokenize(query)
        N = len(self.docs)
        if N == 0 or not q:
            return []
        lengths = [len(d) for d in self.docs]
        avgdl = sum(lengths) / N or 1.0
        df: Counter = Counter()
        for d in self.docs:
            for t in set(d):
                df[t] += 1
        query_terms = [t for t in dict.fromkeys(q) if t in df]
        scores: list[float] = []
        for i, d in enumerate(self.docs):
            tf = Counter(d)
            dl = lengths[i]
            s = 0.0
            for t in query_terms:
                f = tf.get(t, 0)
                if not f:
                    continue
                idf = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1.0)
                s += idf * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / avgdl)
                )
            scores.append(s)
        ranked = sorted(range(N), key=lambda i: -scores[i])
        return [(i, scores[i]) for i in ranked if scores[i] > 0.0][:k]

    def save(self, path: Path) -> None:
        secure_write_text(
            Path(path),
            json.dumps({"k1": self.k1, "b": self.b, "docs": self.docs},
                       ensure_ascii=False),
        )

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(docs=data["docs"], k1=data.get("k1", 1.5), b=data.get("b", 0.75))
