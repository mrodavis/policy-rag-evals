"""Hybrid retrieval: BM25 for exact terms (fee names, numbers) plus a vector index, fused with
reciprocal rank fusion. Both are small and dependency-free so the whole pipeline runs offline."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from .chunking import Chunk

STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it my of on or the their there "
    "this to was what when where which who will with you your me we our get any customer asking asked much many "
    "tell should please did would could".split()
)


def tokenize(text: str) -> list[str]:
    text = re.sub(r"\[[A-Z]+[^\]]*\]", " ", text)  # drop redaction placeholders like [CARD ending 1111]
    tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    out = []
    for t in tokens:
        if t in STOPWORDS:
            continue
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]  # crude plural folding: "fees" -> "fee"
        out.append(t)
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.avg_len = sum(self.lengths) / max(len(docs), 1)
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for counts, length in zip(self.docs, self.lengths):
            s = 0.0
            for t in query:
                tf = counts.get(t, 0)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg_len))
            out.append(s)
        return out


class Embedder(Protocol):
    name: str

    def embed(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Offline stand-in for an embedding model: hashed unigrams and bigrams, L2-normalized.
    Swap in a hosted embedding model (and pgvector or another vector store) in production."""

    name = "hashing-512"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        tokens = tokenize(text)
        vec = [0.0] * self.dim
        for feature in tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]:
            h = int.from_bytes(hashlib.sha1(feature.encode()).digest()[:8], "big")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    bm25: float
    bm25_rank: int
    vector_rank: int


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None, vectors: list[list[float]] | None = None) -> None:
        self.chunks = chunks
        self.embedder = embedder or HashingEmbedder()
        self.bm25 = BM25([tokenize(f"{c.title} {c.section} {c.text}") for c in chunks])
        self.vectors = vectors if vectors is not None else [self.embedder.embed(f"{c.title} {c.section} {c.text}") for c in chunks]

    def search(self, query: str, k: int = 5, rrf_k: int = 60) -> list[Hit]:
        q = tokenize(query)
        lexical = self.bm25.scores(q)
        qv = self.embedder.embed(query)
        dense = [sum(a * b for a, b in zip(qv, v)) for v in self.vectors]

        lex_rank = {i: r for r, i in enumerate(sorted(range(len(lexical)), key=lambda i: -lexical[i]))}
        vec_rank = {i: r for r, i in enumerate(sorted(range(len(dense)), key=lambda i: -dense[i]))}
        fused = {i: 1 / (rrf_k + lex_rank[i] + 1) + 1 / (rrf_k + vec_rank[i] + 1) for i in range(len(self.chunks))}
        top = sorted(fused, key=lambda i: -fused[i])[:k]
        return [Hit(self.chunks[i], fused[i], lexical[i], lex_rank[i] + 1, vec_rank[i] + 1) for i in top]
