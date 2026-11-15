"""A versioned index artifact: chunks plus vectors, with a content hash so every answer can be traced
to the exact knowledge-base build that produced it."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .chunking import Chunk, chunk_document
from .corpus import load_corpus
from .retrieval import Embedder, HashingEmbedder, HybridRetriever


@dataclass
class Index:
    version: str
    content_hash: str
    embedder: str
    built_at: str
    chunks: list[Chunk]
    vectors: list[list[float]]

    def retriever(self, embedder: Embedder | None = None) -> HybridRetriever:
        e = embedder or HashingEmbedder()
        if e.name != self.embedder:
            raise ValueError(f"index was built with {self.embedder}, not {e.name}")
        return HybridRetriever(self.chunks, e, self.vectors)

    @property
    def doc_ids(self) -> list[str]:
        return sorted({c.doc_id for c in self.chunks})


def build_index(corpus_dir: Path, today: dt.date | None = None, embedder: Embedder | None = None) -> Index:
    embedder = embedder or HashingEmbedder()
    docs = load_corpus(corpus_dir, today)
    chunks = [c for d in docs for c in chunk_document(d)]
    digest = hashlib.sha256(json.dumps([c.to_dict() for c in chunks], sort_keys=True).encode()).hexdigest()[:12]
    return Index(
        version=f"kb-{digest}",
        content_hash=digest,
        embedder=embedder.name,
        built_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        chunks=chunks,
        vectors=[embedder.embed(f"{c.title} {c.section} {c.text}") for c in chunks],
    )


def save_index(index: Index, path: Path) -> None:
    """Write to a temp file then rename, so a reader never sees a half-written index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "version": index.version,
        "content_hash": index.content_hash,
        "embedder": index.embedder,
        "built_at": index.built_at,
        "chunks": [c.to_dict() for c in index.chunks],
        "vectors": [[round(v, 6) for v in vec] for vec in index.vectors],
    }
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def load_index(path: Path) -> Index:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Index(
        version=data["version"],
        content_hash=data["content_hash"],
        embedder=data["embedder"],
        built_at=data["built_at"],
        chunks=[Chunk(**c) for c in data["chunks"]],
        vectors=data["vectors"],
    )
