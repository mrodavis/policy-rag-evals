"""Heading-aware chunking: one chunk per section, long sections split into overlapping windows."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .corpus import Document


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    title: str
    section: str
    effective_date: str
    trust: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def chunk_document(doc: Document, max_words: int = 160, overlap: int = 30) -> list[Chunk]:
    sections: list[tuple[str, list[str]]] = []
    current = "Overview"
    lines: list[str] = []
    for line in doc.body.splitlines():
        if line.startswith("## "):
            if any(l.strip() for l in lines):
                sections.append((current, lines))
            current, lines = line[3:].strip(), []
        elif line.startswith("# "):
            continue  # document title; carried on every chunk instead
        else:
            lines.append(line)
    if any(l.strip() for l in lines):
        sections.append((current, lines))

    chunks: list[Chunk] = []
    for section, body_lines in sections:
        words = " ".join(l.strip() for l in body_lines if l.strip()).split()
        step = max_words - overlap
        windows = [words[i : i + max_words] for i in range(0, max(len(words) - overlap, 1), step)] or [words]
        for n, window in enumerate(windows):
            chunks.append(
                Chunk(
                    id=f"{doc.id}#{_slug(section)}" + (f"-{n + 1}" if len(windows) > 1 else ""),
                    doc_id=doc.id,
                    title=doc.title,
                    section=section,
                    effective_date=doc.effective_date.isoformat(),
                    trust=doc.trust,
                    text=" ".join(window),
                )
            )
    return chunks
