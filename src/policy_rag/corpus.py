"""Load policy documents (Markdown with YAML front matter) and drop superseded versions."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import yaml

REQUIRED_FIELDS = ("id", "title", "owner", "effective_date", "trust")
TRUST_LEVELS = ("internal", "partner")


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    owner: str
    effective_date: dt.date
    trust: str
    body: str
    supersedes: str | None = None


class CorpusError(ValueError):
    """A document failed validation. The message lists every problem found."""


def parse_document(text: str, source: str = "<string>") -> Document:
    if not text.startswith("---\n"):
        raise CorpusError(f"{source}: missing front matter")
    try:
        _, front, body = text.split("---\n", 2)
    except ValueError as err:
        raise CorpusError(f"{source}: unterminated front matter") from err
    meta = yaml.safe_load(front) or {}

    problems = [f"missing '{f}'" for f in REQUIRED_FIELDS if f not in meta]
    if "trust" in meta and meta["trust"] not in TRUST_LEVELS:
        problems.append(f"trust must be one of {TRUST_LEVELS}")
    if "effective_date" in meta and not isinstance(meta["effective_date"], dt.date):
        problems.append("effective_date must be a YYYY-MM-DD date")
    if not body.strip():
        problems.append("empty body")
    if problems:
        raise CorpusError(f"{source}: " + "; ".join(problems))

    return Document(
        id=str(meta["id"]),
        title=str(meta["title"]),
        owner=str(meta["owner"]),
        effective_date=meta["effective_date"],
        trust=str(meta["trust"]),
        supersedes=meta.get("supersedes"),
        body=body.strip(),
    )


def validate_corpus(paths: list[Path]) -> list[str]:
    """Every problem in the corpus, so a bad publish fails with the full list instead of the first error."""
    problems: list[str] = []
    seen: dict[str, Path] = {}
    docs: list[Document] = []
    for path in paths:
        try:
            doc = parse_document(path.read_text(encoding="utf-8"), source=path.name)
        except CorpusError as err:
            problems.append(str(err))
            continue
        if doc.id in seen:
            problems.append(f"{path.name}: duplicate id '{doc.id}' (also in {seen[doc.id].name})")
        seen[doc.id] = path
        docs.append(doc)
    ids = {d.id for d in docs}
    for d in docs:
        if d.supersedes and d.supersedes not in ids:
            problems.append(f"{d.id}: supersedes unknown document '{d.supersedes}'")
    return problems


def load_corpus(directory: Path, today: dt.date | None = None) -> list[Document]:
    """Active documents only: future-dated documents are held back and superseded versions are dropped."""
    paths = sorted(directory.glob("*.md"))
    problems = validate_corpus(paths)
    if problems:
        raise CorpusError("\n".join(problems))
    today = today or dt.date.today()
    docs = [parse_document(p.read_text(encoding="utf-8"), p.name) for p in paths]
    live = [d for d in docs if d.effective_date <= today]
    superseded = {d.supersedes for d in live if d.supersedes}
    return [d for d in live if d.id not in superseded]
