"""Question in, cited and checked answer out."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Literal

from .generator import PROMPT_HASH, Generator
from .guardrails import check_answer
from .index import Index
from .redact import redact
from .retrieval import Hit, tokenize

REFUSAL = "The policies I have do not cover this. Please check with a supervisor."
BLOCKED = "I could not produce an answer I can verify against the policies. Please check with a supervisor."


@dataclass
class Answer:
    status: Literal["answered", "refused", "blocked"]
    answer: str
    citations: list[dict]
    retrieved: list[dict]
    redactions: list[str]
    violations: list[str] = field(default_factory=list)
    model: str = ""
    prompt_hash: str = PROMPT_HASH
    index_version: str = ""
    latency_ms: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def query_coverage(question: str, hits: list[Hit]) -> float:
    """Share of the question's content words found anywhere in the top passages."""
    q = set(tokenize(question))
    if not q:
        return 0.0
    vocab = set()
    for h in hits:
        vocab |= set(tokenize(f"{h.chunk.title} {h.chunk.section} {h.chunk.text}"))
    return len(q & vocab) / len(q)


class Pipeline:
    def __init__(
        self, index: Index, generator: Generator, k: int = 5, min_bm25: float = 2.5, min_coverage: float = 0.5, log_path: str | None = None
    ) -> None:
        self.index = index
        self.retriever = index.retriever()
        self.generator = generator
        self.k = k
        self.min_bm25 = min_bm25
        self.min_coverage = min_coverage
        self.log_path = log_path or os.environ.get("DECISION_LOG_PATH")

    def ask(self, question: str) -> Answer:
        started = time.monotonic()
        clean, redactions = redact(question)
        hits = self.retriever.search(clean, k=self.k)
        retrieved = [{"chunk_id": h.chunk.id, "score": round(h.score, 4), "bm25": round(h.bm25, 3)} for h in hits]
        by_id = {h.chunk.id: h.chunk for h in hits}

        def done(status, text, cited=(), violations=()):
            answer = Answer(
                status=status,
                answer=text,
                citations=[
                    {"chunk_id": c, "doc_id": by_id[c].doc_id, "title": by_id[c].title, "section": by_id[c].section, "effective_date": by_id[c].effective_date}
                    for c in cited
                ],
                retrieved=retrieved,
                redactions=redactions,
                violations=list(violations),
                model=self.generator.model,
                index_version=self.index.version,
                latency_ms=int((time.monotonic() - started) * 1000),
            )
            self._log(clean, answer)
            return answer

        # Nothing close enough: refuse without spending a model call. One shared word ("weather" in
        # "delays caused by a known weather event") is not enough; most of the question must be covered.
        if not hits or max(h.bm25 for h in hits) < self.min_bm25 or query_coverage(clean, hits[:3]) <= self.min_coverage:
            return done("refused", REFUSAL)

        draft = self.generator.draft(clean, hits)
        if draft.refused:
            return done("refused", REFUSAL)
        violations = check_answer(draft.answer, draft.citations, by_id, refused=False, question=clean)
        if violations:
            return done("blocked", BLOCKED, violations=violations)
        return done("answered", draft.answer, cited=draft.citations)

    def _log(self, question: str, answer: Answer) -> None:
        if not self.log_path:
            return
        row = {"question": question, **{k: v for k, v in answer.to_dict().items() if k != "retrieved"}}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
