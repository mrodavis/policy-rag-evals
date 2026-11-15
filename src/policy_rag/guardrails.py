"""Deterministic checks on a drafted answer. Anything that fails is blocked, never shown to an agent."""

from __future__ import annotations

import re

from .chunking import Chunk

NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")


def _norm(n: str) -> str:
    n = n.replace("$", "").replace(",", "").rstrip("%")
    return n[:-2] if n.endswith(".0") else n


def numbers_in(text: str) -> set[str]:
    return {_norm(m.group(0)) for m in NUMBER.finditer(text)}


def check_answer(answer: str, cited_ids: list[str], retrieved: dict[str, Chunk], refused: bool, question: str = "") -> list[str]:
    """Returns violations. An empty list means the answer can be shown."""
    if refused:
        return []
    if not cited_ids:
        return ["no_citations"]
    violations: list[str] = []
    unknown = [c for c in cited_ids if c not in retrieved]
    if unknown:
        violations.append(f"cites_unretrieved:{','.join(unknown)}")

    # Every figure in the answer (fees, rates, days, limits) must appear in a cited passage
    # (or in the question itself, e.g. "is there a late fee if the minimum is $8?").
    cited_text = " ".join(retrieved[c].text for c in cited_ids if c in retrieved)
    ungrounded = sorted(numbers_in(answer) - numbers_in(cited_text) - numbers_in(question))
    if ungrounded:
        violations.append(f"ungrounded_numbers:{','.join(ungrounded)}")

    if cited_ids and all(retrieved[c].trust == "partner" for c in cited_ids if c in retrieved) and re.search(
        r"\b(annual fee|apr|interest rate|late fee|liabilit|refund|waived)", answer, re.I
    ):
        # Partner-imported pages may describe partner perks, never Harborline's own fees or terms.
        violations.append("account_terms_from_partner_content")
    return violations
