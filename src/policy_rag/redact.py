"""Mask card numbers and Social Security numbers before a question is searched, logged or sent to a model."""

from __future__ import annotations

import re

CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def redact(text: str) -> tuple[str, list[str]]:
    found: list[str] = []

    def card(m: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and luhn_ok(digits):
            found.append("card_number")
            return f"[CARD ending {digits[-4:]}]"
        return m.group(0)

    def ssn(_: re.Match[str]) -> str:
        found.append("ssn")
        return "[SSN]"

    return SSN.sub(ssn, CARD.sub(card, text)), found
