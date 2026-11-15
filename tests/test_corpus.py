import datetime as dt

import pytest

from policy_rag.chunking import chunk_document
from policy_rag.corpus import CorpusError, load_corpus, parse_document, validate_corpus

DOC = """---
id: {id}
title: Test
owner: QA
effective_date: {date}
trust: internal
{extra}---
## Section A

Alpha text.

## Section B

Beta text.
"""


def write(tmp_path, name, **kw):
    kw.setdefault("extra", "")
    (tmp_path / f"{name}.md").write_text(DOC.format(**kw))


def test_superseded_and_future_documents_are_excluded(corpus_dir):
    ids = {d.id for d in load_corpus(corpus_dir, dt.date(2026, 11, 15))}
    assert "fee-schedule-2026" in ids
    assert "fee-schedule-2025" not in ids, "the 2026 schedule supersedes 2025"

    before_2026 = {d.id for d in load_corpus(corpus_dir, dt.date(2025, 6, 1))}
    assert "fee-schedule-2025" in before_2026 and "fee-schedule-2026" not in before_2026


def test_validation_reports_every_problem(tmp_path):
    write(tmp_path, "a", id="dup", date="2026-01-01")
    write(tmp_path, "b", id="dup", date="2026-01-01", extra="supersedes: missing-doc\n")
    (tmp_path / "c.md").write_text("no front matter")
    problems = validate_corpus(sorted(tmp_path.glob("*.md")))
    joined = "\n".join(problems)
    assert "duplicate id 'dup'" in joined
    assert "supersedes unknown document 'missing-doc'" in joined
    assert "c.md: missing front matter" in joined
    with pytest.raises(CorpusError):
        load_corpus(tmp_path)


def test_bad_trust_level_and_date_are_rejected():
    with pytest.raises(CorpusError, match="trust must be one of"):
        parse_document(DOC.format(id="x", date="2026-01-01", extra="").replace("trust: internal", "trust: blog"))
    with pytest.raises(CorpusError, match="effective_date"):
        parse_document(DOC.format(id="x", date="soon", extra=""))


def test_chunks_follow_sections_and_carry_metadata():
    doc = parse_document(DOC.format(id="policy-x", date="2026-01-01", extra=""))
    chunks = chunk_document(doc)
    assert [c.id for c in chunks] == ["policy-x#section-a", "policy-x#section-b"]
    assert chunks[0].effective_date == "2026-01-01" and chunks[0].trust == "internal"


def test_long_sections_split_with_overlap():
    body = " ".join(f"w{i}" for i in range(400))
    doc = parse_document(DOC.format(id="long", date="2026-01-01", extra="").replace("Alpha text.", body))
    chunks = [c for c in chunk_document(doc, max_words=160, overlap=30) if c.section == "Section A"]
    assert len(chunks) == 3
    assert chunks[0].text.split()[-30:] == chunks[1].text.split()[:30]
