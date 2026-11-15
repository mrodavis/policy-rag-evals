import json
import shutil

import pytest
import yaml

from policy_rag.corpus import CorpusError
from policy_rag.kb_pipeline import build_candidate, evaluate_candidate, promote, read_manifest, should_promote, validate

from .conftest import ROOT, TODAY


def test_refresh_promotes_a_passing_build_and_keeps_the_previous_one(tmp_path, corpus_dir):
    golden = ROOT / "evals" / "golden.yaml"
    first = build_candidate(corpus_dir, tmp_path, "run1", TODAY)
    ev = evaluate_candidate(first["path"], golden)
    ok, reason = should_promote(ev, read_manifest(tmp_path))
    assert ok, reason
    m1 = promote(first, ev, tmp_path)
    assert (tmp_path / "current" / "index.json").exists() and m1["previous_version"] is None

    second = build_candidate(corpus_dir, tmp_path, "run2", TODAY)
    m2 = promote(second, ev, tmp_path)
    assert m2["previous_version"] == m1["version"]
    assert (tmp_path / "history" / f"{m1['version']}.json").exists(), "previous build kept for rollback"


def test_a_build_that_fails_a_gate_is_held_back(tmp_path, corpus_dir):
    # Remove the disputes policy: retrieval for dispute questions collapses below the gate.
    broken = tmp_path / "corpus"
    shutil.copytree(corpus_dir, broken)
    (broken / "disputes-policy.md").unlink()
    cand = build_candidate(broken, tmp_path, "bad", TODAY)
    ev = evaluate_candidate(cand["path"], ROOT / "evals" / "golden.yaml")
    ok, reason = should_promote(ev, None)
    assert not ok and "retrieval_recall_at_5" in reason


def test_retrieval_regression_against_the_live_build_blocks_promotion():
    live = {"metrics": {"retrieval_recall_at_5": 1.0}}
    ok, reason = should_promote({"failures": [], "metrics": {"retrieval_recall_at_5": 0.95}}, live)
    assert not ok and "regressed" in reason


def test_validation_fails_the_run_on_a_bad_document(tmp_path, corpus_dir):
    bad = tmp_path / "corpus"
    shutil.copytree(corpus_dir, bad)
    (bad / "broken.md").write_text("---\nid: broken\n---\n")
    with pytest.raises(CorpusError, match="broken.md"):
        validate(bad)
    assert validate(corpus_dir) == len(list(corpus_dir.glob("*.md")))
