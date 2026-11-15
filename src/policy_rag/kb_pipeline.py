"""Knowledge-base refresh steps. Plain functions so they are unit-tested without Airflow;
dags/kb_refresh.py wires them into a daily DAG."""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
from pathlib import Path

from .corpus import CorpusError, validate_corpus
from .evals import GATES, load_golden, run_evals
from .generator import ExtractiveGenerator, Generator
from .index import build_index, load_index, save_index


def validate(corpus_dir: Path) -> int:
    paths = sorted(corpus_dir.glob("*.md"))
    if not paths:
        raise CorpusError(f"no documents in {corpus_dir}")
    problems = validate_corpus(paths)
    if problems:
        raise CorpusError("\n".join(problems))
    return len(paths)


def build_candidate(corpus_dir: Path, artifacts_dir: Path, run_id: str, today: dt.date | None = None) -> dict:
    index = build_index(corpus_dir, today)
    path = artifacts_dir / "candidates" / f"{run_id}.json"
    save_index(index, path)
    return {"path": str(path), "version": index.version, "documents": len(index.doc_ids), "chunks": len(index.chunks)}


def evaluate_candidate(candidate_path: Path | str, golden_path: Path | str, generator: Generator | None = None) -> dict:
    run = run_evals(load_index(Path(candidate_path)), generator or ExtractiveGenerator(), load_golden(Path(golden_path)))
    return {"metrics": run["metrics"], "failures": run["failures"], "generator": run["generator"]}


def should_promote(evaluation: dict, current_manifest: dict | None) -> tuple[bool, str]:
    """Promote only if every gate passes and retrieval did not get worse than the live build."""
    if evaluation["failures"]:
        return False, "; ".join(evaluation["failures"])
    if current_manifest:
        live = current_manifest["metrics"]["retrieval_recall_at_5"]
        new = evaluation["metrics"]["retrieval_recall_at_5"]
        if new + 1e-9 < live:
            return False, f"retrieval_recall_at_5 regressed from {live:.2f} to {new:.2f}"
    return True, "all gates passed"


def read_manifest(artifacts_dir: Path) -> dict | None:
    path = artifacts_dir / "current" / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else None


def promote(candidate: dict, evaluation: dict, artifacts_dir: Path) -> dict:
    """Swap the candidate in atomically and keep the previous build for one-step rollback."""
    current = artifacts_dir / "current"
    current.mkdir(parents=True, exist_ok=True)
    previous = read_manifest(artifacts_dir)
    if previous and (current / "index.json").exists():
        history = artifacts_dir / "history"
        history.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current / "index.json", history / f"{previous['version']}.json")

    tmp = current / "index.json.tmp"
    shutil.copy2(candidate["path"], tmp)
    os.replace(tmp, current / "index.json")
    manifest = {
        "version": candidate["version"],
        "promoted_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "previous_version": previous["version"] if previous else None,
        "metrics": evaluation["metrics"],
        "gates": GATES,
        "generator": evaluation["generator"],
    }
    (current / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
