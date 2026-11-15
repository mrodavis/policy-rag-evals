"""Daily knowledge-base refresh for the Harborline policy assistant (fictional).

validate_corpus -> build_candidate -> evaluate_candidate -> decide -> promote | hold_back

A new build reaches agents only if it passes the eval gates (retrieval recall, refusal on
unanswerable questions, PII redaction, zero forbidden content) and does not regress retrieval
against the live build. Otherwise the live index stays in place and the reason is logged.
The steps themselves live in policy_rag.kb_pipeline and are unit-tested without Airflow.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import dag, get_current_context, task

REPO = Path(__file__).resolve().parents[1]
CORPUS_DIR = Path(os.environ.get("KB_CORPUS_DIR", REPO / "corpus"))
ARTIFACTS_DIR = Path(os.environ.get("KB_ARTIFACTS_DIR", REPO / "artifacts"))
GOLDEN_PATH = Path(os.environ.get("KB_GOLDEN_PATH", REPO / "evals" / "golden.yaml"))


@dag(
    dag_id="kb_refresh",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["rag", "evals"],
    doc_md=__doc__,
)
def kb_refresh():
    @task
    def validate_corpus() -> int:
        from policy_rag.kb_pipeline import validate

        return validate(CORPUS_DIR)

    @task
    def build_candidate(_documents: int) -> dict:
        from policy_rag.kb_pipeline import build_candidate as build

        run_id = get_current_context()["run_id"].replace(":", "_").replace("+", "_")
        return build(CORPUS_DIR, ARTIFACTS_DIR, run_id)

    @task
    def evaluate_candidate(candidate: dict) -> dict:
        from policy_rag.generator import ClaudeGenerator, ExtractiveGenerator
        from policy_rag.kb_pipeline import evaluate_candidate as evaluate

        # With an API key the gate measures Claude's answers; without one it still gates retrieval,
        # refusal, redaction and forbidden content using the offline generator.
        generator = ClaudeGenerator() if os.environ.get("ANTHROPIC_API_KEY") else ExtractiveGenerator()
        return evaluate(Path(candidate["path"]), GOLDEN_PATH, generator)

    @task.branch
    def decide(evaluation: dict) -> str:
        from policy_rag.kb_pipeline import read_manifest, should_promote

        ok, reason = should_promote(evaluation, read_manifest(ARTIFACTS_DIR))
        print(f"promotion decision: {ok} ({reason})")
        return "promote" if ok else "hold_back"

    @task
    def promote(candidate: dict, evaluation: dict) -> dict:
        from policy_rag.kb_pipeline import promote as do_promote

        return do_promote(candidate, evaluation, ARTIFACTS_DIR)

    @task
    def hold_back(evaluation: dict) -> None:
        # In production this pages the knowledge-base owner. The live index is untouched.
        print("candidate held back:", evaluation["failures"] or "retrieval regressed")

    docs = validate_corpus()
    candidate = build_candidate(docs)
    evaluation = evaluate_candidate(candidate)
    decide(evaluation) >> [promote(candidate, evaluation), hold_back(evaluation)]


kb_refresh()
