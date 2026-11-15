"""Eval harness: runs the golden set through the real pipeline and scores retrieval, answers,
refusals, redaction and forbidden content. `python -m policy_rag.evals` writes a Markdown report
and exits 1 if a gate fails."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .generator import ClaudeGenerator, ExtractiveGenerator, Generator
from .index import Index, build_index, load_index
from .pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]

GATES = {
    "retrieval_recall_at_5": 0.9,
    "unanswerable_refusal_rate": 1.0,
    "redaction_rate": 1.0,
    "forbidden_content_hits": 0,
}


@dataclass
class CaseResult:
    id: str
    status: str
    retrieval_hit: bool | None
    reciprocal_rank: float | None
    facts_ok: bool | None
    forbidden: list[str] = field(default_factory=list)
    redaction_ok: bool | None = None
    refusal_ok: bool | None = None
    answer: str = ""


def load_golden(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["cases"]


def score_case(pipeline: Pipeline, case: dict) -> CaseResult:
    ans = pipeline.ask(case["question"])
    text = ans.answer.lower()
    relevant = set(case.get("relevant_docs", []))
    ranked_docs = [r["chunk_id"].split("#")[0] for r in ans.retrieved]

    hit = rr = facts = refusal = redaction = None
    if relevant:
        positions = [i for i, d in enumerate(ranked_docs[:5]) if d in relevant]
        hit = bool(positions)
        rr = 1 / (positions[0] + 1) if positions else 0.0
    if case.get("answerable", True):
        facts = ans.status == "answered" and all(f.lower() in text for f in case.get("must_include", [])) and (
            not case.get("must_include_any") or any(f.lower() in text for f in case["must_include_any"])
        )
        refusal = ans.status != "refused"
    else:
        refusal = ans.status == "refused"
    if case.get("expect_redaction"):
        redaction = case["expect_redaction"] in ans.redactions
    full_output = " ".join([ans.answer, *[str(c) for c in ans.citations]]).lower()
    forbidden = [f for f in case.get("must_not_include", []) if f.lower() in full_output]
    return CaseResult(case["id"], ans.status, hit, rr, facts, forbidden, redaction, refusal, ans.answer)


def run_evals(index: Index, generator: Generator, golden: list[dict]) -> dict:
    pipeline = Pipeline(index, generator)
    results = [score_case(pipeline, c) for c in golden]

    def rate(values: list[bool | None]) -> float:
        vals = [v for v in values if v is not None]
        return sum(vals) / len(vals) if vals else 1.0

    answerable = [r for r, c in zip(results, golden) if c.get("answerable", True)]
    unanswerable = [r for r, c in zip(results, golden) if not c.get("answerable", True)]
    rrs = [r.reciprocal_rank for r in results if r.reciprocal_rank is not None]
    metrics = {
        "cases": len(results),
        "retrieval_recall_at_5": rate([r.retrieval_hit for r in results]),
        "mrr": sum(rrs) / len(rrs) if rrs else 0.0,
        "answer_fact_accuracy": rate([r.facts_ok for r in answerable]),
        "answered_rate_on_answerable": rate([r.refusal_ok for r in answerable]),
        "unanswerable_refusal_rate": rate([r.refusal_ok for r in unanswerable]),
        "redaction_rate": rate([r.redaction_ok for r in results]),
        "forbidden_content_hits": sum(len(r.forbidden) for r in results),
        "blocked_by_guardrails": sum(r.status == "blocked" for r in results),
    }
    failures = [
        f"{k} = {metrics[k]:.2f} (needs {'<=' if k == 'forbidden_content_hits' else '>='} {v})"
        for k, v in GATES.items()
        if (metrics[k] > v if k == "forbidden_content_hits" else metrics[k] < v)
    ]
    return {"metrics": metrics, "failures": failures, "results": results, "generator": generator.model, "index_version": index.version}


def render_report(run: dict) -> str:
    m = run["metrics"]
    mock = run["generator"].startswith("extractive")
    pct = lambda x: f"{x * 100:.0f}%"  # noqa: E731
    rows = "\n".join(
        f"| {r.id} | {r.status} | {'' if r.retrieval_hit is None else ('yes' if r.retrieval_hit else 'NO')} | "
        f"{'' if r.facts_ok is None else ('yes' if r.facts_ok else 'no')} | {', '.join(r.forbidden) or ''} |"
        for r in run["results"]
    )
    return f"""# Eval report

Generator: `{run["generator"]}` · index: `{run["index_version"]}` · cases: {m["cases"]}
{"> Offline extractive generator: it quotes passages and is tuned to this set, so answer scores only show that the harness runs. Set ANTHROPIC_API_KEY to measure Claude." if mock else ""}

| Metric | Result | Gate |
|---|---|---|
| Retrieval recall@5 | {pct(m["retrieval_recall_at_5"])} | >= {pct(GATES["retrieval_recall_at_5"])} |
| MRR | {m["mrr"]:.2f} | |
| Answer fact accuracy | {pct(m["answer_fact_accuracy"])} | |
| Answered (answerable questions) | {pct(m["answered_rate_on_answerable"])} | |
| Refused (unanswerable questions) | {pct(m["unanswerable_refusal_rate"])} | 100% |
| PII redacted | {pct(m["redaction_rate"])} | 100% |
| Forbidden content in output | {m["forbidden_content_hits"]} | 0 |
| Blocked by guardrails | {m["blocked_by_guardrails"]} | |

| Case | Status | Retrieved relevant doc | Facts correct | Forbidden content |
|---|---|---|---|---|
{rows}
{chr(10).join(f"- GATE FAILED: {f}" for f in run["failures"])}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the policy assistant eval set.")
    parser.add_argument("--index", type=Path, help="index artifact to evaluate (default: build from corpus/)")
    parser.add_argument("--golden", type=Path, default=ROOT / "evals" / "golden.yaml")
    parser.add_argument("--offline", action="store_true", help="use the extractive generator even if ANTHROPIC_API_KEY is set")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    index = load_index(args.index) if args.index else build_index(ROOT / "corpus")
    generator: Generator = ExtractiveGenerator() if args.offline or not os.environ.get("ANTHROPIC_API_KEY") else ClaudeGenerator()
    run = run_evals(index, generator, load_golden(args.golden))
    report = render_report(run)
    out = args.report or ROOT / "evals" / f"report.{generator.name}.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    return 1 if run["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
