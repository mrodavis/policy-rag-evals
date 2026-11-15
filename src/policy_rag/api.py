"""FastAPI service. Loads the promoted index (written by the Airflow pipeline) or builds one from the corpus."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .generator import ClaudeGenerator, ExtractiveGenerator, Generator
from .index import build_index, load_index
from .pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


def default_pipeline() -> Pipeline:
    current = Path(os.environ.get("KB_ARTIFACTS_DIR", ROOT / "artifacts")) / "current" / "index.json"
    index = load_index(current) if current.exists() else build_index(Path(os.environ.get("KB_CORPUS_DIR", ROOT / "corpus")))
    generator: Generator = ClaudeGenerator() if os.environ.get("ANTHROPIC_API_KEY") else ExtractiveGenerator()
    return Pipeline(index, generator)


def create_app(pipeline: Pipeline | None = None) -> FastAPI:
    app = FastAPI(title="Harborline policy assistant (fictional)", version="1.0.0")
    state = {"pipeline": pipeline}

    def get() -> Pipeline:
        if state["pipeline"] is None:
            state["pipeline"] = default_pipeline()
        return state["pipeline"]

    @app.get("/health")
    def health() -> dict:
        p = get()
        return {"ok": True, "index_version": p.index.version, "documents": p.index.doc_ids, "generator": p.generator.model}

    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        return get().ask(req.question).to_dict()

    return app


app = create_app()
