"""DAG integrity checks. Skipped when Airflow is not installed (CI runs them in a separate job)."""

from pathlib import Path

import pytest

pytest.importorskip("airflow")

try:  # Airflow 3.4+
    from airflow.dag_processing.dagbag import DagBag  # noqa: E402
except ImportError:  # earlier Airflow 3.x
    from airflow.models.dagbag import DagBag  # noqa: E402

DAGS = Path(__file__).resolve().parents[1] / "dags"


def test_dag_imports_cleanly_and_has_the_expected_shape():
    bag = DagBag(dag_folder=str(DAGS))
    assert bag.import_errors == {}
    dag = bag.dags.get("kb_refresh")  # read from the parsed bag; no metadata database needed
    assert dag is not None
    assert set(dag.task_ids) == {"validate_corpus", "build_candidate", "evaluate_candidate", "decide", "promote", "hold_back"}
    assert dag.get_task("decide").downstream_task_ids == {"promote", "hold_back"}
    assert dag.get_task("validate_corpus").downstream_task_ids == {"build_candidate"}
    assert dag.catchup is False and dag.max_active_runs == 1
