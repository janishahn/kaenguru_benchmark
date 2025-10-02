import json
from pathlib import Path
import shutil
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

from dashboard.human_baseline import HumanBaselineIndex
from dashboard.server import create_app


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "human_baseline"


def _write_mock_run(path: Path, results: list[dict]) -> None:
    path.mkdir()
    normalized_results: list[dict] = []
    total_points = 0.0
    total_earned = 0.0
    correct = 0
    for idx, item in enumerate(results, start=1):
        base = {
            "id": item.get("id", f"row-{idx}"),
            "year": item.get("year", "2008"),
            "group": item.get("group", "3"),
            "problem_number": idx,
            "language": "de",
            "multimodal": False,
            "answer": "A",
            "predicted": "A" if item.get("is_correct") else "B",
            "is_correct": item.get("is_correct", False),
            "points": float(item.get("points", 0.0)),
            "points_earned": float(item.get("points_earned", 0.0)),
            "reasoning_mode": "default",
            "latency_ms": 100.0,
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "total_tokens": 20,
            "cost_usd": 0.0,
            "rationale": "",
            "raw_text_response": "",
            "generation_id": f"gen-{idx}",
            "error": None,
            "warnings": None,
        }
        base.update(item)
        normalized_results.append(base)
        total_points += base["points"]
        total_earned += base["points_earned"]
        if base.get("is_correct"):
            correct += 1
    metrics = {
        "answered_count": len(results),
        "skipped_count": 0,
        "failed_count": 0,
        "accuracy": (correct / len(results)) if results else 0.0,
        "points_weighted_accuracy": (total_earned / total_points) if total_points else 0.0,
        "total_points_earned": total_earned,
        "breakdown_by_group": {},
        "breakdown_by_year": {},
    }
    config = {
        "timestamp_utc": "2025-09-30T00:00:00Z",
        "args": {"dataset": "dataset_sample_50.parquet", "model": "test-model"},
        "model": {"id": "test-model", "label": "Test Model"},
    }
    (path / "metrics.json").write_text(json.dumps(metrics))
    (path / "config.json").write_text(json.dumps(config))
    (path / "results.json").write_text(json.dumps(normalized_results))


def test_loader_builds_grade_stats():
    index = HumanBaselineIndex(FIXTURE_DIR)

    assert index.available_years() == [2008]
    year = index.get_year(2008)

    grade3 = year.grade("3")
    assert grade3.total_count == 6
    assert pytest.approx(grade3.mean_estimate, rel=1e-3) == 86.5625
    assert grade3.percentile(0.0) == pytest.approx(0.0, abs=1e-6)
    assert grade3.percentile(120.0) == pytest.approx(1.0, abs=1e-6)
    assert grade3.bin_index_for_score(118.0) == 1

    cdf_points = grade3.cdf_points
    assert cdf_points == sorted(cdf_points, key=lambda item: item[0])
    assert cdf_points[-1][1] == pytest.approx(1.0, abs=1e-6)

    # Aggregating two grades should sum counts and recompute stats.
    combined = year.aggregate_grades(["3", "4"], "3-4", "Grades 3-4")
    assert combined is not None
    assert combined.total_count == 12
    assert combined.counts == [1, 5, 6]


def test_missing_directory_is_safe(tmp_path):
    missing_dir = tmp_path / "nope"
    index = HumanBaselineIndex(missing_dir)
    assert index.available_years() == []
    assert not index.has_data()


def test_invalid_counts_raise(tmp_path):
    data_dir = tmp_path / "human_results"
    data_dir.mkdir(parents=True)
    bad_path = data_dir / "human_baseline_1999.json"
    bad_path.write_text(
        """
        {
          "schema_version": "2.0",
          "year": 1999,
          "locale": "DE",
          "grades": [
            {"id": "3-4", "label": "3/4", "members": [3,4], "max_points": 105.0}
          ],
          "bins": [
            {
              "id": "150",
              "label_pdf": "150,00 (105,00)",
              "range_default": {"min": 150.0, "max": 150.0, "inclusive": "both"},
              "ranges_by_grade": {"3-4": {"min": 105.0, "max": 105.0, "inclusive": "both"}}
            }
          ],
          "counts_by_grade": {"3-4": [1]},
          "totals_by_grade": {"3-4": 2},
          "avg_score_by_grade": {"3-4": 90.0}
        }
        """
    )

    with pytest.raises(ValueError):
        HumanBaselineIndex(data_dir)


def test_human_endpoints(tmp_path):
    humans_dir = tmp_path / "humans"
    humans_dir.mkdir()
    fixture_file = FIXTURE_DIR / "human_baseline_2008.json"
    shutil.copy(fixture_file, humans_dir / fixture_file.name)

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    run1 = "20250930_000000-modelA"
    run2 = "20250930_010000-modelB"
    _write_mock_run(
        runs_dir / run1,
        results=[
            {"id": "g3-1", "year": "2008", "group": "3", "points": 40.0, "points_earned": 35.0, "is_correct": True},
            {"id": "g3-2", "year": "2008", "group": "3", "points": 40.0, "points_earned": 40.0, "is_correct": True},
            {"id": "g3-3", "year": "2008", "group": "3", "points": 40.0, "points_earned": 30.0, "is_correct": False},
            {"id": "g4-1", "year": "2008", "group": "4", "points": 40.0, "points_earned": 30.0, "is_correct": False},
            {"id": "g4-2", "year": "2008", "group": "4", "points": 40.0, "points_earned": 25.0, "is_correct": False},
            {"id": "g4-3", "year": "2008", "group": "4", "points": 40.0, "points_earned": 20.0, "is_correct": False}
        ],
    )
    _write_mock_run(
        runs_dir / run2,
        results=[
            {"id": "g3-1", "year": "2008", "group": "3", "points": 40.0, "points_earned": 28.0, "is_correct": False},
            {"id": "g3-2", "year": "2008", "group": "3", "points": 40.0, "points_earned": 32.0, "is_correct": False},
            {"id": "g3-3", "year": "2008", "group": "3", "points": 40.0, "points_earned": 30.0, "is_correct": False},
            {"id": "g4-1", "year": "2008", "group": "4", "points": 40.0, "points_earned": 38.0, "is_correct": True},
            {"id": "g4-2", "year": "2008", "group": "4", "points": 40.0, "points_earned": 35.0, "is_correct": True},
            {"id": "g4-3", "year": "2008", "group": "4", "points": 40.0, "points_earned": 30.0, "is_correct": False}
        ],
    )

    app = create_app(
        runs_dir=runs_dir,
        models_path=Path("models.json"),
        templates_dir=Path("web/templates"),
        static_dir=Path("web/static"),
        human_results_dir=humans_dir,
    )
    client = TestClient(app)

    years = client.get("/api/humans/years").json()
    assert years and years[0]["year"] == 2008

    summary = client.get("/api/humans/2008/summary").json()
    assert summary["totals_by_grade"]["3"] == 6

    cdf = client.get("/api/humans/2008/cdf", params={"grade": "3"}).json()
    assert cdf["grade_id"] == "3"
    assert cdf["points"]

    percentile = client.get(
        "/api/humans/percentile",
        params={"year": 2008, "grade": "3", "score": 115.0},
    ).json()
    assert pytest.approx(percentile["percentile"], rel=1e-2) == 0.629

    run_compare = client.get(f"/api/humans/compare/run/{run1}").json()
    assert run_compare["entries"]
    grade3_entry = next(item for item in run_compare["entries"] if item["grade_id"] == "3")
    assert pytest.approx(grade3_entry["llm_total"], rel=1e-6) == 105.0
    assert grade3_entry["bin_comparison"][2]["llm_share"] == 1.0

    cohort = client.post(
        "/api/humans/compare/aggregate",
        json={"run_ids": [run1, run2]},
    ).json()
    assert cohort["micro"]["entries"]
    micro_grade3 = next(entry for entry in cohort["micro"]["entries"] if entry["grade_id"] == "3")
    assert micro_grade3["sample_count"] == 2
