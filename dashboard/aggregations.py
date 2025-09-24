"""Aggregation utilities for dashboard analytics and comparisons."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Mapping, Sequence

from . import schemas
from .utils import grade_group_sort_key

LETTER_SET = {"A", "B", "C", "D", "E"}


def compute_aggregates(rows: Sequence[schemas.RowRecord]) -> schemas.AggregatesResponse:
    breakdown_group: Dict[str, Dict[str, float]] = defaultdict(lambda: {
        "count": 0,
        "correct": 0,
        "points_total": 0.0,
        "points_earned": 0.0,
    })
    breakdown_year: Dict[str, Dict[str, float]] = defaultdict(lambda: {
        "count": 0,
        "correct": 0,
        "points_total": 0.0,
        "points_earned": 0.0,
    })

    def record_breakdown(target: Dict[str, Dict[str, float]], key: str | None, row: schemas.RowRecord) -> None:
        if not key:
            return
        entry = target[key]
        entry["count"] += 1
        if row.is_correct:
            entry["correct"] += 1
        if row.points is not None:
            entry["points_total"] += float(row.points)
        if row.points_earned is not None:
            entry["points_earned"] += float(row.points_earned)

    for row in rows:
        record_breakdown(breakdown_group, row.group, row)
        record_breakdown(breakdown_year, row.year, row)

    def finalize_breakdown(mapping: Dict[str, Dict[str, float]]) -> Dict[str, schemas.BreakdownEntry]:
        result: Dict[str, schemas.BreakdownEntry] = {}
        for key in sorted(mapping.keys(), key=grade_group_sort_key):
            entry = mapping[key]
            count = int(entry["count"])
            accuracy = (entry["correct"] / count) if count else 0.0
            total_points = entry["points_total"]
            pwa = (entry["points_earned"] / total_points) if total_points else 0.0
            result[key] = schemas.BreakdownEntry(
                count=count,
                accuracy=round(accuracy, 4),
                points_weighted_accuracy=round(pwa, 4),
            )
        return result

    confusion_matrix: Dict[str, Dict[str, int]] = {letter: {p: 0 for p in LETTER_SET} for letter in LETTER_SET}
    predicted_counts: Counter[str] = Counter()
    latency_values: List[float] = []
    token_values: List[float] = []
    reasoning_values: List[float] = []
    warning_counts: Counter[str] = Counter()

    for row in rows:
        answer = row.answer.upper() if row.answer else None
        predicted = row.predicted.upper() if row.predicted else None
        if predicted:
            predicted_counts[predicted] += 1
        if answer in LETTER_SET and predicted in LETTER_SET:
            confusion_matrix[answer][predicted] += 1
        if row.latency_ms is not None:
            latency_values.append(float(row.latency_ms))
        if row.total_tokens is not None:
            token_values.append(float(row.total_tokens))
        if row.reasoning_tokens is not None:
            reasoning_values.append(float(row.reasoning_tokens))
        if row.warnings:
            warning_counts.update(w for w in row.warnings if w)

    latency_hist = build_histogram(latency_values)
    tokens_hist = build_histogram(token_values)
    reasoning_hist = build_histogram(reasoning_values)

    warning_toplist = [
        schemas.WarningBreakdown(warning_type=w, count=c) for w, c in warning_counts.most_common(10)
    ]
    return schemas.AggregatesResponse(
        breakdown_by_group=finalize_breakdown(breakdown_group),
        breakdown_by_year=finalize_breakdown(breakdown_year),
        confusion_matrix=confusion_matrix,
        latency_hist=latency_hist,
        tokens_hist=tokens_hist,
        reasoning_tokens_hist=reasoning_hist,
        predicted_counts=dict(predicted_counts),
        warning_toplist=warning_toplist,
    )


def build_histogram(values: Sequence[float]) -> schemas.Histogram:
    if not values:
        return schemas.Histogram()
    values = sorted(float(v) for v in values)
    min_val = values[0]
    max_val = values[-1]
    if math.isclose(min_val, max_val):
        return schemas.Histogram(bins=[min_val, max_val], counts=[len(values)], min=min_val, max=max_val)
    bucket_count = min(20, max(5, int(math.sqrt(len(values)))))
    step = (max_val - min_val) / bucket_count or 1.0
    bins = [min_val + i * step for i in range(bucket_count + 1)]
    counts = [0 for _ in range(bucket_count)]
    for value in values:
        idx = min(int((value - min_val) / step), bucket_count - 1)
        counts[idx] += 1
    return schemas.Histogram(bins=bins, counts=counts, min=min_val, max=max_val)


def _confusion_for_rows(rows: Mapping[str, schemas.RowRecord] | Sequence[schemas.RowRecord]) -> Dict[str, Dict[str, int]]:
    matrix = {letter: {p: 0 for p in LETTER_SET} for letter in LETTER_SET}
    if isinstance(rows, Mapping):
        iterable = rows.values()
    else:
        iterable = rows
    for row in iterable:
        answer = row.answer.upper() if row.answer else None
        predicted = row.predicted.upper() if row.predicted else None
        if answer in LETTER_SET and predicted in LETTER_SET:
            matrix[answer][predicted] += 1
    return matrix


def compute_compare(
    run_a: schemas.RunSummary,
    run_b: schemas.RunSummary,
    metrics_a: schemas.RunMetrics,
    metrics_b: schemas.RunMetrics,
    rows_a: Mapping[str, schemas.RowRecord],
    rows_b: Mapping[str, schemas.RowRecord],
) -> schemas.CompareResponse:
    metric_keys = [
        "answered_count",
        "skipped_count",
        "failed_count",
        "accuracy",
        "points_weighted_accuracy",
        "mean_latency_ms",
        "median_latency_ms",
        "mean_total_tokens",
        "total_cost_usd_known",
    ]

    metrics: List[schemas.CompareMetricDelta] = []
    for key in metric_keys:
        value_a = getattr(metrics_a, key, None)
        value_b = getattr(metrics_b, key, None)
        delta = None
        if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
            delta = value_b - value_a
        metrics.append(
            schemas.CompareMetricDelta(
                metric=key,
                run_a=float(value_a) if isinstance(value_a, (int, float)) else None,
                run_b=float(value_b) if isinstance(value_b, (int, float)) else None,
                delta=float(delta) if delta is not None else None,
            )
        )

    breakdown_deltas: Dict[str, List[schemas.CompareMetricDelta]] = {}
    mapping_pairs = {
        "group": (metrics_a.breakdown_by_group, metrics_b.breakdown_by_group),
        "year": (metrics_a.breakdown_by_year, metrics_b.breakdown_by_year),
    }
    for name, (map_a, map_b) in mapping_pairs.items():
        sort_key = grade_group_sort_key
        keys = sorted(set(map_a.keys()) | set(map_b.keys()), key=sort_key)
        series: List[schemas.CompareMetricDelta] = []
        for key in keys:
            entry_a = map_a.get(key)
            entry_b = map_b.get(key)
            val_a = entry_a.accuracy if entry_a else None
            val_b = entry_b.accuracy if entry_b else None
            delta = None
            if val_a is not None and val_b is not None:
                delta = val_b - val_a
            series.append(
                schemas.CompareMetricDelta(
                    metric=key,
                    run_a=val_a,
                    run_b=val_b,
                    delta=delta,
                )
            )
        breakdown_deltas[name] = series

    common_ids = sorted(set(rows_a.keys()) & set(rows_b.keys()))
    row_deltas: List[schemas.CompareRowDelta] = []
    for row_id in common_ids:
        row_a = rows_a[row_id]
        row_b = rows_b[row_id]
        delta_points = None
        if row_a.points_earned is not None and row_b.points_earned is not None:
            delta_points = row_b.points_earned - row_a.points_earned
        delta_latency = None
        if row_a.latency_ms is not None and row_b.latency_ms is not None:
            delta_latency = row_b.latency_ms - row_a.latency_ms
        delta_tokens = None
        if row_a.total_tokens is not None and row_b.total_tokens is not None:
            delta_tokens = row_b.total_tokens - row_a.total_tokens
        row_deltas.append(
            schemas.CompareRowDelta(
                id=row_id,
                run_a_correct=row_a.is_correct,
                run_b_correct=row_b.is_correct,
                run_a_predicted=row_a.predicted,
                run_b_predicted=row_b.predicted,
                run_a_points=row_a.points_earned,
                run_b_points=row_b.points_earned,
                delta_points=delta_points,
                delta_latency_ms=delta_latency,
                delta_total_tokens=delta_tokens,
            )
        )

    return schemas.CompareResponse(
        run_a=run_a,
        run_b=run_b,
        metrics=metrics,
        breakdown_deltas=breakdown_deltas,
        row_deltas=row_deltas,
        confusion_matrices={
            "run_a": _confusion_for_rows(rows_a),
            "run_b": _confusion_for_rows(rows_b),
        },
    )


__all__ = ["compute_aggregates", "compute_compare", "build_histogram"]
