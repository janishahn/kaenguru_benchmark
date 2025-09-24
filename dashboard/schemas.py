"""Pydantic schemas shared across the dashboard backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, computed_field


class BreakdownEntry(BaseModel):
    count: int = 0
    accuracy: float = 0.0
    points_weighted_accuracy: float = 0.0


class RunMetrics(BaseModel):
    answered_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    accuracy: float = 0.0
    points_weighted_accuracy: float = 0.0
    mean_latency_ms: Optional[float] = None
    median_latency_ms: Optional[float] = None
    mean_total_tokens: Optional[float] = None
    total_cost_usd_known: Optional[float] = None
    unknown_usage_count: Optional[int] = None
    breakdown_by_group: Dict[str, BreakdownEntry] = Field(default_factory=dict)
    breakdown_by_year: Dict[str, BreakdownEntry] = Field(default_factory=dict)


class RunConfigArgs(BaseModel):
    dataset: Optional[str] = None
    model: Optional[str] = None
    reasoning: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    limit: Optional[int] = None
    seed: Optional[int] = None
    concurrency: Optional[int] = None
    output_dir: Optional[str] = None
    fail_fast: Optional[bool] = None


class RunConfigModel(BaseModel):
    id: Optional[str] = None
    label: Optional[str] = None
    supports_vision: Optional[bool] = None
    supports_json_response_format: Optional[bool] = None


class RunConfig(BaseModel):
    timestamp_utc: Optional[str] = None
    args: RunConfigArgs = Field(default_factory=RunConfigArgs)
    model: RunConfigModel = Field(default_factory=RunConfigModel)


class RowRecord(BaseModel):
    id: str
    year: Optional[str]
    group: Optional[str]
    problem_number: Optional[str | int]
    language: Optional[str]
    multimodal: Optional[bool]
    points: Optional[float]
    answer: Optional[str]
    predicted: Optional[str]
    is_correct: Optional[bool]
    points_earned: Optional[float]
    reasoning_mode: Optional[str]
    latency_ms: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    cost_usd: Optional[float]
    rationale: Optional[str]
    raw_text_response: Optional[str]
    generation_id: Optional[str]
    error: Optional[str]
    warnings: Optional[List[str]] = None


class RunPaths(BaseModel):
    results_json: Optional[str] = None
    results_jsonl: Optional[str] = None
    results_parquet: Optional[str] = None
    metrics_json: str
    config_json: str
    raw_responses_jsonl: Optional[str] = None
    failures_jsonl: Optional[str] = None


class RunSummary(BaseModel):
    run_id: str
    timestamp: Optional[str]
    model_id: Optional[str]
    model_label: Optional[str]
    dataset_name: Optional[str]
    metrics: RunMetrics
    reasoning_mode: Optional[str] = None
    has_failures: bool = False
    results_source: Optional[str] = None


class RunDetail(RunSummary):
    config: RunConfig
    paths: RunPaths


@dataclass
class RunOverviewFilterParams:
    model: List[str] = field(default_factory=list)
    dataset: List[str] = field(default_factory=list)
    reasoning_mode: List[str] = field(default_factory=list)
    results_source: List[str] = field(default_factory=list)
    has_failures: Optional[bool] = None
    q: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    sort_by: str = "timestamp"
    sort_dir: str = "desc"

    def model_dump(self) -> Dict[str, object]:
        """Dictionary representation for templating convenience."""
        return {
            "model": list(self.model),
            "dataset": list(self.dataset),
            "reasoning_mode": list(self.reasoning_mode),
            "results_source": list(self.results_source),
            "has_failures": self.has_failures,
            "q": self.q,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "sort_by": self.sort_by,
            "sort_dir": self.sort_dir,
        }


class FacetOption(BaseModel):
    value: str
    label: str
    total: int = 0
    active: int = 0


class RunOverviewFacets(BaseModel):
    models: List[FacetOption] = Field(default_factory=list)
    datasets: List[FacetOption] = Field(default_factory=list)
    reasoning_modes: List[FacetOption] = Field(default_factory=list)
    results_sources: List[FacetOption] = Field(default_factory=list)
    has_failures: List[FacetOption] = Field(default_factory=list)


class RunListResponse(BaseModel):
    runs: List[RunSummary] = Field(default_factory=list)
    total: int = 0
    total_all: int = 0
    facets: RunOverviewFacets = Field(default_factory=RunOverviewFacets)


class WarningBreakdown(BaseModel):
    warning_type: str
    count: int


class FailureEntry(BaseModel):
    timestamp: Optional[str] = None
    status_code: Optional[int] = None
    message: Optional[str] = None
    id: Optional[str] = None


class DatasetChoice(BaseModel):
    label: Optional[str] = None
    text: Optional[str] = None
    image: Optional[str] = None


class DatasetRow(BaseModel):
    id: str
    problem_statement: Optional[str] = None
    options: List[DatasetChoice] = Field(default_factory=list)
    question_image: Optional[str] = None
    associated_images: List[str] = Field(default_factory=list)
    language: Optional[str] = None
    year: Optional[str] = None
    group: Optional[str] = None
    points: Optional[float] = None


class RowDetailResponse(BaseModel):
    row: RowRecord
    dataset: Optional[DatasetRow] = None


class ResultFilterParams(BaseModel):
    group: List[str] = Field(default_factory=list)
    year: List[str] = Field(default_factory=list)
    language: List[str] = Field(default_factory=list)
    multimodal: Optional[bool] = None
    correctness: Optional[str] = Field(default=None, pattern="^(true|false|unknown)$")
    predicted: List[str] = Field(default_factory=list)
    reasoning_mode: List[str] = Field(default_factory=list)
    points_min: Optional[float] = None
    points_max: Optional[float] = None
    latency_min: Optional[float] = None
    latency_max: Optional[float] = None
    tokens_min: Optional[int] = None
    tokens_max: Optional[int] = None
    cost_min: Optional[float] = None
    cost_max: Optional[float] = None
    warnings_present: Optional[bool] = None
    warning_types: List[str] = Field(default_factory=list)
    predicted_letter: List[str] = Field(default_factory=list)
    sort_by: str = Field(default="id")
    sort_dir: str = Field(default="asc")
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=25, ge=1, le=200)

    model_config = {
        "validate_assignment": True,
        "extra": "ignore",
    }

    @computed_field  # type: ignore[misc]
    def cache_key(self) -> tuple:
        return (
            tuple(sorted(self.group)),
            tuple(sorted(self.year)),
            tuple(sorted(self.language)),
            self.multimodal,
            self.correctness,
            tuple(sorted(self.predicted or self.predicted_letter)),
            tuple(sorted(self.reasoning_mode)),
            self.points_min,
            self.points_max,
            self.latency_min,
            self.latency_max,
            self.tokens_min,
            self.tokens_max,
            self.cost_min,
            self.cost_max,
            self.warnings_present,
            tuple(sorted(self.warning_types)),
        )

    def normalized_predicted(self) -> List[str]:
        if self.predicted:
            return [p.upper() for p in self.predicted]
        return [p.upper() for p in self.predicted_letter]


class ResultsPage(BaseModel):
    items: List[RowRecord]
    total: int
    page: int
    page_size: int


class Histogram(BaseModel):
    bins: List[float] = Field(default_factory=list)
    counts: List[int] = Field(default_factory=list)
    min: Optional[float] = None
    max: Optional[float] = None


class AggregatesResponse(BaseModel):
    breakdown_by_group: Dict[str, BreakdownEntry] = Field(default_factory=dict)
    breakdown_by_year: Dict[str, BreakdownEntry] = Field(default_factory=dict)
    confusion_matrix: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    latency_hist: Histogram = Field(default_factory=Histogram)
    tokens_hist: Histogram = Field(default_factory=Histogram)
    predicted_counts: Dict[str, int] = Field(default_factory=dict)
    warning_toplist: List[WarningBreakdown] = Field(default_factory=list)


class FilterFacets(BaseModel):
    groups: List[str] = Field(default_factory=list)
    years: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    reasoning_modes: List[str] = Field(default_factory=list)
    predicted_letters: List[str] = Field(default_factory=list)
    warning_types: List[str] = Field(default_factory=list)


class CompareMetricDelta(BaseModel):
    metric: str
    run_a: Optional[float] = None
    run_b: Optional[float] = None
    delta: Optional[float] = None


class CompareRowDelta(BaseModel):
    id: str
    run_a_correct: Optional[bool]
    run_b_correct: Optional[bool]
    run_a_predicted: Optional[str]
    run_b_predicted: Optional[str]
    run_a_points: Optional[float]
    run_b_points: Optional[float]
    delta_points: Optional[float]
    delta_latency_ms: Optional[float]
    delta_total_tokens: Optional[float]


class CompareResponse(BaseModel):
    run_a: RunSummary
    run_b: RunSummary
    metrics: List[CompareMetricDelta]
    breakdown_deltas: Dict[str, List[CompareMetricDelta]]
    row_deltas: List[CompareRowDelta]
    confusion_matrices: Dict[str, Dict[str, Dict[str, int]]] = Field(default_factory=dict)


class ExportResponse(BaseModel):
    filename: str
    content_type: str
    content: bytes
