"""FastAPI application wiring for the dashboard."""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, ORJSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import humanize

from . import schemas
from .data_index import RunIndex, RunNotFoundError
from .dataset_access import DatasetAccessor

DEFAULT_RUNS_DIR = Path("runs")
DEFAULT_MODELS_PATH = Path("models.json")
DEFAULT_TEMPLATE_DIR = Path("web/templates")
DEFAULT_STATIC_DIR = Path("web/static")

LIST_FIELDS = {
    "group",
    "year",
    "language",
    "predicted",
    "predicted_letter",
    "reasoning_mode",
    "warning_types",
}
FLOAT_FIELDS = {
    "points_min",
    "points_max",
    "latency_min",
    "latency_max",
    "cost_min",
    "cost_max",
}
INT_FIELDS = {
    "tokens_min",
    "tokens_max",
    "reasoning_tokens_min",
    "reasoning_tokens_max",
    "page",
    "page_size",
}
BOOL_FIELDS = {"multimodal", "warnings_present"}


def create_app(
    runs_dir: Path | str = DEFAULT_RUNS_DIR,
    models_path: Path | str = DEFAULT_MODELS_PATH,
    templates_dir: Path | str = DEFAULT_TEMPLATE_DIR,
    static_dir: Path | str = DEFAULT_STATIC_DIR,
) -> FastAPI:
    app = FastAPI(default_response_class=ORJSONResponse)
    runs_path = Path(runs_dir)
    models_path = Path(models_path)
    templates_path = Path(templates_dir)
    static_path = Path(static_dir)

    index = RunIndex(runs_path, models_path)
    dataset_accessor = DatasetAccessor()
    templates = Jinja2Templates(directory=str(templates_path))
    templates.env.filters.setdefault("intcomma", humanize.intcomma)

    app.state.index = index
    app.state.dataset_accessor = dataset_accessor
    app.state.templates = templates

    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    def _parse_datetime(value: str) -> Optional[datetime]:
        raw = value.strip()
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d.%m.%Y"):
                try:
                    dt = datetime.strptime(raw, fmt)
                    break
                except ValueError:
                    continue
            else:
                return None
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt

    def parse_run_overview_filters(request: Request) -> schemas.RunOverviewFilterParams:
        filters = schemas.RunOverviewFilterParams()
        multi = request.query_params.multi_items()
        for key, value in multi:
            if value == "":
                continue
            if key in {"model", "dataset", "reasoning_mode", "results_source"}:
                getattr(filters, key).append(value)
            elif key == "has_failures":
                lowered = value.lower()
                if lowered == "any":
                    filters.has_failures = None
                elif lowered in {"true", "1", "yes"}:
                    filters.has_failures = True
                elif lowered in {"false", "0", "no"}:
                    filters.has_failures = False
            elif key == "q":
                filters.q = value
            elif key == "date_from":
                parsed = _parse_datetime(value)
                if parsed:
                    filters.date_from = parsed
            elif key == "date_to":
                parsed = _parse_datetime(value)
                if parsed:
                    filters.date_to = parsed
            elif key == "sort_by":
                if value in {"timestamp", "accuracy", "answered", "failed"}:
                    filters.sort_by = value
            elif key == "sort_dir":
                lowered = value.lower()
                if lowered in {"asc", "desc"}:
                    filters.sort_dir = lowered
        return filters

    @app.on_event("startup")
    async def _startup() -> None:
        index.reload()

    @app.get("/", response_class=HTMLResponse)
    async def overview(request: Request) -> HTMLResponse:
        filters = parse_run_overview_filters(request)
        runs = index.list_runs(filters)
        facets = index.get_run_overview_facets(filters)
        total_runs = index.total_runs()
        context = {
            "request": request,
            "runs": runs,
            "filters": filters,
            "filters_data": filters.model_dump(),
            "facets": facets,
            "run_count": len(runs),
            "total_runs": total_runs,
        }
        template_name = "partials/overview_section.html" if request.headers.get("HX-Request") else "overview.html"
        return templates.TemplateResponse(template_name, context)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    async def run_detail(request: Request, run_id: str) -> HTMLResponse:
        try:
            detail = index.get_run_detail(run_id)
        except (RunNotFoundError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return templates.TemplateResponse(
            "run_detail.html",
            {"request": request, "run": detail},
        )

    @app.get("/compare", response_class=HTMLResponse)
    async def compare(request: Request, run_a: Optional[str] = None, run_b: Optional[str] = None) -> HTMLResponse:
        runs = index.list_runs()
        context = {"request": request, "runs": runs, "run_a": run_a, "run_b": run_b}
        if run_a and run_b:
            try:
                compare_payload = index.compare_runs(run_a, run_b)
            except RunNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc))
            context["compare"] = compare_payload
        return templates.TemplateResponse("compare.html", context)

    # ------------------------------------------------------------------
    # API Endpoints
    # ------------------------------------------------------------------

    @app.get("/api/runs", response_model=schemas.RunListResponse)
    async def api_list_runs(request: Request) -> schemas.RunListResponse:
        filters = parse_run_overview_filters(request)
        runs = index.list_runs(filters)
        facets = index.get_run_overview_facets(filters)
        return schemas.RunListResponse(
            runs=runs,
            total=len(runs),
            total_all=index.total_runs(),
            facets=facets,
        )

    @app.get("/api/runs/{run_id}", response_model=schemas.RunDetail)
    async def api_run_detail(run_id: str) -> schemas.RunDetail:
        try:
            return index.get_run_detail(run_id)
        except (RunNotFoundError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    def parse_filters(request: Request) -> schemas.ResultFilterParams:
        data: Dict[str, object] = {}
        multi = request.query_params.multi_items()
        for key, value in multi:
            if key == "download":
                continue
            if value == "":
                continue
            if key in LIST_FIELDS:
                data.setdefault(key, [])
                assert isinstance(data[key], list)
                data[key].append(value)
            elif key in FLOAT_FIELDS:
                try:
                    data[key] = float(value)
                except ValueError as exc:  # pragma: no cover - FastAPI converts to 422
                    raise HTTPException(status_code=400, detail=f"Invalid float for {key}") from exc
            elif key in INT_FIELDS:
                try:
                    data[key] = int(value)
                except ValueError as exc:  # pragma: no cover - FastAPI converts to 422
                    raise HTTPException(status_code=400, detail=f"Invalid integer for {key}") from exc
            elif key in BOOL_FIELDS:
                data[key] = value.lower() in {"1", "true", "yes"}
            else:
                data[key] = value
        return schemas.ResultFilterParams(**data)

    def filtered_rows(run_id: str, filters: schemas.ResultFilterParams) -> List[schemas.RowRecord]:
        return [
            row
            for row in index.iter_results(run_id)
            if index.row_matches_filters(row, filters)
        ]

    @app.get("/api/runs/{run_id}/results", response_model=schemas.ResultsPage)
    async def api_run_results(request: Request, run_id: str, download: Optional[str] = Query(default=None)):
        filters = parse_filters(request)
        try:
            if download:
                rows = filtered_rows(run_id, filters)
                if download == "json":
                    payload = [row.model_dump() for row in rows]
                    content = ORJSONResponse(content=payload)
                    filename = f"{run_id}_results.json"
                    content.headers["Content-Disposition"] = f"attachment; filename={filename}"
                    return content
                if download == "csv":
                    buffer = io.StringIO()
                    fieldnames = list(schemas.RowRecord.model_fields.keys())
                    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row.model_dump())
                    csv_bytes = buffer.getvalue().encode("utf-8")
                    headers = {"Content-Disposition": f"attachment; filename={run_id}_results.csv"}
                    return Response(content=csv_bytes, media_type="text/csv", headers=headers)
                raise HTTPException(status_code=400, detail="Unsupported download format")
            page = index.load_results_page(run_id, filters)
            return page
        except (RunNotFoundError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/runs/{run_id}/aggregates", response_model=schemas.AggregatesResponse)
    async def api_run_aggregates(request: Request, run_id: str):
        filters = parse_filters(request)
        try:
            return index.get_aggregates(run_id, filters)
        except (RunNotFoundError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/runs/{run_id}/facets", response_model=schemas.FilterFacets)
    async def api_run_facets(run_id: str):
        try:
            return index.get_facets(run_id)
        except (RunNotFoundError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/runs/{run_id}/row/{row_id}", response_model=schemas.RowDetailResponse)
    async def api_row_detail(run_id: str, row_id: str):
        try:
            record = index.get_run(run_id)
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        row = next((r for r in index.iter_results(run_id) if r.id == row_id), None)
        if not row:
            raise HTTPException(status_code=404, detail="Row not found")
        dataset = dataset_accessor.get_row(record.config.args.dataset, row_id)
        return schemas.RowDetailResponse(row=row, dataset=dataset)

    @app.get("/api/runs/{run_id}/failures", response_model=List[schemas.FailureEntry])
    async def api_run_failures(run_id: str) -> List[schemas.FailureEntry]:
        try:
            return index.load_failures(run_id)
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/compare", response_model=schemas.CompareResponse)
    async def api_compare(
        run_a: str = Query(..., description="Baseline run id"),
        run_b: str = Query(..., description="Run id to compare"),
        view: str = Query("all", description="Filter row deltas: all|changed|improved|regressed"),
        limit: Optional[int] = Query(None, ge=1, le=2000),
        include_confusion: bool = Query(False, description="Include confusion matrices for both runs"),
    ):
        try:
            compare_payload = index.compare_runs(run_a, run_b)
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

        if view != "all":
            row_deltas = filter_row_deltas(compare_payload.row_deltas, view)
        else:
            row_deltas = compare_payload.row_deltas
        if limit is not None:
            row_deltas = row_deltas[:limit]
        compare_payload.row_deltas = row_deltas
        if not include_confusion:
            compare_payload.confusion_matrices = {}
        return compare_payload

    def filter_row_deltas(rows: List[schemas.CompareRowDelta], view: str) -> List[schemas.CompareRowDelta]:
        if view == "changed":
            return [r for r in rows if r.run_a_correct != r.run_b_correct or r.run_a_predicted != r.run_b_predicted]
        if view == "improved":
            return [
                r
                for r in rows
                if (r.run_a_correct is False and r.run_b_correct is True)
                or (r.delta_points is not None and r.delta_points > 0)
            ]
        if view == "regressed":
            return [
                r
                for r in rows
                if (r.run_a_correct is True and r.run_b_correct is False)
                or (r.delta_points is not None and r.delta_points < 0)
            ]
        return rows

    @app.post("/api/reload")
    async def api_reload() -> Dict[str, str]:
        index.reload()
        return {"status": "reloaded", "run_count": str(len(index.list_runs()))}

    return app


__all__ = ["create_app"]
