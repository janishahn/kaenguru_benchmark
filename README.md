Känguru Benchmark – LLM Evaluation
=================================

Quickstart (uses uv for environment management)

- Requirements: Python 3.11+, `uv` installed, OpenRouter API key.

Setup

- Install deps:
  - `uv sync`
- Set env var:
  - `export OPENROUTER_API_KEY=...`
  - Alternatively, place the key in a local `.env` file (e.g. `OPENROUTER_API_KEY=sk-...`); the script auto-loads it.

Input Data

- Store your Kaguru benchmark `.parquet` file anywhere inside or outside the repo (for example `datasets/kaenguru_2024.parquet`).
- Pass the path via `--dataset`; relative paths are resolved from the current working directory and `~` is supported.
- The script validates that the supplied file exists and has a `.parquet` extension before starting the run.
- Hard-required columns in the dataset (must all be present, values may be `null`/`None`): `id`, `year`, `group`, `points`, `problem_number`, `problem_statement`, `answer`, `multimodal`, `sol_A`, `sol_B`, `sol_C`, `sol_D`, `sol_E`, `question_image`, `sol_A_image_bin`, `sol_B_image_bin`, `sol_C_image_bin`, `sol_D_image_bin`, `sol_E_image_bin`, `associated_images_bin`, `language`.
- Image columns should contain raw bytes (preferred), but base64 strings are tolerated and will be decoded best-effort; undecodable blobs simply trigger warnings.
- `answer` is expected to be a single letter in `{A, B, C, D, E}`; anything else is treated as missing ground truth, disabling accuracy for that row.
- Extra columns are ignored, and numeric/text types are coerced as needed (for example, `points` is cast to `float`).

Run Examples

- Vision (GPT‑5), answer-only:
  - `uv run python eval_run.py --dataset /abs/path/to/dataset.parquet --model openai/gpt-5 --reasoning none --max_tokens 128`
- Vision (GPT‑5), chain-of-thought:
  - `uv run python eval_run.py --dataset /abs/path/to/dataset.parquet --model openai/gpt-5 --reasoning cot --max_tokens 256`
- Omit `--max_tokens` to let the provider choose defaults; the runner only sends a value if you specify one or if a retry needs more budget.
- Add `--sequential` if you need to process requests strictly one at a time.

Artifacts

- Under `runs/{timestamp}_{model}/` the script writes:
  - `results.parquet` – per-row results
  - `results.json` / `results.jsonl` – same results in human-readable form (`results.jsonl` streams each row as soon as it completes)
  - `metrics.json` – aggregates and cost totals
  - `config.json` – run configuration and model info
  - `failures.jsonl` – errors and diagnostics (streamed incrementally)
  - `raw_responses.jsonl` – raw API responses per attempt (useful when a model misbehaves)

Dashboard
---------

- Install/update dependencies: `uv sync`
- Launch the dashboard locally: `uv run python dashboard.py --host 127.0.0.1 --port 8000`
- Open `http://127.0.0.1:8000` in a browser. The UI works entirely offline; all JS/CSS dependencies are vendored.
- Overview page: cards summarize every run (accuracy badges, answered/skipped counts, latency/tokens averages). Failures and unknown-usage are surfaced as warning chips. Use the Compare button to pre-fill selectors on the compare page.
- Run detail: filter sidebar (group, year, language, multimodal, correctness, reasoning mode, value ranges, warnings) drives server-side pagination and charts. Presets are stored in `localStorage`. Charts (group/year breakdowns, confusion matrix, latency/tokens histograms, predicted-letter distribution) update with the active filters. Click a row to open the drawer with dataset content (problem, answer choices, images, rationale text, warnings); a toggle reveals the raw model response when present. The issues panel surfaces top warning types and the failures timeline, with deep links back to affected ids.
- Compare page: pick two runs to view metric deltas, group/year delta bars, and a per-id diff table. The view selector can restrict to changed/improved/regressed rows; optional limit caps the diff table size. Enable the confusion-matrix toggle to preview both runs' confusion matrices side by side.
- Export buttons on the run detail page respect current filters: CSV or JSON downloads via `/api/runs/{id}/results?download={csv|json}`. Filters, sorts, and pagination are all encoded into the request.
- Reload when new run folders appear: `curl -X POST http://127.0.0.1:8000/api/reload` (or use any HTTP client). The index rebuild is fast and re-populates filter facets.
- Known limitations: designed for single-user, local usage (no authentication); very large runs may still take a few seconds to stream filters/aggregates; the vendored chart shim supports only the dashboard’s built-in visualisations.

Notes

- The runner issues OpenRouter requests concurrently by default (worker count is derived from available CPUs); add `--sequential` if you prefer single-request processing.
- A shared adaptive rate limiter keeps all workers within the provider’s limits. You can optionally seed per-model limits in `models.json` (see below); the limiter still backs off automatically when throttled.
- The model registry is in `models.json`. Add entries to extend.
- If a text-only model is added without vision, image questions are skipped.

Example `models.json` entry (all optional fields shown):

```jsonc
{
  "models": [
    {
      "id": "openai/gpt-5",
      "label": "OpenAI GPT-5",
      "supports_vision": true,
      "supports_json_response_format": true,
      "rate_limit": {
        "requests_per_second": 3,
        "requests_per_minute": 150,
        "min_interval_seconds": 0.5
      }
    }
  ]
}
```

- You can supply any subset of the `rate_limit` fields. The evaluator converts the first valid value into an initial inter-request delay and adapts from there based on live responses.
