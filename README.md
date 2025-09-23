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

Artifacts

- Under `runs/{timestamp}_{model}/` the script writes:
  - `results.parquet` – per-row results
  - `results.json` / `results.jsonl` – same results in human-readable form (`results.jsonl` streams each row as soon as it completes)
  - `metrics.json` – aggregates and cost totals
  - `config.json` – run configuration and model info
  - `failures.jsonl` – errors and diagnostics (streamed incrementally)
  - `raw_responses.jsonl` – raw API responses per attempt (useful when a model misbehaves)

Notes

- The model registry is in `models.json`. Add entries to extend.
- If a text-only model is added without vision, image questions are skipped.
