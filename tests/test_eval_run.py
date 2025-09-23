import json
import os
import sys
from pathlib import Path
from io import BytesIO

import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def png_bytes(color=(255, 0, 0), size=(8, 8)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


REQUIRED_COLUMNS = [
    "id",
    "year",
    "group",
    "points",
    "problem_number",
    "problem_statement",
    "answer",
    "multimodal",
    "sol_A",
    "sol_B",
    "sol_C",
    "sol_D",
    "sol_E",
    "question_image",
    "sol_A_image_bin",
    "sol_B_image_bin",
    "sol_C_image_bin",
    "sol_D_image_bin",
    "sol_E_image_bin",
    "associated_images_bin",
    "language",
]


def make_row(idx=1, with_images=False, invalid_question=False):
    qimg = None
    assoc = None
    if with_images:
        qimg = b"not-an-image" if invalid_question else png_bytes()
        assoc = [png_bytes((0, 255, 0))]
    row = {
        "id": idx,
        "year": 2024,
        "group": "Junior",
        "points": 5,
        "problem_number": idx,
        "problem_statement": f"Was ist 2+{idx}?",
        "answer": "A",
        "multimodal": bool(with_images),
        "sol_A": "3",
        "sol_B": "4",
        "sol_C": "5",
        "sol_D": "6",
        "sol_E": "7",
        "question_image": qimg,
        "sol_A_image_bin": None,
        "sol_B_image_bin": None,
        "sol_C_image_bin": None,
        "sol_D_image_bin": None,
        "sol_E_image_bin": None,
        "associated_images_bin": assoc,
        "language": "de",
    }
    return row


def write_parquet(tmp_path: Path, rows):
    # Ensure all required columns exist
    df = pd.DataFrame(rows)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[REQUIRED_COLUMNS]
    path = tmp_path / "data.parquet"
    df.to_parquet(path, engine="pyarrow", index=False)
    return path


class FakeResp:
    def __init__(self, status_code=200, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload
        if text is None and payload is not None:
            text = json.dumps(payload)
        self.text = text or ""

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def run_eval(monkeypatch, tmp_path: Path, dataset_path: Path, model_id: str, responses, reasoning="none"):
    # Import module fresh
    import importlib.util
    mod_path = ROOT / "eval_run.py"
    spec = importlib.util.spec_from_file_location("eval_run", mod_path)
    assert spec and spec.loader, "Could not load eval_run module spec"
    eval_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_run)  # type: ignore

    # Env
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    # Patch models registry to allow custom models if needed
    def fake_registry(_path):
        # use provided model id, supports vision for default
        if model_id == "text-only/test":
            return {
                model_id: eval_run.ModelInfo(
                    id=model_id, label="TextOnly", supports_vision=False, supports_json_response_format=True
                )
            }
        return {
            model_id: eval_run.ModelInfo(
                id=model_id, label="GPT5", supports_vision=True, supports_json_response_format=True
            )
        }

    monkeypatch.setattr(eval_run, "read_models_registry", fake_registry)

    # Patch requests.post with queued responses
    calls = {"count": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        i = calls["count"]
        calls["count"] += 1
        resp = responses[i] if i < len(responses) else responses[-1]
        return resp

    import requests as _requests

    monkeypatch.setattr(_requests, "post", fake_post)

    # Prepare args
    out_dir = tmp_path / "runs"
    argv = [
        "eval_run.py",
        "--dataset",
        str(dataset_path),
        "--model",
        model_id,
        "--output_dir",
        str(out_dir),
        "--reasoning",
        reasoning,
        "--max_tokens",
        "64",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    # Run
    eval_run.main()

    # Find run dir
    safe_model = model_id.replace("/", "-")
    run_dirs = sorted(out_dir.glob(f"*_{safe_model}"))
    assert run_dirs, "run directory not created"
    run_dir = run_dirs[-1]
    return run_dir


def test_answer_json_success(monkeypatch, tmp_path):
    dataset = write_parquet(tmp_path, [make_row(1, with_images=False)])

    payload = {
        "id": "gen_1",
        "model": "openai/gpt-5",
        "choices": [{"message": {"content": "{\"answer\":\"A\"}"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.002},
    }
    run_dir = run_eval(
        monkeypatch,
        tmp_path,
        dataset,
        model_id="openai/gpt-5",
        responses=[FakeResp(200, payload)],
    )

    # Validate results
    import pandas as pd

    df = pd.read_parquet(run_dir / "results.parquet", engine="pyarrow")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["predicted"] == "A"
    assert row["is_correct"] == True
    assert row["points_earned"] == 5
    assert row["prompt_tokens"] == 10
    assert row["total_tokens"] == 15
    assert abs(row["cost_usd"] - 0.002) < 1e-9

    json_records = json.loads((run_dir / "results.json").read_text())
    assert len(json_records) == 1
    assert json_records[0]["predicted"] == "A"

    jsonl_lines = (run_dir / "results.jsonl").read_text().strip().splitlines()
    assert len(jsonl_lines) == 1
    assert json.loads(jsonl_lines[0])["predicted"] == "A"

    raw_responses = (run_dir / "raw_responses.jsonl").read_text().strip().splitlines()
    assert len(raw_responses) == 1
    payload = json.loads(raw_responses[0])
    assert payload["id"] == 1
    assert payload["response"]["choices"][0]["message"]["content"] == "{\"answer\":\"A\"}"

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["answered_count"] == 1
    assert metrics["failed_count"] == 0
    assert metrics["accuracy"] == 1.0
    assert metrics["unknown_usage_count"] == 0


def test_cot_with_reason(monkeypatch, tmp_path):
    dataset = write_parquet(tmp_path, [make_row(2)])
    payload = {
        "id": "gen_2",
        "model": "openai/gpt-5",
        "choices": [{"message": {"content": "{\"answer\":\"A\",\"reason\":\"kurz\"}"}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18, "cost": 0.003},
    }
    run_dir = run_eval(
        monkeypatch,
        tmp_path,
        dataset,
        model_id="openai/gpt-5",
        responses=[FakeResp(200, payload)],
        reasoning="cot",
    )
    df = pd.read_parquet(run_dir / "results.parquet", engine="pyarrow")
    row = df.iloc[0]
    assert row["predicted"] == "A"
    assert row["rationale"] == "kurz"


def test_regex_fallback_and_missing_usage(monkeypatch, tmp_path):
    dataset = write_parquet(tmp_path, [make_row(3)])
    payload = {
        "id": "gen_3",
        "model": "openai/gpt-5",
        "choices": [{"message": {"content": "I think the answer is d."}}],
        # usage omitted
    }
    run_dir = run_eval(
        monkeypatch,
        tmp_path,
        dataset,
        model_id="openai/gpt-5",
        responses=[FakeResp(200, payload)],
    )
    df = pd.read_parquet(run_dir / "results.parquet", engine="pyarrow")
    row = df.iloc[0]
    assert row["predicted"] == "D"
    assert pd.isna(row["total_tokens"]) or row["total_tokens"] is None

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["unknown_usage_count"] == 1


def test_list_content_response(monkeypatch, tmp_path):
    dataset = write_parquet(tmp_path, [make_row(7)])
    payload = {
        "id": "gen_list",
        "model": "openai/gpt-5",
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "output_text", "text": "Here is reasoning."},
                        {"type": "json", "json": {"answer": "A"}},
                    ]
                }
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
    }
    run_dir = run_eval(
        monkeypatch,
        tmp_path,
        dataset,
        model_id="openai/gpt-5",
        responses=[FakeResp(200, payload)],
    )
    import pandas as pd

    df = pd.read_parquet(run_dir / "results.parquet", engine="pyarrow")
    row = df.iloc[0]
    assert row["predicted"] == "A"
    assert "json_extracted" in (row["warnings"] or [])

    raw_payload = json.loads((run_dir / "raw_responses.jsonl").read_text().strip())
    assert raw_payload["id"] == 7


def test_http_error_failure(monkeypatch, tmp_path):
    dataset = write_parquet(tmp_path, [make_row(4)])
    run_dir = run_eval(
        monkeypatch,
        tmp_path,
        dataset,
        model_id="openai/gpt-5",
        responses=[FakeResp(500, payload=None, text="Server error")],
    )
    df = pd.read_parquet(run_dir / "results.parquet", engine="pyarrow")
    row = df.iloc[0]
    assert row["error"].startswith("http_500")
    assert pd.isna(row["predicted"]) or row["predicted"] is None

    failures = (run_dir / "failures.jsonl").read_text().strip().splitlines()
    assert failures and json.loads(failures[0])["error"].startswith("http_500")


def test_skip_images_for_text_only_model(monkeypatch, tmp_path):
    dataset = write_parquet(tmp_path, [make_row(5, with_images=True)])
    # Use custom text-only model
    run_dir = run_eval(
        monkeypatch,
        tmp_path,
        dataset,
        model_id="text-only/test",
        responses=[],
    )
    # No results because the only row was skipped
    import pandas as pd

    df = pd.read_parquet(run_dir / "results.parquet", engine="pyarrow")
    assert len(df) == 0
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["answered_count"] == 0
    assert metrics["skipped_count"] == 1


def test_image_decode_warning(monkeypatch, tmp_path):
    dataset = write_parquet(tmp_path, [make_row(6, with_images=True, invalid_question=True)])
    payload = {
        "id": "gen_6",
        "model": "openai/gpt-5",
        "choices": [{"message": {"content": "{\"answer\":\"A\"}"}}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12, "cost": 0.001},
    }
    run_dir = run_eval(
        monkeypatch,
        tmp_path,
        dataset,
        model_id="openai/gpt-5",
        responses=[FakeResp(200, payload)],
    )
    df = pd.read_parquet(run_dir / "results.parquet", engine="pyarrow")
    row = df.iloc[0]
    warnings = row["warnings"]
    # Stored as list in parquet via pandas -> may deserialize as list
    assert warnings and "question_image_decode_failed" in warnings
