import argparse
import base64
import dataclasses
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


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


LETTER_SET = {"A", "B", "C", "D", "E"}
DEFAULT_RETRY_MAX_TOKENS = 256


@dataclass
class ModelInfo:
    id: str
    label: Optional[str] = None
    supports_vision: bool = False
    supports_json_response_format: bool = False


@dataclass
class RowRecord:
    id: Any
    year: Any
    group: Any
    problem_number: Any
    language: Any
    multimodal: Any
    points: Any
    answer: Optional[str]
    predicted: Optional[str]
    is_correct: Optional[bool]
    points_earned: Optional[float]
    reasoning_mode: str
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


def read_models_registry(path: str) -> Dict[str, ModelInfo]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    models = {}
    for m in data.get("models", []):
        info = ModelInfo(
            id=m.get("id"),
            label=m.get("label"),
            supports_vision=bool(m.get("supports_vision", False)),
            supports_json_response_format=bool(m.get("supports_json_response_format", False)),
        )
        models[info.id] = info
    return models


def validate_dataset_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")


def coerce_bytes(x: Any) -> Optional[bytes]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, (bytes, bytearray, memoryview)):
        return bytes(x)
    # Sometimes parquet may roundtrip as Python "Binary" type; attempt fallback
    if isinstance(x, str):
        # Try base64 decode if it looks like base64; otherwise treat as no-bytes
        try:
            # Heuristic: ignore tiny strings
            if len(x) > 16:
                return base64.b64decode(x, validate=False)
        except Exception:
            return None
    return None


def coerce_list_of_bytes(x: Any) -> Optional[List[bytes]]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, (list, tuple)):
        out: List[bytes] = []
        for item in x:
            b = coerce_bytes(item)
            if b is not None:
                out.append(b)
        return out if out else None
    # Sometimes arrow returns numpy arrays or other sequences
    try:
        from collections.abc import Sequence

        if isinstance(x, Sequence) and not isinstance(x, (bytes, bytearray, str)):
            out = []
            for item in x:
                b = coerce_bytes(item)
                if b is not None:
                    out.append(b)
            return out if out else None
    except Exception:
        pass
    return None


def pil_from_bytes(img_bytes: bytes) -> Optional[Image.Image]:
    try:
        return Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return None


def image_to_data_url(img: Image.Image, prefer_format: Optional[str] = None) -> Tuple[str, str]:
    # Resize if needed
    max_dim = 1024
    if max(img.size) > max_dim:
        img = img.copy()
        img.thumbnail((max_dim, max_dim))

    # Choose encoding format
    fmt = (prefer_format or "PNG").upper()
    if fmt not in {"PNG", "JPEG", "JPG"}:
        fmt = "PNG"
    if fmt == "JPG":
        fmt = "JPEG"
    mime = "image/png" if fmt == "PNG" else "image/jpeg"

    buf = BytesIO()
    save_kwargs = {"format": fmt}
    if fmt == "JPEG":
        save_kwargs.update({"quality": 90})
    img.save(buf, **save_kwargs)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}", fmt


def build_messages(
    row: Dict[str, Any],
    reasoning: str,
    model: ModelInfo,
    encoded_images: Dict[str, Optional[str]],
) -> List[Dict[str, Any]]:
    language = (row.get("language") or "de").lower()
    is_de = language == "de"

    if reasoning == "cot":
        sys_text = (
            "Du bist ein hilfreicher Assistent für Multiple-Choice-Aufgaben."
            if is_de
            else "You are a helpful assistant for multiple-choice tasks."
        )
        instr = (
            'Denke Schritt für Schritt. Gib am Ende ausschließlich ein einzelnes JSON-Objekt aus: {"answer":"A|B|C|D|E","reason":"kurze Begründung"}. Kein weiterer Text.'
            if is_de
            else 'Think step by step. At the end output only a single JSON object: {"answer":"A|B|C|D|E","reason":"short justification"}. No other text.'
        )
    else:
        sys_text = (
            "Du bist ein hilfreicher Assistent für Multiple-Choice-Aufgaben."
            if is_de
            else "You are a helpful assistant for multiple-choice tasks."
        )
        instr = (
            'Gib ausschließlich ein einzelnes JSON-Objekt im Format {"answer":"A|B|C|D|E"} aus. Keine Erklärungen und kein weiterer Text. Wähle nur aus: [A, B, C, D, E].'
            if is_de
            else 'Output only a single JSON object in the format {"answer":"A|B|C|D|E"}. No explanations and no additional text. Choose only from: [A, B, C, D, E].'
        )

    content_parts: List[Dict[str, Any]] = []

    # Question text
    q_label = "Frage:" if is_de else "Question:"
    content_parts.append({"type": "text", "text": f"{q_label} {row['problem_statement']}"})

    # Question image
    if encoded_images.get("question"):
        q_img_label = "Fragebild:" if is_de else "Question image:"
        content_parts.append({"type": "text", "text": q_img_label})
        content_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": encoded_images["question"], "detail": "auto"},
            }
        )

    # Associated images
    assoc = encoded_images.get("assoc_list") or []
    for i, url in enumerate(assoc, start=1):
        if url:
            a_label = ("Zusatzbild" if is_de else "Additional image") + f" {i}:"
            content_parts.append({"type": "text", "text": a_label})
            content_parts.append(
                {"type": "image_url", "image_url": {"url": url, "detail": "auto"}}
            )

    # Options A..E
    choice_hdr = "Antwortmöglichkeiten:" if is_de else "Answer choices:"
    content_parts.append({"type": "text", "text": choice_hdr})
    for letter in ["A", "B", "C", "D", "E"]:
        opt_text = row.get(f"sol_{letter}") or ""
        content_parts.append({"type": "text", "text": f"{letter}) {opt_text}"})
        url = encoded_images.get(f"opt_{letter}")
        if url:
            lbl = ("Option" if not is_de else "Option") + f" {letter} Bild:"
            content_parts.append({"type": "text", "text": lbl})
            content_parts.append(
                {"type": "image_url", "image_url": {"url": url, "detail": "auto"}}
            )

    # Instruction last
    content_parts.append({"type": "text", "text": instr})

    messages = [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": content_parts},
    ]
    return messages


def ensure_output_dir(base_dir: str, model_id: str) -> Tuple[str, str]:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_model = model_id.replace("/", "-")
    run_dir = os.path.join(base_dir, f"{ts}_{safe_model}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, ts


def normalize_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                json_payload = item.get("json")
                if json_payload is not None:
                    try:
                        parts.append(json.dumps(json_payload, ensure_ascii=False))
                    except Exception:
                        parts.append(str(json_payload))
                    continue

                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    continue

                nested = item.get("content")
                if isinstance(nested, (str, list)):
                    nested_text = normalize_message_content(nested)
                    if nested_text:
                        parts.append(nested_text)
                    continue

                # Fall back to serialising remaining primitive entries for debugging
                for key in ("tool_calls", "arguments"):
                    value = item.get(key)
                    if value is not None:
                        try:
                            parts.append(json.dumps(value, ensure_ascii=False))
                        except Exception:
                            parts.append(str(value))
                        break
        return "\n".join([p for p in parts if p])
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        json_payload = content.get("json")
        if json_payload is not None:
            try:
                return json.dumps(json_payload, ensure_ascii=False)
            except Exception:
                return str(json_payload)
        nested = content.get("content")
        if isinstance(nested, (str, list)):
            return normalize_message_content(nested)
    return str(content)


def build_response_format(reasoning: str) -> Dict[str, Any]:
    base_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "pattern": "^[A-E]$",
                "description": "One of A, B, C, D, E",
            },
        },
        "required": ["answer"],
        "additionalProperties": False,
    }
    if reasoning == "cot":
        base_schema["properties"]["reason"] = {
            "type": "string",
            "description": "Short justification",
        }
        base_schema["required"] = ["answer", "reason"]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "multiple_choice_answer",
            "schema": base_schema,
        },
    }


def write_jsonl_line(handle, obj: Dict[str, Any]) -> None:
    handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
    handle.flush()


def load_env_file(path: str = ".env") -> None:
    if not path:
        return
    if not os.path.exists(path) or not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key:
                    continue
                if (value.startswith("\"") and value.endswith("\"")) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]
                os.environ.setdefault(key, value)
    except Exception:
        pass


def resolve_dataset_path(raw_path: str) -> str:
    expanded = os.path.expanduser(raw_path)
    if not os.path.isabs(expanded):
        expanded = os.path.abspath(expanded)
    if not os.path.exists(expanded):
        raise FileNotFoundError(f"Dataset file not found: {expanded}")
    if not os.path.isfile(expanded):
        raise ValueError(f"Dataset path is not a file: {expanded}")
    if not expanded.lower().endswith(".parquet"):
        raise ValueError("Dataset must be a .parquet file")
    return expanded


def parse_answer_from_text(text: str, reasoning: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    # returns (answer, rationale, parse_warning)
    if not text:
        return None, None, "empty_response"
    # Try direct JSON parse
    def try_json(s: str) -> Tuple[Optional[str], Optional[str]]:
        try:
            data = json.loads(s)
            if isinstance(data, dict):
                ans = data.get("answer")
                if isinstance(ans, str):
                    ans = ans.strip().upper()
                    rat = None
                    if reasoning == "cot":
                        r = data.get("reason")
                        if isinstance(r, str):
                            rat = r
                    if ans in LETTER_SET:
                        return ans, rat
        except Exception:
            pass
        return None, None

    ans, rat = try_json(text)
    if ans:
        return ans, rat, None

    # Try to extract first JSON object in text
    match = re.search(r"\{[\s\S]*?\}", text)
    if match:
        ans2, rat2 = try_json(match.group(0))
        if ans2:
            return ans2, rat2, "json_extracted"

    # Fallback regex: first standalone A–E
    match2 = re.search(r"\b([A-Ea-e])\b", text)
    if match2:
        return match2.group(1).upper(), None if reasoning != "cot" else None, "regex_fallback"

    return None, None, "no_parse"


def request_with_retries(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Tuple[requests.Response, float]:
    delays = [0.5, 1.0, 2.0]
    last_exc: Optional[Exception] = None
    start = time.perf_counter()
    for attempt in range(len(delays) + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt < len(delays):
                    time.sleep(delays[attempt])
                    continue
            latency_ms = (time.perf_counter() - start) * 1000.0
            return resp, latency_ms
        except Exception as e:
            last_exc = e
            if attempt < len(delays):
                time.sleep(delays[attempt])
                continue
            raise
    # Should not reach here
    raise last_exc if last_exc else RuntimeError("request failed")


def main():
    parser = argparse.ArgumentParser(description="LLM evaluation runner for Känguru benchmark")
    parser.add_argument("--dataset", required=True, help="Path to dataset .parquet")
    parser.add_argument("--model", required=True, help="Model ID from models.json")
    parser.add_argument("--reasoning", choices=["none", "cot"], default="none")
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output_dir", default="runs")
    parser.add_argument("--fail_fast", action="store_true")
    args = parser.parse_args()

    load_env_file()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable is required.", file=sys.stderr)
        sys.exit(2)

    # Load model info
    models = read_models_registry(os.path.join(os.path.dirname(__file__), "models.json"))
    if args.model not in models:
        print(f"ERROR: model '{args.model}' not found in models.json", file=sys.stderr)
        sys.exit(2)
    model_info = models[args.model]

    # Resolve dataset and outputs
    try:
        dataset_path = resolve_dataset_path(args.dataset)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    run_dir, ts = ensure_output_dir(args.output_dir, args.model)

    # Load dataset
    df = pd.read_parquet(dataset_path, engine="pyarrow")
    validate_dataset_columns(df)

    # Limit rows
    if args.limit is not None and args.limit < len(df):
        if args.seed is not None:
            df = df.sample(n=args.limit, random_state=args.seed)
        else:
            df = df.head(args.limit)

    rows: List[RowRecord] = []
    results_records: List[Dict[str, Any]] = []
    skipped = 0

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "kaenguru-benchmark-eval",
    }
    url = "https://openrouter.ai/api/v1/chat/completions"

    # Iterate rows
    iterable = df.to_dict(orient="records")
    pbar = tqdm(iterable, desc="Evaluating", unit="q")

    results_jsonl_path = os.path.join(run_dir, "results.jsonl")
    raw_responses_path = os.path.join(run_dir, "raw_responses.jsonl")
    failures_jsonl_path = os.path.join(run_dir, "failures.jsonl")

    with open(results_jsonl_path, "w", encoding="utf-8") as results_jsonl, \
        open(raw_responses_path, "w", encoding="utf-8") as raw_responses_file, \
        open(failures_jsonl_path, "w", encoding="utf-8") as failures_file:

        for row in pbar:
            warnings: List[str] = []
            # Collect images
            q_bytes = coerce_bytes(row.get("question_image"))
            opt_bytes = {letter: coerce_bytes(row.get(f"sol_{letter}_image_bin")) for letter in LETTER_SET}
            assoc_bytes = coerce_list_of_bytes(row.get("associated_images_bin")) or []

            has_images = bool(q_bytes or any(opt_bytes.values()) or assoc_bytes)
            if has_images and not model_info.supports_vision:
                skipped += 1
                continue

            # Encode images (vision models only)
            encoded_images: Dict[str, Any] = {"assoc_list": []}
            if model_info.supports_vision:
                # Question image
                if q_bytes:
                    img = pil_from_bytes(q_bytes)
                    if img is None:
                        warnings.append("question_image_decode_failed")
                    else:
                        url_data, _ = image_to_data_url(img)
                        encoded_images["question"] = url_data

                # Option images
                for letter in ["A", "B", "C", "D", "E"]:
                    b = opt_bytes.get(letter)
                    if b:
                        img = pil_from_bytes(b)
                        if img is None:
                            warnings.append(f"opt_{letter}_image_decode_failed")
                        else:
                            url_data, _ = image_to_data_url(img)
                            encoded_images[f"opt_{letter}"] = url_data

                # Associated images
                for i, b in enumerate(assoc_bytes):
                    img = pil_from_bytes(b)
                    if img is None:
                        warnings.append(f"assoc_{i+1}_image_decode_failed")
                        encoded_images["assoc_list"].append(None)
                    else:
                        url_data, _ = image_to_data_url(img)
                        encoded_images["assoc_list"].append(url_data)

            # Build messages
            messages = build_messages(row, args.reasoning, model_info, encoded_images)

            # Payload
            payload: Dict[str, Any] = {
                "model": model_info.id,
                "messages": messages,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "usage": {"include": True},
            }
            if model_info.supports_json_response_format:
                payload["response_format"] = build_response_format(args.reasoning)
            max_tokens_current = args.max_tokens
            attempt = 0
            max_attempts = 5
            combined_usage: Dict[str, float] = {}
            last_usage: Optional[Dict[str, Any]] = None
            latencies: List[float] = []
            content_text = ""
            gen_id = None
            model_echo = None
            ans = None
            rat = None
            parse_warn: Optional[str] = None
            finish_reason = None
            native_finish = None
            request_failed = False

            while attempt < max_attempts:
                attempt += 1
                if max_tokens_current is not None:
                    payload["max_tokens"] = max_tokens_current
                else:
                    payload.pop("max_tokens", None)

                try:
                    resp, latency_ms = request_with_retries(url, headers, payload)
                except Exception as e:
                    err_msg = f"request_failed: {e}"
                    write_jsonl_line(failures_file, {"id": row.get("id"), "error": err_msg})
                    record = RowRecord(
                        id=row.get("id"),
                        year=row.get("year"),
                        group=row.get("group"),
                        problem_number=row.get("problem_number"),
                        language=row.get("language"),
                        multimodal=row.get("multimodal"),
                        points=row.get("points"),
                        answer=(row.get("answer") or "").strip().upper() or None,
                        predicted=None,
                        is_correct=None,
                        points_earned=0,
                        reasoning_mode=args.reasoning,
                        latency_ms=None,
                        prompt_tokens=None,
                        completion_tokens=None,
                        total_tokens=None,
                        cost_usd=None,
                        rationale=None,
                        raw_text_response=None,
                        generation_id=None,
                        error=err_msg,
                        warnings=(warnings + [f"attempt_{attempt}_request_failed"]) if warnings else [f"attempt_{attempt}_request_failed"],
                    )
                    rows.append(record)
                    record_dict = asdict(record)
                    results_records.append(record_dict)
                    write_jsonl_line(results_jsonl, record_dict)
                    request_failed = True
                    break

                latencies.append(latency_ms)

                if resp.status_code != 200:
                    err_msg = f"http_{resp.status_code}: {resp.text[:500]}"
                    write_jsonl_line(failures_file, {"id": row.get("id"), "error": err_msg})
                    record = RowRecord(
                        id=row.get("id"),
                        year=row.get("year"),
                        group=row.get("group"),
                        problem_number=row.get("problem_number"),
                        language=row.get("language"),
                        multimodal=row.get("multimodal"),
                        points=row.get("points"),
                        answer=(row.get("answer") or "").strip().upper() or None,
                        predicted=None,
                        is_correct=None,
                        points_earned=0,
                        reasoning_mode=args.reasoning,
                        latency_ms=sum(latencies) if latencies else None,
                        prompt_tokens=None,
                        completion_tokens=None,
                        total_tokens=None,
                        cost_usd=None,
                        rationale=None,
                        raw_text_response=None,
                        generation_id=None,
                        error=err_msg,
                        warnings=warnings or None,
                    )
                    rows.append(record)
                    record_dict = asdict(record)
                    results_records.append(record_dict)
                    write_jsonl_line(results_jsonl, record_dict)
                    request_failed = True
                    break

                try:
                    data = resp.json()
                except Exception:
                    data = None

                content_text = resp.text or ""
                finish_reason = None
                native_finish = None

                if isinstance(data, dict):
                    gen_id = data.get("id")
                    model_echo = data.get("model")
                    choices = data.get("choices") or []
                    if choices and isinstance(choices, list):
                        choice0 = choices[0] or {}
                        finish_reason = choice0.get("finish_reason")
                        native_finish = choice0.get("native_finish_reason")
                        content_raw = ((choice0.get("message") or {}).get("content"))
                        content_text = normalize_message_content(content_raw)
                    usage = data.get("usage")
                    if isinstance(usage, dict):
                        last_usage = usage
                        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                            val = usage.get(key)
                            if isinstance(val, (int, float)):
                                combined_usage[key] = combined_usage.get(key, 0.0) + float(val)
                        cost_val = usage.get("cost") or usage.get("total_cost")
                        if isinstance(cost_val, str):
                            try:
                                cost_val = float(cost_val)
                            except Exception:
                                cost_val = None
                        if isinstance(cost_val, (int, float)):
                            combined_usage["cost"] = combined_usage.get("cost", 0.0) + float(cost_val)
                    write_jsonl_line(
                        raw_responses_file,
                        {
                            "id": row.get("id"),
                            "attempt": attempt,
                            "response": data,
                        },
                    )
                else:
                    write_jsonl_line(
                        raw_responses_file,
                        {
                            "id": row.get("id"),
                            "attempt": attempt,
                            "response": {"raw_text": content_text},
                        },
                    )

                ans, rat, parse_warn = parse_answer_from_text(content_text or "", args.reasoning)

                should_retry = False
                finish_values = [finish_reason, native_finish]
                if ans is None and attempt < max_attempts:
                    for reason in finish_values:
                        if isinstance(reason, str) and reason.lower() in {"length", "max_tokens", "max_tokens_exceeded"}:
                            should_retry = True
                            break

                if should_retry:
                    if max_tokens_current is None:
                        next_tokens = DEFAULT_RETRY_MAX_TOKENS
                    else:
                        next_tokens = min(max_tokens_current * 2, 4096)
                    warnings.append(
                        f"max_tokens_retry_{max_tokens_current if max_tokens_current is not None else 'auto'}->{next_tokens}"
                    )
                    max_tokens_current = next_tokens
                    continue

                if parse_warn:
                    warnings.append(parse_warn)

                break

            if request_failed:
                if args.fail_fast:
                    break
                else:
                    continue

            answer_value = row.get("answer")
            if pd.isna(answer_value) or answer_value is None:
                normalized_answer = ""
            else:
                normalized_answer = str(answer_value)

            gt = normalized_answer.strip().upper()
            if not gt or gt not in LETTER_SET:
                gt = None

            is_correct = (ans == gt) if (ans is not None and gt is not None) else None

            points_earned = 0.0
            if is_correct:
                raw_points = row.get("points")
                if pd.isna(raw_points) or raw_points is None or raw_points == "":
                    points_earned = 0.0
                else:
                    try:
                        points_earned = float(raw_points)
                    except (TypeError, ValueError):
                        points_earned = 0.0

            # Usage fields
            prompt_tokens = completion_tokens = total_tokens = None
            cost_usd = None
            if combined_usage:
                if "prompt_tokens" in combined_usage:
                    prompt_tokens = int(combined_usage["prompt_tokens"])
                if "completion_tokens" in combined_usage:
                    completion_tokens = int(combined_usage["completion_tokens"])
                if "total_tokens" in combined_usage:
                    total_tokens = int(combined_usage["total_tokens"])
                if "cost" in combined_usage:
                    cost_usd = float(combined_usage["cost"])
            elif isinstance(last_usage, dict):
                prompt_tokens = last_usage.get("prompt_tokens")
                completion_tokens = last_usage.get("completion_tokens")
                total_tokens = last_usage.get("total_tokens")
                cost_usd = last_usage.get("cost") or last_usage.get("total_cost")
                if isinstance(cost_usd, str):
                    try:
                        cost_usd = float(cost_usd)
                    except Exception:
                        cost_usd = None

            record = RowRecord(
                id=row.get("id"),
                year=row.get("year"),
                group=row.get("group"),
                problem_number=row.get("problem_number"),
                language=row.get("language"),
                multimodal=row.get("multimodal"),
                points=row.get("points"),
                answer=gt,
                predicted=ans,
                is_correct=is_correct,
                points_earned=points_earned,
                reasoning_mode=args.reasoning,
                latency_ms=sum(latencies) if latencies else None,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                rationale=rat,
                raw_text_response=content_text,
                generation_id=gen_id,
                error=None,
                warnings=warnings or None,
            )
            rows.append(record)
            record_dict = asdict(record)
            results_records.append(record_dict)
            write_jsonl_line(results_jsonl, record_dict)

    # Save artifacts
    # Build results DataFrame (ensure columns even if no rows)
    if rows:
        results_df = pd.DataFrame(results_records)
    else:
        results_df = pd.DataFrame(columns=[f.name for f in dataclasses.fields(RowRecord)])
    results_path = os.path.join(run_dir, "results.parquet")
    results_df.to_parquet(results_path, engine="pyarrow", index=False)

    results_json_path = os.path.join(run_dir, "results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_records, f, ensure_ascii=False, indent=2)

    # Failures
    # files already written incrementally

    # Metrics
    if len(results_df.columns) > 0:
        answered_mask = results_df["predicted"].notna() & results_df["error"].isna()
    else:
        answered_mask = pd.Series([], dtype=bool)
    answered_count = int(answered_mask.sum())
    if len(results_df.columns) > 0:
        failed_count = int(results_df["error"].notna().sum())
    else:
        failed_count = 0

    # Skipped count tracked externally
    skipped_count = skipped

    # Accuracy metrics
    if len(results_df.columns) > 0:
        correct_mask = results_df["is_correct"] == True
        accuracy = float((correct_mask & answered_mask).sum() / answered_count) if answered_count else 0.0
    else:
        accuracy = 0.0
    # Points-weighted accuracy
    if len(results_df.columns) > 0 and not answered_mask.empty:
        answered_points = results_df.loc[answered_mask, "points"].astype(float)
        earned_points = results_df.loc[answered_mask, "points_earned"].astype(float)
        p_weighted_accuracy = float(earned_points.sum() / answered_points.sum()) if answered_points.sum() > 0 else 0.0
    else:
        p_weighted_accuracy = 0.0

    # Latency
    if len(results_df.columns) > 0 and not answered_mask.empty:
        lat = results_df.loc[answered_mask, "latency_ms"].dropna().astype(float)
        mean_latency = float(lat.mean()) if not lat.empty else None
        median_latency = float(lat.median()) if not lat.empty else None
    else:
        mean_latency = None
        median_latency = None

    # Tokens / cost
    if len(results_df.columns) > 0 and not answered_mask.empty:
        total_tokens_series = results_df.loc[answered_mask, "total_tokens"].dropna().astype(int)
        mean_tokens = float(total_tokens_series.mean()) if not total_tokens_series.empty else None
        cost_series = results_df.loc[answered_mask, "cost_usd"].dropna().astype(float)
        total_cost = float(cost_series.sum()) if not cost_series.empty else 0.0
        unknown_usage_count = int(answered_count - len(total_tokens_series))
    else:
        mean_tokens = None
        total_cost = 0.0
        unknown_usage_count = 0

    # Breakdowns
    def breakdown_by(col: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if len(results_df.columns) == 0 or col not in results_df.columns or answered_count == 0:
            return result
        for key, sub in results_df.loc[answered_mask].groupby(col):
            sub_correct = (sub["is_correct"] == True).sum()
            sub_count = len(sub)
            sub_points = sub["points"].astype(float)
            sub_earned = sub["points_earned"].astype(float)
            sub_acc = float(sub_correct / sub_count) if sub_count else 0.0
            sub_pwa = float(sub_earned.sum() / sub_points.sum()) if sub_points.sum() > 0 else 0.0
            result[str(key)] = {"count": int(sub_count), "accuracy": sub_acc, "points_weighted_accuracy": sub_pwa}
        return result

    metrics = {
        "answered_count": answered_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
        "accuracy": accuracy,
        "points_weighted_accuracy": p_weighted_accuracy,
        "mean_latency_ms": mean_latency,
        "median_latency_ms": median_latency,
        "mean_total_tokens": mean_tokens,
        "total_cost_usd_known": total_cost,
        "unknown_usage_count": unknown_usage_count,
        "breakdown_by_group": breakdown_by("group"),
        "breakdown_by_year": breakdown_by("year"),
    }

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Save config
    args_snapshot = vars(args).copy()
    args_snapshot["dataset"] = dataset_path
    config = {
        "timestamp_utc": ts,
        "args": args_snapshot,
        "model": dataclasses.asdict(model_info),
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # Console summary
    print("Summary:")
    print(f"  Answered: {answered_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Points-weighted accuracy: {p_weighted_accuracy:.3f}")
    if mean_latency is not None:
        print(f"  Mean latency: {mean_latency:.1f} ms (median {median_latency:.1f} ms)")
    else:
        print("  Mean latency: n/a")
    print(f"  Known total cost: ${total_cost:.4f}")
    print(f"  Unknown usage rows: {unknown_usage_count}")


if __name__ == "__main__":
    main()
