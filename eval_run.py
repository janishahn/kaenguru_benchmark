import argparse
import asyncio
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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
import pandas as pd
from PIL import Image
from tqdm import tqdm

from console.metrics import Aggregator, UsageEvent

try:
    from console.dashboard import Dashboard

    DASHBOARD_AVAILABLE = True
except Exception:  # pragma: no cover - rich not installed
    Dashboard = None  # type: ignore
    DASHBOARD_AVAILABLE = False


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


def default_worker_count() -> int:
    cpu = os.cpu_count() or 4
    return max(2, min(8, cpu))


@dataclass
class ModelInfo:
    id: str
    label: Optional[str] = None
    supports_vision: bool = False
    supports_json_response_format: bool = False
    min_request_interval: Optional[float] = None


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
    # Extra usage details when available from providers (e.g., OpenRouter)
    reasoning_tokens: Optional[int] = None
    cached_prompt_tokens: Optional[int] = None
    audio_prompt_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    rationale: Optional[str] = None
    raw_text_response: Optional[str] = None
    generation_id: Optional[str] = None
    error: Optional[str] = None
    warnings: Optional[List[str]] = None


@dataclass
class WorkerOutcome:
    record: Optional[RowRecord]
    raw_entries: List[Dict[str, Any]]
    failure_entry: Optional[Dict[str, Any]]
    skipped: bool
    fail_fast_trigger: bool
    attempts: int = 1
    status_code: Optional[int] = None
    row_id: Optional[Any] = None


class AdaptiveRateLimiter:
    def __init__(self, initial_interval: float = 0.0) -> None:
        self._min_interval = max(0.0, initial_interval)
        self._last_timestamp = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = asyncio.get_running_loop().time()
                target = self._last_timestamp + self._min_interval
                if now >= target:
                    self._last_timestamp = now
                    return
                wait_time = target - now
            await asyncio.sleep(wait_time)

    async def record_throttle(self) -> None:
        async with self._lock:
            if self._min_interval == 0.0:
                self._min_interval = 0.5
            else:
                self._min_interval = min(self._min_interval * 2.0, 30.0)

    async def record_success(self) -> None:
        async with self._lock:
            if self._min_interval == 0.0:
                return
            self._min_interval = max(self._min_interval * 0.8, 0.0)

def read_models_registry(path: str) -> Dict[str, ModelInfo]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    models = {}
    for m in data.get("models", []):
        min_interval: Optional[float] = None
        rate_limit = m.get("rate_limit")
        if isinstance(rate_limit, dict):
            raw_min = rate_limit.get("min_interval_seconds")
            if isinstance(raw_min, (int, float)) and raw_min >= 0:
                min_interval = float(raw_min)
            raw_rps = rate_limit.get("requests_per_second")
            if min_interval is None and isinstance(raw_rps, (int, float)) and raw_rps > 0:
                min_interval = 1.0 / float(raw_rps)
            raw_rpm = rate_limit.get("requests_per_minute")
            if min_interval is None and isinstance(raw_rpm, (int, float)) and raw_rpm > 0:
                min_interval = 60.0 / float(raw_rpm)
        elif isinstance(rate_limit, (int, float)) and rate_limit > 0:
            min_interval = 1.0 / float(rate_limit)

        info = ModelInfo(
            id=m.get("id"),
            label=m.get("label"),
            supports_vision=bool(m.get("supports_vision", False)),
            supports_json_response_format=bool(m.get("supports_json_response_format", False)),
            min_request_interval=min_interval,
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
    """Open image bytes without forcing colorspace.

    Preserve alpha and original mode; downstream encoders decide the right
    output format and perform conversion only when needed (e.g. JPEG).
    """
    try:
        img = Image.open(BytesIO(img_bytes))
        # Ensure the image is actually loaded to catch decoding errors early
        img.load()
        return img
    except Exception:
        return None


def image_to_data_url(
    img: Image.Image,
    *,
    prefer_format: Optional[str] = None,
    max_dim: int = 1024,
    jpeg_quality: int = 85,
) -> Tuple[str, str]:
    # Downscale while preserving aspect ratio; never upscale.
    if max(img.size) > max_dim:
        img = img.copy()
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

    # Choose encoding format based on alpha channel and preference
    fmt = (prefer_format or "").upper().strip()
    has_alpha = (getattr(img, "mode", "").upper() in {"RGBA", "LA"})
    if not fmt:
        fmt = "PNG" if has_alpha else "JPEG"
    if fmt == "JPG":
        fmt = "JPEG"
    if fmt not in {"PNG", "JPEG"}:
        fmt = "PNG" if has_alpha else "JPEG"
    mime = "image/png" if fmt == "PNG" else "image/jpeg"

    # Encode
    buf = BytesIO()
    save_kwargs = {"format": fmt}
    if fmt == "JPEG":
        save_kwargs.update({"quality": int(max(50, min(100, jpeg_quality))), "optimize": True, "progressive": True})
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
    else:
        # For PNG, enable optimization; avoid palette conversion to preserve details.
        save_kwargs.update({"optimize": True})
    img.save(buf, **save_kwargs)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}", fmt


def build_messages(
    row: Dict[str, Any],
    reasoning: str,
    model: ModelInfo,
    encoded_images: Dict[str, Optional[str]],
    *,
    image_detail: str = "auto",
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
                "image_url": {"url": encoded_images["question"], "detail": image_detail},
            }
        )

    # Associated images
    assoc = encoded_images.get("assoc_list") or []
    for i, url in enumerate(assoc, start=1):
        if url:
            a_label = ("Zusatzbild" if is_de else "Additional image") + f" {i}:"
            content_parts.append({"type": "text", "text": a_label})
            content_parts.append(
                {"type": "image_url", "image_url": {"url": url, "detail": image_detail}}
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
                {"type": "image_url", "image_url": {"url": url, "detail": image_detail}}
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


async def request_with_retries(
    client: httpx.AsyncClient,
    limiter: AdaptiveRateLimiter,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    on_status: Optional[Callable[[int], None]] = None,
    on_throttle: Optional[Callable[[], None]] = None,
) -> Tuple[httpx.Response, float]:
    delays = [0.5, 1.0, 2.0]
    last_exc: Optional[Exception] = None
    start = time.perf_counter()
    for attempt in range(len(delays) + 1):
        await limiter.acquire()
        try:
            resp = await client.post(url, headers=headers, json=payload)
        except Exception as exc:
            last_exc = exc
            if attempt < len(delays):
                await asyncio.sleep(delays[attempt])
                continue
            raise

        if on_status is not None:
            try:
                on_status(resp.status_code)
            except Exception:
                pass

        if resp.status_code in (429, 500, 502, 503, 504):
            if on_throttle is not None:
                try:
                    on_throttle()
                except Exception:
                    pass
            await limiter.record_throttle()
            if attempt < len(delays):
                await asyncio.sleep(delays[attempt])
                continue
        else:
            await limiter.record_success()

        latency_ms = (time.perf_counter() - start) * 1000.0
        return resp, latency_ms

    raise last_exc if last_exc else RuntimeError("request failed")


def build_failure_record(
    row: Dict[str, Any],
    args: argparse.Namespace,
    latencies: List[float],
    warnings: List[str],
    error_msg: str,
    raw_text_response: Optional[str],
) -> RowRecord:
    answer_value = row.get("answer")
    if pd.isna(answer_value) or answer_value is None:
        normalized_answer = ""
    else:
        normalized_answer = str(answer_value)
    gt = normalized_answer.strip().upper()
    if not gt or gt not in LETTER_SET:
        gt = None

    return RowRecord(
        id=row.get("id"),
        year=row.get("year"),
        group=row.get("group"),
        problem_number=row.get("problem_number"),
        language=row.get("language"),
        multimodal=row.get("multimodal"),
        points=row.get("points"),
        answer=gt,
        predicted=None,
        is_correct=None,
        points_earned=0.0,
        reasoning_mode=args.reasoning,
        latency_ms=sum(latencies) if latencies else None,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        cost_usd=None,
        rationale=None,
        raw_text_response=raw_text_response,
        generation_id=None,
        error=error_msg,
        warnings=warnings or None,
    )


async def evaluate_single_row(
    row: Dict[str, Any],
    args: argparse.Namespace,
    model_info: ModelInfo,
    client: httpx.AsyncClient,
    limiter: AdaptiveRateLimiter,
    url: str,
    headers: Dict[str, str],
    metrics: Optional[Aggregator] = None,
) -> WorkerOutcome:
    warnings: List[str] = []
    raw_entries: List[Dict[str, Any]] = []

    q_bytes = coerce_bytes(row.get("question_image"))
    opt_bytes = {letter: coerce_bytes(row.get(f"sol_{letter}_image_bin")) for letter in LETTER_SET}
    assoc_bytes = coerce_list_of_bytes(row.get("associated_images_bin")) or []

    has_images = bool(q_bytes or any(opt_bytes.values()) or assoc_bytes)
    if has_images and not model_info.supports_vision:
        return WorkerOutcome(
            record=None,
            raw_entries=[],
            failure_entry=None,
            skipped=True,
            fail_fast_trigger=False,
            attempts=0,
            status_code=None,
            row_id=row.get("id"),
        )

    encoded_images: Dict[str, Any] = {"assoc_list": []}
    if model_info.supports_vision:
        if q_bytes:
            img = pil_from_bytes(q_bytes)
            if img is None:
                warnings.append("question_image_decode_failed")
            else:
                url_data, _ = image_to_data_url(
                    img,
                    max_dim=(args.image_max_dim or 1024),
                    jpeg_quality=(args.image_jpeg_quality or 85),
                )
                encoded_images["question"] = url_data

        for letter in ["A", "B", "C", "D", "E"]:
            b = opt_bytes.get(letter)
            if b:
                img = pil_from_bytes(b)
                if img is None:
                    warnings.append(f"opt_{letter}_image_decode_failed")
                else:
                    url_data, _ = image_to_data_url(
                        img,
                        max_dim=(args.image_max_dim or 1024),
                        jpeg_quality=(args.image_jpeg_quality or 85),
                    )
                    encoded_images[f"opt_{letter}"] = url_data

        for idx, b in enumerate(assoc_bytes):
            img = pil_from_bytes(b)
            if img is None:
                warnings.append(f"assoc_{idx+1}_image_decode_failed")
                encoded_images["assoc_list"].append(None)
            else:
                url_data, _ = image_to_data_url(
                    img,
                    max_dim=(args.image_max_dim or 1024),
                    jpeg_quality=(args.image_jpeg_quality or 85),
                )
                encoded_images["assoc_list"].append(url_data)

    messages = build_messages(
        row,
        args.reasoning,
        model_info,
        encoded_images,
        image_detail=(args.image_detail or "auto"),
    )

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
    combined_usage_details: Dict[str, float] = {}  # reasoning_tokens, cached_prompt_tokens, audio_prompt_tokens
    last_usage: Optional[Dict[str, Any]] = None
    latencies: List[float] = []
    last_status_code: Optional[int] = None
    ans: Optional[str] = None
    rat: Optional[str] = None
    parse_warn: Optional[str] = None
    content_text = ""
    gen_id = None
    finish_reason = None
    native_finish = None

    while attempt < max_attempts:
        attempt += 1
        if max_tokens_current is not None:
            payload["max_tokens"] = max_tokens_current
        else:
            payload.pop("max_tokens", None)

        try:
            resp, latency_ms = await request_with_retries(
                client,
                limiter,
                url,
                headers,
                payload,
                on_status=(metrics.record_status if metrics else None),
                on_throttle=(metrics.record_throttle if metrics else None),
            )
            latencies.append(latency_ms)
            last_status_code = resp.status_code
        except Exception as exc:
            err_msg = f"request_failed: {exc}"
            failure_entry = {"id": row.get("id"), "error": err_msg}
            record = build_failure_record(row, args, latencies, warnings, err_msg, None)
            return WorkerOutcome(
                record=record,
                raw_entries=raw_entries,
                failure_entry=failure_entry,
                skipped=False,
                fail_fast_trigger=args.fail_fast,
                attempts=attempt,
                status_code=None,
                row_id=row.get("id"),
            )

        if resp.status_code != 200:
            content_text = resp.text or ""
            snippet = re.sub(r"\s+", " ", content_text.strip()) if content_text else ""
            if snippet:
                snippet = snippet[:200]
            err_msg = f"http_error_{resp.status_code}"
            if snippet:
                err_msg = f"{err_msg}: {snippet}"
            warnings.append(f"http_status_{resp.status_code}")
            try:
                error_payload = resp.json()
            except Exception:
                error_payload = {"raw_text": content_text}
            raw_entries.append(
                {
                    "id": row.get("id"),
                    "attempt": attempt,
                    "response": error_payload,
                    "status_code": resp.status_code,
                }
            )
            failure_entry = {
                "id": row.get("id"),
                "status_code": resp.status_code,
                "error": err_msg,
            }
            record = build_failure_record(row, args, latencies, warnings, err_msg, content_text or None)
            return WorkerOutcome(
                record=record,
                raw_entries=raw_entries,
                failure_entry=failure_entry,
                skipped=False,
                fail_fast_trigger=args.fail_fast,
                attempts=attempt,
                status_code=resp.status_code,
                row_id=row.get("id"),
            )

        try:
            data = resp.json()
        except Exception:
            data = None

        content_text = resp.text or ""
        finish_reason = None
        native_finish = None

        if isinstance(data, dict):
            gen_id = data.get("id")
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
                # Details when available
                prompt_details = usage.get("prompt_tokens_details") or {}
                if isinstance(prompt_details, dict):
                    for k_src, k_dst in (("cached_tokens", "cached_prompt_tokens"), ("audio_tokens", "audio_prompt_tokens")):
                        val = prompt_details.get(k_src)
                        if isinstance(val, (int, float)):
                            combined_usage_details[k_dst] = combined_usage_details.get(k_dst, 0.0) + float(val)
                completion_details = usage.get("completion_tokens_details") or {}
                if isinstance(completion_details, dict):
                    rt = completion_details.get("reasoning_tokens")
                    if isinstance(rt, (int, float)):
                        combined_usage_details["reasoning_tokens"] = combined_usage_details.get("reasoning_tokens", 0.0) + float(rt)
                cost_val = usage.get("cost") or usage.get("total_cost")
                if isinstance(cost_val, str):
                    try:
                        cost_val = float(cost_val)
                    except Exception:
                        cost_val = None
                if isinstance(cost_val, (int, float)):
                    combined_usage["cost"] = combined_usage.get("cost", 0.0) + float(cost_val)
            # Fallback: some providers put usage into an X-Usage header
            elif hasattr(resp, "headers"):
                hdr = None
                try:
                    hdr = resp.headers.get("x-usage") or resp.headers.get("X-Usage")
                except Exception:
                    hdr = None
                if hdr:
                    parsed: Optional[Dict[str, Any]] = None
                    # First try JSON
                    try:
                        parsed_json = json.loads(hdr)
                        if isinstance(parsed_json, dict):
                            parsed = parsed_json
                    except Exception:
                        parsed = None
                    # Then try a simple key=value parser (comma/semicolon separated)
                    if parsed is None:
                        try:
                            kv: Dict[str, float] = {}
                            for part in re.split(r"[,;]", hdr):
                                if "=" not in part:
                                    continue
                                k, v = part.split("=", 1)
                                k = k.strip()
                                v = v.strip()
                                if not k:
                                    continue
                                try:
                                    kv[k] = float(v)
                                except Exception:
                                    continue
                            if kv:
                                parsed = dict(kv)
                        except Exception:
                            parsed = None
                    # Apply parsed usage fields if any
                    if isinstance(parsed, dict):
                        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                            val = parsed.get(key)
                            if isinstance(val, (int, float)):
                                combined_usage[key] = combined_usage.get(key, 0.0) + float(val)
                        # Details: either nested objects or dot-keys
                        prompt_details = parsed.get("prompt_tokens_details") if isinstance(parsed.get("prompt_tokens_details"), dict) else {}
                        completion_details = parsed.get("completion_tokens_details") if isinstance(parsed.get("completion_tokens_details"), dict) else {}

                        # Also support flattened keys like "prompt_tokens_details.cached_tokens=123"
                        def _maybe_from_flat(src_key: str) -> Optional[float]:
                            val = parsed.get(src_key)
                            try:
                                if isinstance(val, str):
                                    return float(val)
                                if isinstance(val, (int, float)):
                                    return float(val)
                            except Exception:
                                return None
                            return None

                        for k_src, k_dst in (("cached_tokens", "cached_prompt_tokens"), ("audio_tokens", "audio_prompt_tokens")):
                            val = (prompt_details or {}).get(k_src)
                            if not isinstance(val, (int, float)):
                                val = _maybe_from_flat(f"prompt_tokens_details.{k_src}")
                            if isinstance(val, (int, float)):
                                combined_usage_details[k_dst] = combined_usage_details.get(k_dst, 0.0) + float(val)

                        rt = (completion_details or {}).get("reasoning_tokens")
                        if not isinstance(rt, (int, float)):
                            rt = _maybe_from_flat("completion_tokens_details.reasoning_tokens")
                        if isinstance(rt, (int, float)):
                            combined_usage_details["reasoning_tokens"] = combined_usage_details.get("reasoning_tokens", 0.0) + float(rt)

                        cost_val = parsed.get("cost") or parsed.get("total_cost")
                        if isinstance(cost_val, (int, float)):
                            combined_usage["cost"] = combined_usage.get("cost", 0.0) + float(cost_val)
            raw_entries.append(
                {
                    "id": row.get("id"),
                    "attempt": attempt,
                    "response": data,
                }
            )
        else:
            raw_entries.append(
                {
                    "id": row.get("id"),
                    "attempt": attempt,
                    "response": {"raw_text": content_text},
                }
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

    prompt_tokens = completion_tokens = total_tokens = None
    reasoning_tokens = cached_prompt_tokens = audio_prompt_tokens = None
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
        if "reasoning_tokens" in combined_usage_details:
            reasoning_tokens = int(combined_usage_details["reasoning_tokens"])
        if "cached_prompt_tokens" in combined_usage_details:
            cached_prompt_tokens = int(combined_usage_details["cached_prompt_tokens"])
        if "audio_prompt_tokens" in combined_usage_details:
            audio_prompt_tokens = int(combined_usage_details["audio_prompt_tokens"])
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
        # Details (single-attempt fallback)
        try:
            prompt_details_map = last_usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details_map, dict):
                cached_prompt_tokens = (
                    prompt_details_map.get("cached_tokens") if isinstance(prompt_details_map.get("cached_tokens"), int) else cached_prompt_tokens
                )
                audio_prompt_tokens = (
                    prompt_details_map.get("audio_tokens") if isinstance(prompt_details_map.get("audio_tokens"), int) else audio_prompt_tokens
                )
            completion_details_map = last_usage.get("completion_tokens_details") or {}
            if isinstance(completion_details_map, dict):
                reasoning_tokens = (
                    completion_details_map.get("reasoning_tokens")
                    if isinstance(completion_details_map.get("reasoning_tokens"), int)
                    else reasoning_tokens
                )
        except Exception:
            pass

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
        reasoning_tokens=reasoning_tokens,
        cached_prompt_tokens=cached_prompt_tokens,
        audio_prompt_tokens=audio_prompt_tokens,
        cost_usd=cost_usd,
        rationale=rat,
        raw_text_response=content_text,
        generation_id=gen_id,
        error=None,
        warnings=warnings or None,
    )

    return WorkerOutcome(
        record=record,
        raw_entries=raw_entries,
        failure_entry=None,
        skipped=False,
        fail_fast_trigger=False,
        attempts=attempt,
        status_code=last_status_code or 200,
        row_id=row.get("id"),
    )


async def evaluate_rows_async(
    rows_data: List[Dict[str, Any]],
    args: argparse.Namespace,
    model_info: ModelInfo,
    url: str,
    headers: Dict[str, str],
    results_jsonl,
    raw_responses_file,
    failures_file,
    worker_count: int,
    *,
    dashboard_enabled: bool,
    dashboard_refresh_hz: float,
    dashboard_recent: int,
    dashboard_compact: bool,
    events_path: Optional[Path],
) -> Tuple[List[RowRecord], List[Dict[str, Any]], int]:
    rows: List[RowRecord] = []
    results_records: List[Dict[str, Any]] = []
    skipped = 0

    limiter = AdaptiveRateLimiter(initial_interval=model_info.min_request_interval or 0.0)
    work_queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
    result_queue: asyncio.Queue[Optional[WorkerOutcome]] = asyncio.Queue()
    stop_event = asyncio.Event()

    inflight_lock = asyncio.Lock()
    active_inflight = 0

    aggregator = Aggregator(
        len(rows_data),
        model_id=model_info.id,
        events_path=events_path,
        recent_items=dashboard_recent,
        min_request_interval=model_info.min_request_interval,
    )

    dashboard = None
    progress: Optional[tqdm] = None
    if dashboard_enabled and DASHBOARD_AVAILABLE and Dashboard is not None:
        dashboard = Dashboard(
            aggregator,
            refresh_hz=dashboard_refresh_hz,
            compact=dashboard_compact,
            recent_rows=dashboard_recent,
        )
        dashboard.start()
        dashboard.update(aggregator.snapshot(in_flight=0, worker_count=worker_count))
    else:
        progress = tqdm(total=len(rows_data), desc="Evaluating", unit="q")

    def build_usage_event(outcome: WorkerOutcome) -> UsageEvent:
        record = outcome.record

        def as_int(value: Any) -> Optional[int]:
            if value is None:
                return None
            try:
                if pd.isna(value):  # type: ignore[arg-type]
                    return None
            except Exception:
                pass
            try:
                return int(value)
            except Exception:
                return None

        def as_float(value: Any) -> Optional[float]:
            if value is None:
                return None
            try:
                if pd.isna(value):  # type: ignore[arg-type]
                    return None
            except Exception:
                pass
            try:
                return float(value)
            except Exception:
                return None

        event_type = "skipped"
        if not outcome.skipped:
            if record is not None and record.error:
                event_type = "failure"
            else:
                event_type = "success"

        group_value: Optional[str] = None
        multimodal_value: Optional[bool] = None
        if record is not None:
            if record.group is not None and record.group != "":
                group_value = str(record.group)
            if isinstance(record.multimodal, bool):
                multimodal_value = record.multimodal
            elif record.multimodal is not None:
                multimodal_value = bool(record.multimodal)

        event = UsageEvent(
            type=event_type,
            row_id=(record.id if record is not None else outcome.row_id),
            year=as_int(record.year) if record is not None else None,
            group=group_value,
            points=as_float(record.points) if record is not None else None,
            problem_number=as_int(record.problem_number) if record is not None else None,
            multimodal=multimodal_value,
            latency_ms=record.latency_ms if record is not None else None,
            attempts=outcome.attempts or 1,
            prompt_tokens=record.prompt_tokens if record is not None else None,
            completion_tokens=record.completion_tokens if record is not None else None,
            total_tokens=record.total_tokens if record is not None else None,
            reasoning_tokens=getattr(record, "reasoning_tokens", None) if record is not None else None,
            cached_prompt_tokens=getattr(record, "cached_prompt_tokens", None) if record is not None else None,
            audio_prompt_tokens=getattr(record, "audio_prompt_tokens", None) if record is not None else None,
            cost_usd_known=record.cost_usd if record is not None else None,
            predicted=record.predicted if record is not None else None,
            correct=record.is_correct if record is not None else None,
            status_code=outcome.status_code,
            warnings=record.warnings if record is not None else None,
        )
        return event

    async def worker(client: httpx.AsyncClient) -> None:
        nonlocal active_inflight
        while True:
            item = await work_queue.get()
            if item is None:
                work_queue.task_done()
                break
            if args.fail_fast and stop_event.is_set():
                work_queue.task_done()
                continue
            async with inflight_lock:
                active_inflight += 1
            outcome = await evaluate_single_row(item, args, model_info, client, limiter, url, headers, metrics=aggregator)
            await result_queue.put(outcome)
            async with inflight_lock:
                active_inflight = max(0, active_inflight - 1)
            work_queue.task_done()
            if args.fail_fast and outcome.fail_fast_trigger:
                stop_event.set()

    async def consumer() -> None:
        nonlocal skipped
        while True:
            outcome = await result_queue.get()
            if outcome is None:
                result_queue.task_done()
                break
            if outcome.skipped:
                skipped += 1
            if outcome.failure_entry is not None:
                write_jsonl_line(failures_file, outcome.failure_entry)
            for entry in outcome.raw_entries:
                write_jsonl_line(raw_responses_file, entry)
            if outcome.record is not None:
                rows.append(outcome.record)
                record_dict = asdict(outcome.record)
                results_records.append(record_dict)
                write_jsonl_line(results_jsonl, record_dict)
            event = build_usage_event(outcome)
            aggregator.record_event(event)
            if dashboard is not None:
                async with inflight_lock:
                    inflight_current = active_inflight
                snapshot = aggregator.snapshot(in_flight=inflight_current, worker_count=worker_count)
                dashboard.update(snapshot)
            else:
                if progress is not None:
                    progress.update(1)
            result_queue.task_done()

    limits = httpx.Limits(max_connections=max(4, worker_count * 2), max_keepalive_connections=max(2, worker_count))
    timeout = httpx.Timeout(120.0)

    try:
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            for item in rows_data:
                await work_queue.put(item)
            for _ in range(worker_count):
                await work_queue.put(None)

            workers = [asyncio.create_task(worker(client)) for _ in range(worker_count)]
            consumer_task = asyncio.create_task(consumer())

            await asyncio.gather(*workers)
            await result_queue.put(None)
            await consumer_task
    finally:
        if dashboard is not None:
            dashboard.stop()
        if progress is not None:
            progress.close()
        aggregator.close()

    return rows, results_records, skipped



def main():
    import time
    start_time = time.time()  # Track start time for total run time
    
    parser = argparse.ArgumentParser(description="LLM evaluation runner for Känguru benchmark")
    parser.add_argument("--dataset", required=True, help="Path to dataset .parquet")
    parser.add_argument("--model", required=True, help="Model ID from models.json")
    parser.add_argument("--reasoning", choices=["none", "cot"], default="none")
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Process rows sequentially instead of using the default concurrent worker pool.",
    )
    parser.add_argument("--output_dir", default="runs")
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--live-dashboard", dest="live_dashboard", action="store_true", help="Force enable the Rich dashboard even on non-tty outputs.")
    parser.add_argument("--no-live-dashboard", dest="live_dashboard", action="store_false", help="Disable the Rich dashboard even on ttys.")
    parser.set_defaults(live_dashboard=None)
    parser.add_argument("--dashboard-refresh-hz", type=float, default=5.0, help="Refresh rate for the live dashboard (updates per second).")
    parser.add_argument("--events-jsonl", default="auto", help="Path to usage events JSONL log ('off' to disable, default auto).")
    parser.add_argument("--no-events-jsonl", dest="events_jsonl", action="store_const", const="off", help="Disable usage events capture.")
    parser.add_argument("--recent-items", type=int, default=20, help="Number of recent items to show in the dashboard table.")
    parser.add_argument("--ui-compact", action="store_true", help="Use compact dashboard layout suitable for smaller terminals.")
    # Image controls for multimodal inputs
    parser.add_argument("--image_max_dim", type=int, default=1024, help="Max image dimension (long edge) in pixels; no upscaling")
    parser.add_argument("--image_jpeg_quality", type=int, default=85, help="JPEG quality (50–100)")
    parser.add_argument("--image_detail", choices=["auto", "low", "high"], default="auto", help="Vision detail hint for providers that support it")
    args = parser.parse_args()

    stdout_isatty = sys.stdout.isatty()
    if args.live_dashboard is True:
        dashboard_enabled = True
    elif args.live_dashboard is False:
        dashboard_enabled = False
    else:
        dashboard_enabled = stdout_isatty
    if dashboard_enabled and not DASHBOARD_AVAILABLE:
        print("Rich dashboard unavailable; falling back to tqdm progress bar.", file=sys.stderr)
        dashboard_enabled = False
    dashboard_refresh = max(1.0, float(args.dashboard_refresh_hz or 5.0))
    dashboard_recent = max(5, int(args.recent_items or 20))
    events_config = args.events_jsonl or "auto"

    if dashboard_enabled and not stdout_isatty:
        print("Warning: live dashboard forced on a non-TTY output; layout may degrade.", file=sys.stderr)

    load_env_file()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable is required.", file=sys.stderr)
        sys.exit(2)

    models = read_models_registry(os.path.join(os.path.dirname(__file__), "models.json"))
    if args.model not in models:
        print(f"ERROR: model '{args.model}' not found in models.json", file=sys.stderr)
        sys.exit(2)
    model_info = models[args.model]

    try:
        dataset_path = resolve_dataset_path(args.dataset)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    run_dir, ts = ensure_output_dir(args.output_dir, args.model)

    events_path: Optional[Path]
    if isinstance(events_config, str):
        cfg = events_config.strip().lower()
    else:
        cfg = "auto"
    if cfg == "off":
        events_path = None
    elif cfg in {"", "auto"}:
        events_path = Path(run_dir) / "usage_events.jsonl"
    else:
        events_path = Path(events_config).expanduser()

    df = pd.read_parquet(dataset_path, engine="pyarrow")
    validate_dataset_columns(df)

    if args.limit is not None and args.limit < len(df):
        if args.seed is not None:
            df = df.sample(n=args.limit, random_state=args.seed)
        else:
            df = df.head(args.limit)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "kaenguru-benchmark-eval",
    }
    url = "https://openrouter.ai/api/v1/chat/completions"

    rows_data = df.to_dict(orient="records")

    worker_count = 1 if args.sequential else default_worker_count()
    if not rows_data:
        worker_count = 1

    results_jsonl_path = os.path.join(run_dir, "results.jsonl")
    raw_responses_path = os.path.join(run_dir, "raw_responses.jsonl")
    failures_jsonl_path = os.path.join(run_dir, "failures.jsonl")

    initial_interval = model_info.min_request_interval or 0.0
    if initial_interval > 0:
        initial_rpm = 60.0 / initial_interval
        limiter_desc = f"seeded at {initial_rpm:.1f} rpm"
    else:
        limiter_desc = "adaptive (no seed)"

    multimodal_count = sum(1 for row in rows_data if bool(row.get("multimodal")))
    avg_points = float(df["points"].dropna().astype(float).mean()) if not df["points"].dropna().empty else None
    year_counts = df["year"].value_counts().to_dict() if "year" in df.columns else {}
    year_desc = ", ".join(f"{k}:{v}" for k, v in sorted(year_counts.items())) if year_counts else "n/a"

    print("Configuration:")
    print(f"  Model: {model_info.id}")
    if model_info.label:
        print(f"  Label: {model_info.label}")
    print(f"  Dataset rows: {len(rows_data)}")
    print(f"  Multimodal rows: {multimodal_count} ({multimodal_count/len(rows_data)*100:.1f}% )" if rows_data else "  Multimodal rows: 0")
    print(f"  Mode: {'sequential' if worker_count == 1 else f'concurrent x{worker_count}'}")
    print(f"  Rate limiter: {limiter_desc}")
    print(f"  Live dashboard: {'on' if dashboard_enabled else 'off'}")
    if events_path is not None:
        print(f"  Usage events: {str(events_path)}")
    else:
        print("  Usage events: disabled")
    print(f"  Reasoning: {args.reasoning}")
    if avg_points is not None:
        print(f"  Avg points: {avg_points:.2f}")
    print(f"  Year distribution: {year_desc}")
    # Add proper spacing after pre-run summary to separate from dashboard
    print()

    with open(results_jsonl_path, "w", encoding="utf-8") as results_jsonl, \
        open(raw_responses_path, "w", encoding="utf-8") as raw_responses_file, \
        open(failures_jsonl_path, "w", encoding="utf-8") as failures_file:

        rows, results_records, skipped = asyncio.run(
            evaluate_rows_async(
                rows_data,
                args,
                model_info,
                url,
                headers,
                results_jsonl,
                raw_responses_file,
                failures_file,
                worker_count,
                dashboard_enabled=dashboard_enabled,
                dashboard_refresh_hz=dashboard_refresh,
                dashboard_recent=dashboard_recent,
                dashboard_compact=bool(args.ui_compact),
                events_path=events_path,
            )
        )

    if rows:
        results_df = pd.DataFrame(results_records)
    else:
        results_df = pd.DataFrame(columns=[f.name for f in dataclasses.fields(RowRecord)])
    results_path = os.path.join(run_dir, "results.parquet")
    results_df.to_parquet(results_path, engine="pyarrow", index=False)

    results_json_path = os.path.join(run_dir, "results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_records, f, ensure_ascii=False, indent=2)

    if len(results_df.columns) > 0:
        answered_mask = results_df["predicted"].notna() & results_df["error"].isna()
    else:
        answered_mask = pd.Series([], dtype=bool)
    answered_count = int(answered_mask.sum())
    if len(results_df.columns) > 0:
        failed_count = int(results_df["error"].notna().sum())
    else:
        failed_count = 0

    skipped_count = skipped

    if len(results_df.columns) > 0:
        correct_mask = results_df["is_correct"] == True
        accuracy = float((correct_mask & answered_mask).sum() / answered_count) if answered_count else 0.0
    else:
        accuracy = 0.0

    if len(results_df.columns) > 0 and not answered_mask.empty:
        answered_points = results_df.loc[answered_mask, "points"].astype(float)
        earned_points = results_df.loc[answered_mask, "points_earned"].astype(float)
        p_weighted_accuracy = float(earned_points.sum() / answered_points.sum()) if answered_points.sum() > 0 else 0.0
    else:
        p_weighted_accuracy = 0.0

    if len(results_df.columns) > 0 and not answered_mask.empty:
        lat = results_df.loc[answered_mask, "latency_ms"].dropna().astype(float)
        mean_latency = float(lat.mean()) if not lat.empty else None
        median_latency = float(lat.median()) if not lat.empty else None
    else:
        mean_latency = None
        median_latency = None

    total_tokens_series = pd.Series(dtype="int64")
    if len(results_df.columns) > 0 and not answered_mask.empty:
        total_tokens_series = results_df.loc[answered_mask, "total_tokens"].dropna().astype(int)
        mean_tokens = float(total_tokens_series.mean()) if not total_tokens_series.empty else None
        cost_series = results_df.loc[answered_mask, "cost_usd"].dropna().astype(float)
        total_cost = float(cost_series.sum()) if not cost_series.empty else 0.0
        if "reasoning_tokens" in results_df.columns:
            reasoning_tokens_series = (
                results_df.loc[answered_mask, "reasoning_tokens"].dropna().astype(int)
            )
        else:
            reasoning_tokens_series = pd.Series(dtype="int64")
        mean_reasoning_tokens = (
            float(reasoning_tokens_series.mean()) if not reasoning_tokens_series.empty else None
        )
        total_reasoning_tokens = (
            int(reasoning_tokens_series.sum()) if not reasoning_tokens_series.empty else None
        )
        reasoning_tokens_known_count = int(len(reasoning_tokens_series))
        unknown_usage_count = int(answered_count - len(total_tokens_series))
    else:
        mean_tokens = None
        total_cost = 0.0
        unknown_usage_count = 0
        mean_reasoning_tokens = None
        total_reasoning_tokens = None
        reasoning_tokens_known_count = 0

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
        "mean_reasoning_tokens": mean_reasoning_tokens,
        "total_reasoning_tokens": total_reasoning_tokens,
        "reasoning_tokens_known_count": reasoning_tokens_known_count,
        "total_cost_usd_known": total_cost,
        "unknown_usage_count": unknown_usage_count,
        "breakdown_by_group": breakdown_by("group"),
        "breakdown_by_year": breakdown_by("year"),
    }

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    args_snapshot = vars(args).copy()
    args_snapshot["dataset"] = dataset_path
    args_snapshot["worker_count"] = worker_count
    config = {
        "timestamp_utc": ts,
        "args": args_snapshot,
        "model": dataclasses.asdict(model_info),
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # Add proper spacing before post-run summary to avoid cramped appearance
    print()
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
    total_tokens_known = int(len(total_tokens_series))
    if mean_tokens is not None:
        print(
            "  Total tokens: "
            f"mean {mean_tokens:.1f}, total {int(total_tokens_series.sum()) if total_tokens_series.size else 0} (rows {total_tokens_known})"
        )
    else:
        print("  Total tokens: n/a")
    if total_reasoning_tokens is not None and mean_reasoning_tokens is not None:
        print(
            "  Reasoning tokens: "
            f"mean {mean_reasoning_tokens:.1f}, total {total_reasoning_tokens} (rows {reasoning_tokens_known_count})"
        )
    else:
        print("  Reasoning tokens: n/a")
    print(f"  Known total cost: ${total_cost:.4f}")
    print(f"  Unknown usage rows: {unknown_usage_count}")
    print(f"  Worker count: {worker_count}")
    
    # Calculate and display total runtime
    end_time = time.time()
    total_runtime = end_time - start_time
    hours, rem = divmod(int(total_runtime), 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        runtime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        runtime_str = f"{minutes:02d}:{seconds:02d}"
    
    print(f"  Total runtime: {runtime_str}")


if __name__ == "__main__":
    main()
