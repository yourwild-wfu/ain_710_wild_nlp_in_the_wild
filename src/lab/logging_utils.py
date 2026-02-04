"""
logging_utils.py

Lightweight experiment logging utilities.

Design goals:
- Write JSON Lines (.jsonl): one record per line (append-only).
- Safe by default: avoid leaking secrets into logs.
- Minimal dependencies and easy to analyze later with pandas.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# --- Defaults ---
# Resolve repo root dynamically (…/ain710-week5-nlp-in-the-wild)
REPO_ROOT = Path(__file__).resolve()
for _ in range(3):
    REPO_ROOT = REPO_ROOT.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_LOG_FILE = "runs.jsonl"


# --- Redaction helpers ---
_API_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9]{10,}\b")


def _utc_now_iso() -> str:
    """
    Return current UTC time in ISO 8601 format.
    Ensures the result is a clean string.
    """
    return str(datetime.now(timezone.utc).isoformat())


def _safe_json(obj: Any) -> Any:
    """
    Convert objects into JSON-serializable forms when possible.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    # Fallback: string representation
    return str(obj)


def redact_secrets(value: Any) -> Any:
    """
    Redact secrets in strings/dicts/lists recursively.
    Specifically masks patterns like OpenAI API keys (sk-...).
    """
    if value is None:
        return None

    if isinstance(value, str):
        return _API_KEY_PATTERN.sub("sk-REDACTED", value)

    if isinstance(value, dict):
        redacted: Dict[str, Any] = {}
        for k, v in value.items():
            key_lower = str(k).lower()
            if "api_key" in key_lower or "authorization" in key_lower:
                redacted[str(k)] = "REDACTED"
            else:
                redacted[str(k)] = redact_secrets(v)
        return redacted

    if isinstance(value, (list, tuple)):
        return [redact_secrets(v) for v in value]

    return value


def get_default_log_path(
    outputs_dir: Path = DEFAULT_OUTPUT_DIR,
    log_filename: str = DEFAULT_LOG_FILE,
) -> Path:
    """
    Get the absolute path to the log file.
    """
    return outputs_dir / log_filename


def ensure_output_dir(path: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """
    Ensure output directory exists. Returns the path.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def new_run_id(prefix: str = "run") -> str:
    """
    Create a short, sortable-ish run id.
    Example: run_20260203T235959Z_ab12cd34
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = uuid.uuid4().hex[:8]
    return f"{prefix}_{ts}_{short}"


@dataclass(frozen=True)
class RunContext:
    """
    Small container to keep consistent metadata across events.
    """
    run_id: str
    project: str
    model: str


def append_jsonl(
    record: Dict[str, Any],
    log_path: Path,
) -> None:
    """
    Append a single JSON object as one line into a .jsonl file.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_event(
    *,
    event_type: str,
    context: RunContext,
    payload: Optional[Dict[str, Any]] = None,
    outputs_dir: Path = DEFAULT_OUTPUT_DIR,
    log_filename: str = DEFAULT_LOG_FILE,
    extra: Optional[Dict[str, Any]] = None,
    step: Optional[str] = None,
) -> Path:
    """
    Log an event (append-only) to outputs/<log_filename>.

    event_type: e.g. "sentiment_request", "sentiment_result", "error"
    payload: core event data (prompt, input text, results, etc.)
    extra: optional additional metadata (timings, parameters)
    step: optional logical step name or sequence
    """
    ensure_output_dir(outputs_dir)
    log_path = get_default_log_path(outputs_dir, log_filename)

    record: Dict[str, Any] = {
        "timestamp_utc": _utc_now_iso(),
        "event_type": event_type,
        "run_id": context.run_id,
        "project": context.project,
        "model": context.model,
    }

    if step:
        record["step"] = step
    if payload:
        record["payload"] = redact_secrets(_safe_json(payload))
    if extra:
        record["extra"] = redact_secrets(_safe_json(extra))

    append_jsonl(record, log_path)
    return log_path


class Timer:
    """
    Simple timing context manager.

    Usage:
        with Timer() as t:
            ...
        elapsed = t.elapsed_ms
    """

    def __init__(self) -> None:
        self._start = 0.0
        self._end = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        end = self._end if self._end else time.perf_counter()
        return (end - self._start) * 1000.0
