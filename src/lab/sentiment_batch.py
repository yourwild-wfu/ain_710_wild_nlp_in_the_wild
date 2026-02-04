"""
sentiment_batch.py

Batch runner for sentiment classification.

Features:
- Run sentiment on many inputs
- Optional repeated runs per input (useful for stability/variance experiments)
- Logs batch start/end + per-item results (via sentiment.classify_sentiment)
- Saves a compact results JSONL to outputs/ (optional)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.lab.logging_utils import REPO_ROOT, RunContext, Timer, log_event
from src.lab.sentiment import SentimentRequest, SentimentResponse, classify_sentiment


DEFAULT_INPUTS_JSONL = Path("data/inputs.jsonl")


@dataclass(frozen=True)
class BatchConfig:
    n_runs_per_text: int = 1  # repeat each text N times
    temperature: float = 0.2
    max_output_tokens: int = 200
    include_logprobs: bool = True
    # If set, write a compact results jsonl file under outputs/
    write_results_jsonl: bool = True
    results_filename: str = "sentiment_results.jsonl"


def load_texts_from_jsonl(path: Path = DEFAULT_INPUTS_JSONL) -> List[str]:
    """
    Load texts from a JSONL file.

    Expected format per line (either is fine):
      {"text": "..."}
    or
      {"id": "...", "text": "..."}
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Inputs file not found: {path}. Create it or pass texts directly."
        )

    texts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = str(obj.get("text", "")).strip()
            if not text:
                raise ValueError(f"Line {i} missing non-empty 'text' field in {path}")
            texts.append(text)

    if not texts:
        raise ValueError(f"No valid texts found in {path}")
    return texts


def _append_results_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_sentiment_batch(
    *,
    ctx: RunContext,
    texts: Optional[List[str]] = None,
    inputs_jsonl: Optional[Path] = None,
    config: Optional[BatchConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Run sentiment classification over a set of texts.

    Returns: list of dict records (easy to dump / convert to DataFrame).
    Each record includes:
      - item_index, run_index, text
      - label, confidence, rationale
      - elapsed_ms, response_id
    """
    cfg = config or BatchConfig()

    if texts is None:
        path = inputs_jsonl or DEFAULT_INPUTS_JSONL
        texts = load_texts_from_jsonl(path)

    if not isinstance(texts, list) or not texts:
        raise ValueError("texts must be a non-empty list of strings")

    # Log batch start
    log_event(
        event_type="batch_start",
        context=ctx,
        step="batch_init",
        payload={
            "num_texts": len(texts),
            "n_runs_per_text": cfg.n_runs_per_text,
            "temperature": cfg.temperature,
            "max_output_tokens": cfg.max_output_tokens,
            "include_logprobs": cfg.include_logprobs,
        },
    )

    results: List[Dict[str, Any]] = []
    results_path = REPO_ROOT / "outputs" / cfg.results_filename

    with Timer() as batch_timer:
        for item_idx, text in enumerate(texts):
            for run_idx in range(cfg.n_runs_per_text):
                req = SentimentRequest(
                    text=text,
                    temperature=cfg.temperature,
                    max_output_tokens=cfg.max_output_tokens,
                    include_logprobs=cfg.include_logprobs,
                )

                # classify_sentiment already logs request/result/error
                resp: SentimentResponse = classify_sentiment(req, ctx)

                record = {
                    "run_id": ctx.run_id,
                    "project": ctx.project,
                    "model": ctx.model,
                    "item_index": item_idx,
                    "run_index": run_idx,
                    "text": text,
                    "label": resp.label,
                    "confidence": resp.confidence,
                    "rationale": resp.rationale,
                    "elapsed_ms": resp.elapsed_ms,
                    "response_id": resp.response_id,
                    "logprobs_included": resp.logprobs_included,
                }
                results.append(record)

                if cfg.write_results_jsonl:
                    _append_results_jsonl(results_path, record)

    # Log batch end
    log_event(
        event_type="batch_end",
        context=ctx,
        step="batch_complete",
        payload={
            "num_texts": len(texts),
            "n_runs_per_text": cfg.n_runs_per_text,
            "num_records": len(results),
        },
        extra={"elapsed_ms": batch_timer.elapsed_ms},
    )

    return results
