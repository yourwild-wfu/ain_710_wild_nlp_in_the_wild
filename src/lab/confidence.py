"""
confidence.py

Compute stability + confidence summaries from sentiment batch results.

Primary input:
- <repo>/outputs/sentiment_results.jsonl (JSON Lines), written by sentiment_batch.py

Metrics per unique text:
- label_mode (most frequent label)
- label_flip_rate (fraction of runs not equal to mode)
- confidence_mean / std / min / max
- low_confidence_flag (mean confidence below threshold)
- unstable_flag (any label flips)

This is intentionally simple, explainable, and "business friendly".
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.lab.logging_utils import REPO_ROOT


DEFAULT_RESULTS_PATH = REPO_ROOT / "outputs" / "sentiment_results.jsonl"


@dataclass(frozen=True)
class ConfidenceConfig:
    """
    Configuration for confidence and stability summarization.

    Attributes:
        low_conf_threshold: Threshold below which confidence is flagged as low.
        group_by: The field name to group results by ('text' or 'item_index').
    """
    low_conf_threshold: float = 0.60
    group_by: str = "text"  # "text" or "item_index"


def load_results_jsonl(path: Path = DEFAULT_RESULTS_PATH) -> List[Dict[str, Any]]:
    """
    Load sentiment results from a JSONL file into a list of dictionaries.

    Args:
        path: Path to the JSONL file.

    Returns:
        A list of dictionaries representing each result row.

    Raises:
        FileNotFoundError: If the results file does not exist.
        ValueError: If the file contains invalid JSON or no rows.
    """
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i} in {path}: {e}") from e

    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _mean(xs: List[float]) -> float:
    """Calculates the arithmetic mean."""
    return sum(xs) / len(xs) if xs else float("nan")


def _std_pop(xs: List[float]) -> float:
    """
    Calculates the population standard deviation.
    """
    if not xs:
        return float("nan")
    mu = _mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / len(xs)
    return math.sqrt(var)


def _mode_label(labels: List[str]) -> Tuple[str, int]:
    """
    Finds the most frequent label.

    Args:
        labels: A list of label strings.

    Returns:
        A tuple of (mode_label, count). Ties are broken alphabetically.
    """
    counts: Dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1

    mode_label = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return mode_label, counts[mode_label]


def summarize_confidence(
    *,
    results: Optional[List[Dict[str, Any]]] = None,
    results_path: Path = DEFAULT_RESULTS_PATH,
    config: Optional[ConfidenceConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Summarize stability and confidence metrics per group.

    Args:
        results: Optional list of result dicts. If None, loaded from results_path.
        results_path: Path to load results from if results is None.
        config: Configuration for summarization.

    Returns:
        A list of summary dictionaries, one per group.

    Raises:
        ValueError: If configuration or data is invalid.
    """
    cfg = config or ConfidenceConfig()
    rows = results if results is not None else load_results_jsonl(results_path)

    if cfg.group_by not in {"text", "item_index"}:
        raise ValueError("ConfidenceConfig.group_by must be 'text' or 'item_index'")

    # Group rows
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        key = r.get(cfg.group_by)
        groups.setdefault(key, []).append(r)

    summaries: List[Dict[str, Any]] = []

    for key, items in groups.items():
        labels = [str(x.get("label")) for x in items]
        confidences = [float(x.get("confidence")) for x in items]

        mode_label, mode_count = _mode_label(labels)
        n = len(items)

        flip_count = sum(1 for lab in labels if lab != mode_label)
        flip_rate = flip_count / n if n else 0.0

        conf_mean = _mean(confidences)

        # Pull common metadata from the first row
        first = items[0]

        summary = {
            "group_by": cfg.group_by,
            "group_key": key,
            "n_runs": n,
            "label_mode": mode_label,
            "label_mode_count": mode_count,
            "label_flip_count": flip_count,
            "label_flip_rate": round(flip_rate, 4),
            "confidence_mean": round(conf_mean, 4),
            "confidence_std": round(_std_pop(confidences), 4),
            "confidence_min": round(min(confidences), 4),
            "confidence_max": round(max(confidences), 4),
            "low_confidence_flag": conf_mean < cfg.low_conf_threshold,
            "unstable_flag": flip_rate > 0.0,
            "project": first.get("project"),
            "model": first.get("model"),
        }

        summaries.append(summary)

    # Stable ordering (useful for diffs / reproducibility)
    summaries.sort(key=lambda s: str(s["group_key"]))
    return summaries


def write_confidence_summary_json(path: Path, summaries: List[Dict[str, Any]]) -> None:
    """
    Writes summary records to a JSON file (pretty-printed).

    Args:
        path: Destination path for the JSON file.
        summaries: The list of summary dictionaries to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Example usage:
    # try:
    #     summaries = summarize_confidence()
    #     for s in summaries:
    #         print(f"Key: {s['group_key']}, Mean Conf: {s['confidence_mean']}")
    # except Exception as e:
    #     print(f"Error generating summary: {e}")
    print("Confidence module loaded. Use summarize_confidence() to analyze results.")
