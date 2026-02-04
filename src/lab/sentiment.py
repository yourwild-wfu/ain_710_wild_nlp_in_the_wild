"""
sentiment.py

Sentiment classification lab using the OpenAI Responses API with Structured Outputs.

What this module does:
- Calls the model to classify sentiment (positive/neutral/negative)
- Enforces a strict JSON schema for the response
- Logs request + response events to outputs/runs.jsonl
- Optionally requests token logprobs for "under the hood" analysis
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.lab.client import load_config, get_client
from src.lab.logging_utils import RunContext, Timer, log_event


# ---- Prompting ----

SYSTEM_INSTRUCTIONS = (
    "You are an NLP evaluator. Your job is to classify the sentiment of the user's text.\n"
    "Return ONLY a JSON object that matches the provided schema.\n"
    "Rules:\n"
    "- label must be one of: positive, neutral, negative\n"
    "- confidence must be a number from 0.0 to 1.0\n"
    "- rationale must be short and grounded in the text (no speculation)\n"
)

SENTIMENT_SCHEMA = {
    "type": "json_schema",
    "name": "sentiment_result",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": ["positive", "neutral", "negative"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "rationale": {"type": "string", "maxLength": 280},
        },
        "required": ["label", "confidence", "rationale"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class SentimentRequest:
    text: str
    temperature: float = 0.2
    max_output_tokens: int = 200
    include_logprobs: bool = True


@dataclass(frozen=True)
class SentimentResponse:
    label: str
    confidence: float
    rationale: str
    response_id: str
    elapsed_ms: float
    # Raw structured output text (JSON) for debugging/audit
    raw_output_text: str
    # Optional: the SDK may include logprobs data depending on availability/settings
    logprobs_included: bool


def _parse_structured_json(output_text: str) -> Dict[str, Any]:
    """
    Parse the model output text into a dict. With Structured Outputs + strict schema,
    this should always succeed unless something upstream fails.
    """
    try:
        return json.loads(output_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model output was not valid JSON: {e}\nRaw: {output_text}") from e


def classify_sentiment(req: SentimentRequest, ctx: RunContext) -> SentimentResponse:
    """
    Run sentiment classification for a single text input and log:
    - sentiment_request
    - sentiment_result (or error)
    """
    cfg = load_config()
    client = get_client()

    # Build "include" fields for extra response data (e.g., logprobs)
    include = ["message.output_text.logprobs"] if req.include_logprobs else None

    # Log the request (do NOT log secrets; logging_utils redacts patterns)
    log_event(
        event_type="sentiment_request",
        context=ctx,
        step="sentiment_request",
        payload={
            "text": req.text,
            "temperature": req.temperature,
            "max_output_tokens": req.max_output_tokens,
            "include_logprobs": req.include_logprobs,
        },
    )

    try:
        with Timer() as t:
            response = client.responses.create(
                model=cfg.model,
                input=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": req.text},
                ],
                text={"format": SENTIMENT_SCHEMA},
                temperature=req.temperature,
                max_output_tokens=req.max_output_tokens,
                include=include,
            )

        # The OpenAI SDK provides a convenience string for "final text output"
        output_text = getattr(response, "output_text", "") or ""
        parsed = _parse_structured_json(output_text)

        label = parsed["label"]
        confidence = float(parsed["confidence"])
        rationale = parsed["rationale"]

        # Determine whether logprobs were requested (and likely present)
        logprobs_included = bool(include)

        result = SentimentResponse(
            label=label,
            confidence=confidence,
            rationale=rationale,
            response_id=response.id,
            elapsed_ms=t.elapsed_ms,
            raw_output_text=output_text,
            logprobs_included=logprobs_included,
        )

        # Log the result
        log_event(
            event_type="sentiment_result",
            context=ctx,
            step="sentiment_result",
            payload={
                "label": result.label,
                "confidence": result.confidence,
                "rationale": result.rationale,
            },
            extra={
                "elapsed_ms": result.elapsed_ms,
                "response_id": result.response_id,
                "logprobs_included": result.logprobs_included,
            },
        )

        return result

    except Exception as e:
        # Log the error for later debugging/analysis
        log_event(
            event_type="error",
            context=ctx,
            step="sentiment_error",
            payload={
                "message": "Sentiment classification failed",
                "error_type": type(e).__name__,
                "error": str(e),
            },
        )
        raise


def build_default_context() -> RunContext:
    """
    Convenience helper for scripts/notebooks.
    """
    from src.lab.logging_utils import new_run_id

    cfg = load_config()
    return RunContext(run_id=new_run_id(), project=cfg.project_name, model=cfg.model)
