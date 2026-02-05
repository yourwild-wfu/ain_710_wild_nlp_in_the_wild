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
    """
    Request parameters for sentiment classification.

    Attributes:
        text: The text to classify.
        temperature: Sampling temperature (0.0 to 2.0). If None, uses SENTIMENT_TEMPERATURE from .env.
        max_output_tokens: Maximum tokens in the model response. If None, uses SENTIMENT_MAX_OUTPUT_TOKENS from .env.
        include_logprobs: Whether to request logprobs from the API. If None, uses SENTIMENT_INCLUDE_LOGPROBS from .env.
    """
    text: str
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    include_logprobs: Optional[bool] = None


@dataclass(frozen=True)
class SentimentResponse:
    """
    Parsed results from sentiment classification.

    Attributes:
        label: The classified sentiment (positive/neutral/negative).
        confidence: The model's reported confidence (0.0 to 1.0).
        rationale: Brief explanation for the classification.
        response_id: Unique ID from the OpenAI response.
        elapsed_ms: Time taken for the API call in milliseconds.
        raw_output_text: The raw JSON string returned by the model.
        logprobs_included: Boolean indicating if logprobs were returned.
    """
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
    Parse the model output text into a dict. 

    Args:
        output_text: The JSON string returned by the model.

    Returns:
        A dictionary containing the parsed sentiment data.

    Raises:
        ValueError: If the output text is not valid JSON.
    """
    try:
        return json.loads(output_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model output was not valid JSON: {e}\nRaw: {output_text}") from e


def classify_sentiment(req: SentimentRequest, ctx: RunContext) -> SentimentResponse:
    """
    Run sentiment classification for a single text input and log the event.

    Args:
        req: The sentiment request parameters.
        ctx: The current run context for logging.

    Returns:
        A SentimentResponse object containing the results.

    Raises:
        Exception: Re-raises any exceptions encountered during the API call.
    """
    cfg = load_config()
    client = get_client()

    # Resolve effective parameters from request or environment defaults
    eff_temperature = req.temperature if req.temperature is not None else cfg.sentiment_temperature
    eff_max_tokens = req.max_output_tokens if req.max_output_tokens is not None else cfg.sentiment_max_output_tokens
    eff_include_logprobs = req.include_logprobs if req.include_logprobs is not None else cfg.sentiment_include_logprobs

    # Build "include" fields for extra response data (e.g., logprobs)
    include = ["message.output_text.logprobs"] if eff_include_logprobs else None

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
                    {"role": "system", "content": cfg.sentiment_prompt},
                    {"role": "user", "content": req.text},
                ],
                text={"format": SENTIMENT_SCHEMA},
                temperature=eff_temperature,
                max_output_tokens=eff_max_tokens,
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
    Convenience helper for scripts/notebooks to create a standard RunContext.

    Returns:
        A RunContext initialized with a new run ID and loaded configuration.
    """
    from src.lab.logging_utils import new_run_id

    cfg = load_config()
    return RunContext(run_id=new_run_id(), project=cfg.project_name, model=cfg.model)


if __name__ == "__main__":
    # Example usage:
    # try:
    #     context = build_default_context()
    #     request = SentimentRequest(text="I love this new feature!")
    #     response = classify_sentiment(request, context)
    #     print(f"Label: {response.label}, Confidence: {response.confidence}")
    #     print(f"Rationale: {response.rationale}")
    # except Exception as err:
    #     print(f"Error: {err}")
    print("Sentiment module loaded. Use classify_sentiment() to process text.")
