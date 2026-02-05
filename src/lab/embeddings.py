"""
embeddings.py

Lab module for generating embeddings from text strings.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.lab.client import get_client, load_config
from src.lab.logging_utils import DEFAULT_OUTPUT_DIR, RunContext, Timer, log_event, new_run_id


@dataclass(frozen=True)
class EmbeddingRequest:
    """
    Request parameters for generating embeddings.
    """
    text: str
    model: Optional[str] = None


@dataclass(frozen=True)
class EmbeddingResponse:
    """
    Response from the embedding generation.
    """
    text: str
    embedding: List[float]
    magnitude: float
    entities: List[str]
    model: str
    usage: dict
    elapsed_ms: float


def generate_embedding(req: EmbeddingRequest, ctx: RunContext) -> EmbeddingResponse:
    """
    Generates an embedding for the given text and extracts entities.

    Args:
        req: The EmbeddingRequest object.
        ctx: The current RunContext.

    Returns:
        An EmbeddingResponse object.
    """
    client = get_client()
    config = load_config()
    model = req.model or config.embedding_model

    log_event(
        event_type="embedding_request",
        context=ctx,
        payload={"text": req.text, "model": model},
        step="generate_embedding"
    )

    with Timer() as timer:
        # 1. Generate Embedding
        response = client.embeddings.create(
            input=req.text,
            model=model
        )
        embedding_data = response.data[0].embedding
        usage = response.usage.model_dump()

        # 2. Extract Entities using LLM
        # Using the project's default model for extraction
        ner_response = client.chat.completions.create(
            model=config.model,
            messages=[
                {
                    "role": "system",
                    "content": config.entity_extraction_prompt
                },
                {"role": "user", "content": f"Extract entities from: '{req.text}'"}
            ]
        )
        entities_raw = ner_response.choices[0].message.content or ""
        if entities_raw.lower() == "none":
            entities = []
        else:
            entities = [e.strip() for e in entities_raw.split(",") if e.strip()]

    # Calculate L2 Norm (Magnitude)
    magnitude = math.sqrt(sum(x**2 for x in embedding_data))

    res = EmbeddingResponse(
        text=req.text,
        embedding=embedding_data,
        magnitude=magnitude,
        entities=entities,
        model=model,
        usage=usage,
        elapsed_ms=timer.elapsed_ms
    )

    log_event(
        event_type="embedding_response",
        context=ctx,
        payload={
            "text": res.text,
            "entities": res.entities,
            "magnitude": res.magnitude,
            "embedding_length": len(res.embedding),
            "model": res.model,
            "usage": res.usage,
            "elapsed_ms": res.elapsed_ms
        },
        step="generate_embedding"
    )

    return res


def build_default_context() -> RunContext:
    """
    Helper to create a default RunContext for embeddings.
    """
    cfg = load_config()
    return RunContext(
        run_id=new_run_id("embed"),
        project=cfg.project_name,
        model=cfg.embedding_model
    )


def load_embedding_results(path: Path = DEFAULT_OUTPUT_DIR / "embedding_results.jsonl") -> List[Dict[str, Any]]:
    """
    Loads embedding results from a JSONL file.
    """
    if not path.exists():
        return []
    
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def generate_embedding_narrative(summary: Dict[str, Any]) -> str:
    """
    Generates a narrative overview of the embedding results using GPT-4o.
    Provides insights on mean magnitude, entities, and what they mean for the class.
    """
    client = get_client()
    config = load_config()

    prompt = config.narrative_prompt.format(
        total_processed=summary['total_processed'],
        mean_magnitude=summary['mean_magnitude'],
        unique_entities_count=summary['unique_entities_count'],
        entities_list=", ".join(summary['unique_entities']),
        model=summary['model']
    )

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant providing educational insights into NLP metrics."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content or "Narrative generation failed."
    except Exception as e:
        return f"Could not generate narrative: {e}"


def summarize_embeddings(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarizes embedding results and adds a narrative overview.
    """
    if not results:
        return {}

    n = len(results)
    magnitudes = [r.get("magnitude", 0.0) for r in results]
    latencies = [r.get("elapsed_ms", 0.0) for r in results]
    
    all_entities = []
    for r in results:
        all_entities.extend(r.get("entities", []))
    unique_entities = sorted(list(set(all_entities)))
    
    total_prompt_tokens = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in results)
    total_completion_tokens = sum(r.get("usage", {}).get("completion_tokens", 0) for r in results)
    total_tokens = sum(r.get("usage", {}).get("total_tokens", 0) for r in results)

    summary = {
        "total_processed": n,
        "mean_magnitude": round(sum(magnitudes) / n, 6) if n else 0,
        "mean_latency_ms": round(sum(latencies) / n, 2) if n else 0,
        "unique_entities_count": len(unique_entities),
        "unique_entities": unique_entities,
        "usage_summary": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens
        },
        "model": results[0].get("model"),
        "project": results[0].get("run_id").split('_')[0] if "_" in results[0].get("run_id", "") else "unknown"
    }
    
    # Add narrative overview
    summary["narrative_overview"] = generate_embedding_narrative(summary)
    
    return summary


def write_embedding_summary_json(path: Path, summary: Dict[str, Any]) -> None:
    """
    Writes embedding summary to a JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Quick smoke test
    try:
        context = build_default_context()
        request = EmbeddingRequest(text="Hello world!")
        result = generate_embedding(request, context)
        print(f"Text: {result.text}")
        print(f"Embedding length: {len(result.embedding)}")
        print(f"First 5 values: {result.embedding[:5]}")
        print(f"Elapsed: {result.elapsed_ms:.2f}ms")
    except Exception as e:
        print(f"Error: {e}")
