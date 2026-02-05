"""
run_embeddings.py

Script to demonstrate string-to-embedding conversion.
Saves results to a JSONL file in the outputs directory.
"""

import json
import math
from pathlib import Path
from typing import List

from src.lab.embeddings import (
    EmbeddingRequest,
    build_default_context,
    generate_embedding,
    load_config,
)
from src.lab.logging_utils import DEFAULT_OUTPUT_DIR, append_jsonl

OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "embedding_results.jsonl"


def main():
    # 1. Load config and inputs
    config = load_config()
    texts = config.embedding_texts

    # 2. Build context
    ctx = build_default_context()
    print(f"Starting embedding run: {ctx.run_id}")

    results = []

    # 3. Generate embeddings
    for text in texts:
        print(f"Processing: '{text[:30]}...'")
        req = EmbeddingRequest(text=text)
        res = generate_embedding(req, ctx)
        
        # Prepare record for JSONL
        record = {
            "run_id": ctx.run_id,
            "text": res.text,
            "entities": res.entities,
            "magnitude": res.magnitude,
            "model": res.model,
            "embedding_length": len(res.embedding),
            "embedding_sample": res.embedding[:5],  # Just a sample for readability
            "full_embedding": res.embedding,        # The full vector
            "usage": res.usage,
            "elapsed_ms": res.elapsed_ms
        }
        
        # 4. Save to JSONL
        append_jsonl(record, OUTPUT_FILE)
        results.append(res)  # Store the full response object for pretty printing later

    print(f"\nResults saved to {OUTPUT_FILE}")
    print(f"Processed {len(results)} strings.")
    
    # 5. Show detailed class demonstration output
    print("\n" + "="*60)
    print("EMBEDDING CLASS DEMONSTRATION")
    print("="*60)
    
    for res in results:
        print(f"\n[TEXT]: \"{res.text}\"")
        print(f"[ENTITIES]: {', '.join(res.entities) if res.entities else 'None'}")
        
        # Vector visualization: Head and Tail
        head = res.embedding[:3]
        tail = res.embedding[-3:]
        vector_str = f"[{', '.join(f'{x:.4f}' for x in head)}, ..., {', '.join(f'{x:.4f}' for x in tail)}]"
        
        print(f"[VECTOR]: {vector_str} (Dim: {len(res.embedding)})")
        
        # Mathematical Proof
        is_normalized = math.isclose(res.magnitude, 1.0, rel_tol=1e-5)
        print(f"[MAGNITUDE]: {res.magnitude:.6f} (Normalized: {is_normalized})")
        
        # Performance/Usage
        print(f"[METRICS]: {res.usage['total_tokens']} tokens | {res.elapsed_ms:.2f}ms")
        print("-" * 40)


if __name__ == "__main__":
    main()
