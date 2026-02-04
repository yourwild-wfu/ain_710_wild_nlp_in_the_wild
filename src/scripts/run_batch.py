"""
run_batch.py

Script to run sentiment classification on a batch of texts.
"""

from src.lab.sentiment import build_default_context
from src.lab.sentiment_batch import BatchConfig, run_sentiment_batch


def main() -> None:
    """
    Main entry point for running the sentiment batch process.
    """
    ctx = build_default_context()

    # Define some example texts to classify
    texts = [
        "I love the design, but the setup was frustrating.",
        "This was a complete waste of time.",
        "Absolutely fantastic experience — would recommend.",
        "Yeah, great… just what I needed (eye roll).",
    ]

    print(f"Starting batch sentiment classification for {len(texts)} texts...")
    
    results = run_sentiment_batch(
        ctx=ctx,
        texts=texts,
        config=BatchConfig(n_runs_per_text=1, temperature=0.2, write_results_jsonl=True),
    )

    print(f"Batch processing complete. Records generated: {len(results)}")
    if results:
        print("\nFirst result sample:")
        print(f"  Text: {results[0].get('text')}")
        print(f"  Label: {results[0].get('label')}, Confidence: {results[0].get('confidence')}")


if __name__ == "__main__":
    main()
