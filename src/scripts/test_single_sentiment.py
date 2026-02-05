"""
test_single_sentiment.py

Basic smoke test for sentiment classification.
"""

from src.lab.sentiment import SentimentRequest, classify_sentiment, build_default_context
from src.lab.client import load_config


def main() -> None:
    """
    Runs a single sentiment classification test.
    """
    config = load_config()
    ctx = build_default_context()
    req = SentimentRequest(text=config.test_text)
    
    print(f"Testing sentiment classification for: '{req.text}'")
    
    try:
        result = classify_sentiment(req, ctx)
        print("\nSuccess!")
        print(f"Label: {result.label}")
        print(f"Confidence: {result.confidence}")
        print(f"Rationale: {result.rationale}")
        print(f"Elapsed: {result.elapsed_ms:.2f}ms")
    except Exception as e:
        print(f"\nFailed: {e}")


if __name__ == "__main__":
    main()
