"""
run_confidence.py

Script to summarize confidence and stability from sentiment results.
"""

from src.lab.confidence import summarize_confidence, ConfidenceConfig, REPO_ROOT, write_confidence_summary_json


def main() -> None:
    """
    Main entry point for generating the confidence summary.
    """
    print("Generating confidence summary from results...")
    
    config = ConfidenceConfig(low_conf_threshold=0.70, group_by="text")
    summaries = summarize_confidence(config=config)

    print(f"Summary generated for {len(summaries)} unique items.")
    
    for s in summaries[:3]:  # Print first 3
        print(f" - {s['group_key'][:50]}... -> Mode: {s['label_mode']}, Mean Conf: {s['confidence_mean']}")

    out_path = REPO_ROOT / "outputs" / "confidence_summary.json"
    write_confidence_summary_json(out_path, summaries)
    print(f"\nFull summary written to: {out_path}")


if __name__ == "__main__":
    main()
