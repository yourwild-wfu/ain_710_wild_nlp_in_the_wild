"""
run_embedding_summary.py

Script to summarize embedding results and save to a JSON file.
"""

from src.lab.embeddings import (
    load_embedding_results,
    summarize_embeddings,
    write_embedding_summary_json,
)
from src.lab.logging_utils import DEFAULT_OUTPUT_DIR

def main():
    print("Generating embedding summary...")
    
    # 1. Load results
    results = load_embedding_results()
    if not results:
        print("No results found to summarize. Run run_embeddings.py first.")
        return

    # 2. Generate summary
    summary = summarize_embeddings(results)
    
    # 3. Save to JSON
    out_path = DEFAULT_OUTPUT_DIR / "embeddings_summary.json"
    write_embedding_summary_json(out_path, summary)
    
    print(f"\nSummary generated for {summary['total_processed']} items.")
    print(f"Mean Magnitude: {summary['mean_magnitude']}")
    print(f"Unique Entities: {summary['unique_entities_count']}")
    print(f"Total Tokens: {summary['usage_summary']['total_tokens']}")
    
    print("\n" + "="*60)
    print("NARRATIVE OVERVIEW")
    print("="*60)
    print(summary.get("narrative_overview", "No narrative generated."))
    print("="*60)

    print(f"\nResults saved to: {out_path}")

if __name__ == "__main__":
    main()
