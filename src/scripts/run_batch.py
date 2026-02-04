from src.lab.sentiment import build_default_context
from src.lab.sentiment_batch import BatchConfig, run_sentiment_batch

ctx = build_default_context()

results = run_sentiment_batch(
    ctx=ctx,
    # If you created data/inputs.jsonl, you can omit texts=
    texts=[
        "I love the design, but the setup was frustrating.",
        "This was a complete waste of time.",
        "Absolutely fantastic experience — would recommend.",
        "Yeah, great… just what I needed (eye roll).",
    ],
    config=BatchConfig(n_runs_per_text=1, temperature=0.2, write_results_jsonl=True),
)

print(f"Batch records: {len(results)}")
print(results[0])
