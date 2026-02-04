from src.lab.confidence import summarize_confidence, ConfidenceConfig, REPO_ROOT, write_confidence_summary_json

summaries = summarize_confidence(config=ConfidenceConfig(low_conf_threshold=0.70, group_by="text"))

for s in summaries:
    print(s)

out_path = REPO_ROOT / "outputs" / "confidence_summary.json"
write_confidence_summary_json(out_path, summaries)
print(f"\nWrote: {out_path}")
