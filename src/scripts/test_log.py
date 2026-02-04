"""
test_log.py

Verifies that the logging utilities are working as expected.
"""

from src.lab.client import load_config
from src.lab.logging_utils import RunContext, new_run_id, log_event, Timer


def main() -> None:
    """
    Demonstrates and verifies the logging of events and timing.
    """
    print("Testing logging utilities...")
    
    cfg = load_config()
    ctx = RunContext(run_id=new_run_id(), project=cfg.project_name, model=cfg.model)

    print(f"Logging startup event for run: {ctx.run_id}")
    log_event(event_type="startup", context=ctx, payload={"message": "lab started"}, step="init")

    print("Running timed operation...")
    with Timer() as t:
        # pretend work
        _ = sum(range(10_000))

    print(f"Operation took {t.elapsed_ms:.2f}ms. Logging result...")
    log_event(
        event_type="timing_test",
        context=ctx,
        payload={"note": "timing sample"},
        extra={"elapsed_ms": t.elapsed_ms},
        step="processing"
    )
    
    print("\nLogging test complete. Check 'outputs/runs.jsonl' for results.")


if __name__ == "__main__":
    main()
