from src.lab.client import load_config
from src.lab.logging_utils import RunContext, new_run_id, log_event, Timer

cfg = load_config()
ctx = RunContext(run_id=new_run_id(), project=cfg.project_name, model=cfg.model)

log_event(event_type="startup", context=ctx, payload={"message": "lab started"}, step="init")

with Timer() as t:
    # pretend work
    x = sum(range(10_000))

log_event(
    event_type="timing_test",
    context=ctx,
    payload={"note": "timing sample"},
    extra={"elapsed_ms": t.elapsed_ms},
    step="processing"
)
