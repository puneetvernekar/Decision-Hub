"""Simple trace logger -- records steps in memory, console, and decision-hub.log."""

import json
import time
from dataclasses import asdict
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parent.parent / "decision-hub.log"

log_list = []
start_time = 0.0


def reset_trace():
    global log_list, start_time
    log_list = []
    start_time = time.monotonic()
    # clear the log file at the start of each session
    LOG_FILE.write_text("", encoding="utf-8")


def _write(line):
    """Append a line to the log file."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def trace(source, event, detail):
    if not start_time:
        reset_trace()

    elapsed = time.monotonic() - start_time
    step = len(log_list) + 1
    log_list.append({
        "step": step, "elapsed_s": round(elapsed, 3),
        "source": source, "event": event, "detail": detail,
    })
    msg = f"  [{step:>3}] +{elapsed:6.2f}s  {source:20s}  {event:12s}  {detail}"
    print(msg)
    _write(msg)


def log_verdict(label, verdict):
    """Log a single agent verdict as JSON."""
    from .models import AgentVerdict
    data = asdict(verdict)
    data["decision"] = verdict.decision.value
    _write(f"\n--- {label} ---")
    _write(json.dumps(data, indent=2))


def log_outcome(outcome):
    """Log the final WarRoomOutcome as JSON."""
    from .models import WarRoomOutcome
    data = {
        "final_decision": outcome.final_decision.value,
        "decision_rationale": outcome.decision_rationale,
        "confidence_score": outcome.confidence_score,
        "confidence_drivers": outcome.confidence_drivers,
        "action_plan": outcome.action_plan,
        "risks_and_mitigations": outcome.risks_and_mitigations,
        "communication_plan": outcome.communication_plan,
        "follow_up_monitoring": outcome.follow_up_monitoring,
        "dissenting_opinions": outcome.dissenting_opinions,
    }
    _write(f"\n{'=' * 60}")
    _write("FINAL WAR-ROOM OUTCOME")
    _write("=" * 60)
    _write(json.dumps(data, indent=2))


def get_trace():
    """Returns a copy so callers can't mess with the internal list."""
    return list(log_list)
