"""
Lightweight traceability logger for the war-room pipeline.

Every significant step (phase start, tool invocation, LLM call, verdict)
is recorded with a monotonic timestamp and printed to the console.  The
full trace is also available programmatically via ``get_trace()``.

Usage
─────
    from war_room.trace import trace, get_trace, reset_trace

    trace("phase_1", "start", "Beginning independent analysis")
    trace("pm_agent", "tool_call", "aggregate_metrics → health: critical")
    ...
    all_steps = get_trace()   # list[dict]
"""

from __future__ import annotations

import time
from typing import Any

_trace_log: list[dict[str, Any]] = []
_start_time: float = 0.0


def reset_trace() -> None:
    """Clear the trace log and reset the clock."""
    global _trace_log, _start_time
    _trace_log = []
    _start_time = time.monotonic()


def trace(source: str, event: str, detail: str) -> None:
    """Record and print a single trace step.

    Parameters
    ----------
    source : str
        Who is emitting the trace (e.g. ``"orchestrator"``, ``"PM Agent"``).
    event : str
        Event type (e.g. ``"phase_start"``, ``"tool_call"``, ``"llm_call"``,
        ``"verdict"``, ``"info"``).
    detail : str
        Human-readable description of what happened.
    """
    if _start_time == 0.0:
        reset_trace()

    elapsed = time.monotonic() - _start_time
    step = len(_trace_log) + 1
    entry = {
        "step": step,
        "elapsed_s": round(elapsed, 3),
        "source": source,
        "event": event,
        "detail": detail,
    }
    _trace_log.append(entry)
    print(f"  [{step:>3}] +{elapsed:6.2f}s  {source:20s}  {event:12s}  {detail}")


def get_trace() -> list[dict[str, Any]]:
    """Return a copy of the full trace log."""
    return list(_trace_log)
