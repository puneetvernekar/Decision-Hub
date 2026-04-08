"""
Lightweight traceability logger for the war-room pipeline.

Every significant step (phase start, tool invocation, LLM call, verdict)
is recorded with a monotonic timestamp, printed to the console, and
persisted to a log file (``war_room_trace.log`` by default).

The full trace is also available programmatically via ``get_trace()``.

Usage
─────
    from war_room.trace import trace, get_trace, reset_trace

    trace("phase_1", "start", "Beginning independent analysis")
    trace("pm_agent", "tool_call", "aggregate_metrics → health: critical")
    ...
    all_steps = get_trace()   # list[dict]
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

_trace_log: list[dict[str, Any]] = []
_start_time: float = 0.0
_log_file: str = ""


def _get_log_path() -> str:
    """Resolve the log file path from env or default."""
    return os.environ.get("LOG_FILE_PATH", "war_room_trace.log")


def reset_trace() -> None:
    """Clear the trace log, reset the clock, and start a fresh log file."""
    global _trace_log, _start_time, _log_file
    _trace_log = []
    _start_time = time.monotonic()
    _log_file = _get_log_path()

    # Write header to log file
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with open(_log_file, "w", encoding="utf-8") as f:
        f.write(f"# War-Room Execution Trace — {timestamp}\n")
        f.write(f"# {'Step':>4}  {'Elapsed':>8}  {'Source':20s}  {'Event':12s}  Detail\n")
        f.write(f"# {'─'*4}  {'─'*8}  {'─'*20}  {'─'*12}  {'─'*40}\n")


def trace(source: str, event: str, detail: str) -> None:
    """Record, print, and log a single trace step.

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

    line = f"  [{step:>3}] +{elapsed:6.2f}s  {source:20s}  {event:12s}  {detail}"
    print(line)

    # Append to log file
    try:
        with open(_log_file or _get_log_path(), "a", encoding="utf-8") as f:
            f.write(f"  [{step:>3}] +{elapsed:6.2f}s  {source:20s}  {event:12s}  {detail}\n")
    except OSError:
        pass  # don't let logging failures break the pipeline


def get_trace() -> list[dict[str, Any]]:
    """Return a copy of the full trace log."""
    return list(_trace_log)
