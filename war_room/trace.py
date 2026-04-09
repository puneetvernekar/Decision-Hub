"""Simple trace logger — records steps in memory and prints to console."""

import time

_log = []
_t0 = 0.0


def reset_trace():
    global _log, _t0
    _log = []
    _t0 = time.monotonic()


def trace(source, event, detail):
    if not _t0:
        reset_trace()

    elapsed = time.monotonic() - _t0
    step = len(_log) + 1
    _log.append({
        "step": step, "elapsed_s": round(elapsed, 3),
        "source": source, "event": event, "detail": detail,
    })
    print(f"  [{step:>3}] +{elapsed:6.2f}s  {source:20s}  {event:12s}  {detail}")


def get_trace():
    """Returns a copy so callers can't mess with the internal list."""
    return list(_log)
