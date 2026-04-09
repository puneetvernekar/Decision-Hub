"""Simple trace logger — records steps in memory and prints to console."""

import time

log_list = []
start_time = 0.0


def reset_trace():
    global log_list, start_time
    log_list = []
    start_time = time.monotonic()


def trace(source, event, detail):
    if not start_time:
        reset_trace()

    elapsed = time.monotonic() - start_time
    step = len(log_list) + 1
    log_list.append({
        "step": step, "elapsed_s": round(elapsed, 3),
        "source": source, "event": event, "detail": detail,
    })
    print(f"  [{step:>3}] +{elapsed:6.2f}s  {source:20s}  {event:12s}  {detail}")


def get_trace():
    """Returns a copy so callers can't mess with the internal list."""
    return list(log_list)
