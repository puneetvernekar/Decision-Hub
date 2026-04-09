"""
Mock dashboard data — loaded from CSV / Markdown files in ``data/``.

Input data (the stuff you'd pull from a real dashboard):
  data/daily_metrics.csv   — 10-day time series (9 metric dimensions)
  data/user_feedback.csv   — 35 feedback entries across 6 channels
  data/known_issues.csv    — 5 known issues with severity & status
  data/release_notes.md    — feature description & rollout plan

Reference constants (baselines, KPIs, success criteria) are defined
below — they frame how the agents interpret the data.
"""

import csv
from pathlib import Path

# ── Resolve data directory relative to project root ─────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FEATURE_NAME = "Smart Compose 2.0"
ROLLOUT_SCHEDULE = "5% → 10% → 25% → 50% staged rollout over 10 days"
CURRENT_ROLLOUT_PERCENTAGE = 25
TOTAL_USER_BASE = 520_000


# ── CSV helpers ─────────────────────────────────────────────────────────────

def read_csv_rows(filename):
    """Read a CSV file from data/ and return a list of dicts with auto-cast values."""
    path = DATA_DIR / filename
    with open(path, newline="", encoding="utf-8") as fh:
        rows = []
        for row in csv.DictReader(fh):
            typed = {}
            for k, v in row.items():
                if v == "" or v is None:
                    typed[k] = None
                else:
                    try:
                        f = float(v)
                        typed[k] = int(f) if f == int(f) and "." not in v else f
                    except (ValueError, OverflowError):
                        typed[k] = v
            rows.append(typed)
        return rows


def read_text(filename: str) -> str:
    """Read a plain-text / markdown file from data/."""
    path = DATA_DIR / filename
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# ── Load input data from CSV / Markdown ─────────────────────────────────────

DAILY_METRICS: list[dict] = read_csv_rows("daily_metrics.csv")
USER_FEEDBACK: list[dict] = read_csv_rows("user_feedback.csv")
KNOWN_ISSUES:  list[dict] = read_csv_rows("known_issues.csv")
RELEASE_NOTES: str        = read_text("release_notes.md")


# ── Reference constants ─────────────────────────────────────────────────────

# Pre-launch baselines (7-day averages + survey scores)
BASELINE_METRICS: dict = {
    "signup_conversion_pct": 4.4,
    "retention_d1_pct": 73.0,
    "retention_d7_pct": 45.0,
    "crash_rate_pct": 0.35,
    "p95_latency_ms": 195,
    "payment_success_pct": 99.7,
    "avg_support_tickets_per_day": 38,
    "nps": 42,
    "csat": 4.1,
}

# PM-defined success criteria (thresholds)
SUCCESS_CRITERIA: dict = {
    "max_crash_rate_pct": 0.75,
    "max_p95_latency_ms": 300,
    "min_signup_conversion_pct": 3.8,
    "min_retention_d1_pct": 68.0,
    "min_retention_d7_pct": 42.0,
    "min_payment_success_pct": 99.0,
    "max_support_tickets_per_day": 80,
    "min_nps": 35,
    "min_csat": 3.8,
}


# ── Public API ──────────────────────────────────────────────────────────────

def get_dashboard_snapshot() -> dict:
    """Return the full mock dashboard as a single dictionary."""
    return {
        "feature_name": FEATURE_NAME,
        "rollout_schedule": ROLLOUT_SCHEDULE,
        "current_rollout_percentage": CURRENT_ROLLOUT_PERCENTAGE,
        "total_user_base": TOTAL_USER_BASE,
        "daily_metrics": DAILY_METRICS,
        "baseline_metrics": BASELINE_METRICS,
        "user_feedback": USER_FEEDBACK,
        "success_criteria": SUCCESS_CRITERIA,
        "release_notes": RELEASE_NOTES,
        "known_issues": KNOWN_ISSUES,
    }
