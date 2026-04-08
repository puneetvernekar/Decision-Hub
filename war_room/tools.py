"""
Programmatic tools that agents invoke before calling the LLM.

Each tool accepts the full dashboard dict and returns a structured result.
Agents specify which tools they need via their ``tools`` class attribute;
the base-agent machinery calls them automatically and injects the output
into the LLM prompt so the model works from *processed* data rather than
raw JSON.

Tool registry
─────────────
  aggregate_metrics   — per-metric stats, threshold breach flags, health score
  detect_anomalies    — Z-score anomaly detection across all time-series
  summarize_sentiment — channel/theme/timeline breakdown of user feedback
  compare_trends      — baseline deltas, direction, velocity, recovery ETA
"""

from __future__ import annotations

import math
from typing import Any

# ════════════════════════════════════════════════════════════════════════════
#  Tool 1 — Metric Aggregation & Threshold Check
# ════════════════════════════════════════════════════════════════════════════

_METRIC_KEYS = [
    "crash_rate_pct",
    "p95_latency_ms",
    "signup_conversion_pct",
    "retention_d1_pct",
    "retention_d7_pct",
    "payment_success_pct",
    "support_tickets",
    "feature_funnel_completion_pct",
    "churn_cancellations",
]

# Maps metric → (criteria key, "max" | "min")
_CRITERIA_MAP: dict[str, tuple[str, str]] = {
    "crash_rate_pct":               ("max_crash_rate_pct", "max"),
    "p95_latency_ms":               ("max_p95_latency_ms", "max"),
    "signup_conversion_pct":        ("min_signup_conversion_pct", "min"),
    "retention_d1_pct":             ("min_retention_d1_pct", "min"),
    "retention_d7_pct":             ("min_retention_d7_pct", "min"),
    "payment_success_pct":          ("min_payment_success_pct", "min"),
    "support_tickets":              ("max_support_tickets_per_day", "max"),
    "feature_funnel_completion_pct": ("min_feature_funnel_completion_pct", "min"),
}

# Maps metric → baseline key
_BASELINE_MAP: dict[str, str] = {
    "crash_rate_pct":               "crash_rate_pct",
    "p95_latency_ms":               "p95_latency_ms",
    "signup_conversion_pct":        "signup_conversion_pct",
    "retention_d1_pct":             "retention_d1_pct",
    "retention_d7_pct":             "retention_d7_pct",
    "payment_success_pct":          "payment_success_pct",
    "support_tickets":              "avg_support_tickets_per_day",
    "feature_funnel_completion_pct": "feature_funnel_completion_pct",
    "churn_cancellations":          "avg_churn_cancellations_per_day",
}

_LOWER_IS_BETTER = {"crash_rate_pct", "p95_latency_ms", "support_tickets", "churn_cancellations"}


def aggregate_metrics(dashboard: dict) -> dict[str, Any]:
    """Per-metric summary: latest, min, max, mean, trend, threshold breach.

    Returns
    -------
    dict with keys ``metric_summaries``, ``total_metrics``,
    ``breaching_count``, ``overall_health``.
    """
    daily = dashboard["daily_metrics"]
    baseline = dashboard["baseline_metrics"]
    criteria = dashboard["success_criteria"]

    summaries: dict[str, Any] = {}
    breaching = 0

    for metric in _METRIC_KEYS:
        values = [d[metric] for d in daily if d.get(metric) is not None]
        if not values:
            continue

        latest = values[-1]
        mn = min(values)
        mx = max(values)
        avg = sum(values) / len(values)

        # Trend: last-3-day avg vs first-3-day avg
        if len(values) >= 6:
            early = sum(values[:3]) / 3
            late = sum(values[-3:]) / 3
            if metric in _LOWER_IS_BETTER:
                trend = "improving" if late < early else ("worsening" if late > early else "stable")
            else:
                trend = "improving" if late > early else ("worsening" if late < early else "stable")
        else:
            trend = "insufficient_data"

        # Threshold breach
        breached = False
        threshold = None
        if metric in _CRITERIA_MAP:
            crit_key, crit_type = _CRITERIA_MAP[metric]
            threshold = criteria.get(crit_key)
            if threshold is not None:
                if crit_type == "max" and latest > threshold:
                    breached = True
                elif crit_type == "min" and latest < threshold:
                    breached = True
            if breached:
                breaching += 1

        # Baseline delta
        bl_key = _BASELINE_MAP.get(metric)
        bl_val = baseline.get(bl_key) if bl_key else None
        delta_pct = (
            round(((latest - bl_val) / bl_val) * 100, 1)
            if bl_val not in (None, 0) else None
        )

        summaries[metric] = {
            "latest": latest,
            "min": mn,
            "max": mx,
            "mean": round(avg, 2),
            "trend": trend,
            "threshold": threshold,
            "breached": breached,
            "baseline": bl_val,
            "delta_from_baseline_pct": delta_pct,
        }

    health = "healthy" if breaching == 0 else ("degraded" if breaching <= 2 else "critical")

    return {
        "metric_summaries": summaries,
        "total_metrics": len(summaries),
        "breaching_count": breaching,
        "overall_health": health,
    }


# ════════════════════════════════════════════════════════════════════════════
#  Tool 2 — Anomaly Detection (Z-score)
# ════════════════════════════════════════════════════════════════════════════

_ANOMALY_METRICS = [
    "crash_rate_pct",
    "p95_latency_ms",
    "signup_conversion_pct",
    "retention_d1_pct",
    "retention_d7_pct",
    "payment_success_pct",
    "support_tickets",
    "churn_cancellations",
]


def detect_anomalies(dashboard: dict, *, z_threshold: float = 2.0) -> dict[str, Any]:
    """Z-score anomaly detection across all time-series metrics.

    Returns
    -------
    dict with ``anomalies`` list (sorted by severity) and ``total_anomalies``.
    """
    daily = dashboard["daily_metrics"]
    anomalies: list[dict[str, Any]] = []

    for metric in _ANOMALY_METRICS:
        values = [(d["day"], d[metric]) for d in daily if d.get(metric) is not None]
        if len(values) < 3:
            continue

        nums = [v for _, v in values]
        mean = sum(nums) / len(nums)
        variance = sum((x - mean) ** 2 for x in nums) / len(nums)
        std = math.sqrt(variance)
        if std == 0:
            continue

        for day, val in values:
            z = abs(val - mean) / std
            if z >= z_threshold:
                direction = "spike" if val > mean else "drop"
                anomalies.append({
                    "metric": metric,
                    "day": day,
                    "value": val,
                    "mean": round(mean, 2),
                    "std_dev": round(std, 2),
                    "z_score": round(z, 2),
                    "direction": direction,
                    "severity": "high" if z >= 3.0 else "medium",
                })

    anomalies.sort(key=lambda a: (-{"high": 2, "medium": 1}[a["severity"]], -a["z_score"]))
    return {"anomalies": anomalies, "total_anomalies": len(anomalies)}


# ════════════════════════════════════════════════════════════════════════════
#  Tool 3 — Sentiment Summary
# ════════════════════════════════════════════════════════════════════════════

_THEME_KEYWORDS: dict[str, list[str]] = {
    "performance/latency": ["slow", "lag", "latency", "speed", "loading", "fast"],
    "data_loss":           ["lost", "disappeared", "data loss", "draft", "vanished", "gone"],
    "opt_out":             ["opt-out", "opt out", "disable", "turn off", "toggle", "opt_out"],
    "crash/freeze":        ["crash", "froze", "freeze", "hang", "unresponsive"],
    "ai_quality":          ["wrong", "inappropriate", "embarrassing", "autocomplete",
                            "suggestion", "compose", "weird"],
    "positive_experience": ["love", "amazing", "great", "awesome", "helpful",
                            "impressed", "game-changer", "incredible"],
    "accessibility":       ["accessibility", "disability", "screen reader", "a11y"],
}

_HIGH_IMPACT_CHANNELS = {"reddit", "twitter", "app_store"}
_HIGH_IMPACT_SIGNALS  = ["reddit", "twitter", "psa", "app store", "data loss", "embarrass"]


def summarize_sentiment(dashboard: dict) -> dict[str, Any]:
    """Channel/theme/timeline breakdown of user feedback.

    Returns
    -------
    dict with ``total_feedback``, ``sentiment_breakdown``, ``by_channel``,
    ``top_themes``, ``sentiment_timeline``, ``high_impact_items``,
    ``net_sentiment_score``.
    """
    feedback = dashboard["user_feedback"]

    by_channel: dict[str, dict[str, int]] = {}
    by_sentiment: dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0}
    themes: dict[str, int] = {}
    timeline: dict[int, dict[str, int]] = {}
    high_impact: list[dict[str, Any]] = []

    for entry in feedback:
        channel = entry.get("channel", "unknown")
        sentiment = entry.get("sentiment", "neutral")
        day = entry.get("day", 0)
        text = entry.get("feedback_text", "")

        # Channel breakdown
        if channel not in by_channel:
            by_channel[channel] = {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0, "total": 0}
        by_channel[channel][sentiment] = by_channel[channel].get(sentiment, 0) + 1
        by_channel[channel]["total"] += 1

        # Overall sentiment
        by_sentiment[sentiment] = by_sentiment.get(sentiment, 0) + 1

        # Timeline
        if day not in timeline:
            timeline[day] = {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0}
        timeline[day][sentiment] = timeline[day].get(sentiment, 0) + 1

        # Theme extraction
        text_lower = text.lower()
        for theme, keywords in _THEME_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                themes[theme] = themes.get(theme, 0) + 1

        # High-impact items
        if any(sig in text_lower for sig in _HIGH_IMPACT_SIGNALS):
            if channel in _HIGH_IMPACT_CHANNELS:
                high_impact.append({
                    "day": day,
                    "channel": channel,
                    "sentiment": sentiment,
                    "snippet": text[:120],
                })

    sorted_timeline = dict(sorted(timeline.items()))
    sorted_themes = dict(sorted(themes.items(), key=lambda x: -x[1]))
    total = max(len(feedback), 1)
    net_score = round((by_sentiment["positive"] - by_sentiment["negative"]) / total, 2)

    return {
        "total_feedback": len(feedback),
        "sentiment_breakdown": by_sentiment,
        "by_channel": by_channel,
        "top_themes": sorted_themes,
        "sentiment_timeline": sorted_timeline,
        "high_impact_items": high_impact[:10],
        "net_sentiment_score": net_score,
    }


# ════════════════════════════════════════════════════════════════════════════
#  Tool 4 — Trend Comparison (vs baseline)
# ════════════════════════════════════════════════════════════════════════════

_TREND_CONFIGS: list[tuple[str, str, str]] = [
    ("crash_rate_pct",               "crash_rate_pct",               "lower_is_better"),
    ("p95_latency_ms",               "p95_latency_ms",               "lower_is_better"),
    ("signup_conversion_pct",        "signup_conversion_pct",        "higher_is_better"),
    ("retention_d1_pct",             "retention_d1_pct",             "higher_is_better"),
    ("retention_d7_pct",             "retention_d7_pct",             "higher_is_better"),
    ("payment_success_pct",          "payment_success_pct",          "higher_is_better"),
    ("support_tickets",              "avg_support_tickets_per_day",  "lower_is_better"),
    ("churn_cancellations",          "avg_churn_cancellations_per_day", "lower_is_better"),
]


def compare_trends(dashboard: dict) -> dict[str, Any]:
    """Baseline deltas, direction, 3-day velocity, linear recovery ETA.

    Returns
    -------
    dict with ``comparisons`` list and ``improving_count`` / ``worsening_count``.
    """
    daily = dashboard["daily_metrics"]
    baseline = dashboard["baseline_metrics"]

    comparisons: list[dict[str, Any]] = []
    improving = 0
    worsening = 0

    for metric_key, baseline_key, polarity in _TREND_CONFIGS:
        values = [d[metric_key] for d in daily if d.get(metric_key) is not None]
        if len(values) < 2:
            continue

        bl_val = baseline.get(baseline_key)
        latest = values[-1]

        # 3-day moving average velocity
        if len(values) >= 6:
            recent_avg = sum(values[-3:]) / 3
            earlier_avg = sum(values[-6:-3]) / 3
            velocity = round(recent_avg - earlier_avg, 2)
        else:
            velocity = 0.0

        # Direction
        if abs(velocity) < 0.01:
            direction = "stable"
        elif (velocity > 0 and polarity == "higher_is_better") or \
             (velocity < 0 and polarity == "lower_is_better"):
            direction = "improving"
            improving += 1
        else:
            direction = "worsening"
            worsening += 1

        # Linear extrapolation: days to reach baseline
        days_to_baseline = None
        if bl_val is not None and velocity != 0:
            gap = (latest - bl_val) if polarity == "lower_is_better" else (bl_val - latest)
            if gap > 0:  # not yet recovered
                daily_rate = abs(velocity) / 3
                if daily_rate > 0:
                    days_to_baseline = round(gap / daily_rate, 1)

        delta = round(latest - bl_val, 2) if bl_val is not None else None
        delta_pct = (
            round(((latest - bl_val) / bl_val) * 100, 1)
            if bl_val not in (None, 0) else None
        )

        comparisons.append({
            "metric": metric_key,
            "latest": latest,
            "baseline": bl_val,
            "delta": delta,
            "delta_pct": delta_pct,
            "direction": direction,
            "velocity_per_3d": velocity,
            "projected_days_to_baseline": days_to_baseline,
        })

    return {
        "comparisons": comparisons,
        "improving_count": improving,
        "worsening_count": worsening,
    }


# ════════════════════════════════════════════════════════════════════════════
#  Tool Registry
# ════════════════════════════════════════════════════════════════════════════

TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "aggregate_metrics": {
        "fn": aggregate_metrics,
        "description": "Per-metric stats (latest/min/max/mean), trend direction, "
                       "threshold breach flags, and overall health score.",
    },
    "detect_anomalies": {
        "fn": detect_anomalies,
        "description": "Z-score anomaly detection across all daily metric "
                       "time-series.  Returns spikes/drops with severity.",
    },
    "summarize_sentiment": {
        "fn": summarize_sentiment,
        "description": "Channel/theme/timeline breakdown of user feedback "
                       "with high-impact item detection.",
    },
    "compare_trends": {
        "fn": compare_trends,
        "description": "Baseline deltas, 3-day velocity, direction, and "
                       "linear extrapolation of recovery time.",
    },
}
