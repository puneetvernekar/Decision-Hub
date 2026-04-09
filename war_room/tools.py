# Tools that agents call before the LLM to process raw dashboard data.

from __future__ import annotations

import math
from typing import Any


# Unified metric config: (criteria_key, "max"|"min"|None, baseline_key, lower_is_better)
METRIC_CONFIG = {
    "crash_rate_pct":               ("max_crash_rate_pct",              "max", "crash_rate_pct",               True),
    "p95_latency_ms":               ("max_p95_latency_ms",             "max", "p95_latency_ms",               True),
    "signup_conversion_pct":        ("min_signup_conversion_pct",      "min", "signup_conversion_pct",        False),
    "retention_d1_pct":             ("min_retention_d1_pct",           "min", "retention_d1_pct",             False),
    "retention_d7_pct":             ("min_retention_d7_pct",           "min", "retention_d7_pct",             False),
    "payment_success_pct":          ("min_payment_success_pct",        "min", "payment_success_pct",          False),
    "support_tickets":              ("max_support_tickets_per_day",    "max", "avg_support_tickets_per_day",  True),
    "feature_funnel_completion_pct": ("min_feature_funnel_completion_pct", "min", "feature_funnel_completion_pct", False),
    "churn_cancellations":          (None,                              None, "avg_churn_cancellations_per_day", True),
}

THEME_KEYWORDS = {
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

HIGH_IMPACT_CHANNELS = {"reddit", "twitter", "app_store"}
HIGH_IMPACT_SIGNALS = ["reddit", "twitter", "psa", "app store", "data loss", "embarrass"]

# Trend comparison config: (metric_key, baseline_key, polarity)
TREND_CONFIGS = [
    ("crash_rate_pct",               "crash_rate_pct",                  "lower_is_better"),
    ("p95_latency_ms",               "p95_latency_ms",                  "lower_is_better"),
    ("signup_conversion_pct",        "signup_conversion_pct",           "higher_is_better"),
    ("retention_d1_pct",             "retention_d1_pct",                "higher_is_better"),
    ("retention_d7_pct",             "retention_d7_pct",                "higher_is_better"),
    ("payment_success_pct",          "payment_success_pct",             "higher_is_better"),
    ("support_tickets",              "avg_support_tickets_per_day",     "lower_is_better"),
    ("churn_cancellations",          "avg_churn_cancellations_per_day", "lower_is_better"),
]


def aggregate_metrics(dashboard: dict) -> dict[str, Any]:
    """Per-metric summary with trend direction and threshold breach checks."""
    daily = dashboard["daily_metrics"]
    baseline = dashboard["baseline_metrics"]
    criteria = dashboard["success_criteria"]

    summaries = {}
    breaching = 0

    for metric, (crit_key, crit_type, bl_key, lower_better) in METRIC_CONFIG.items():
        values = [d[metric] for d in daily if d.get(metric) is not None]
        if not values:
            continue

        latest = values[-1]
        mn, mx = min(values), max(values)
        avg = sum(values) / len(values)

        # Trend: compare first-3 vs last-3 day averages
        if len(values) >= 6:
            early = sum(values[:3]) / 3
            late = sum(values[-3:]) / 3
            if lower_better:
                trend = "improving" if late < early else ("worsening" if late > early else "stable")
            else:
                trend = "improving" if late > early else ("worsening" if late < early else "stable")
        else:
            trend = "insufficient_data"

        # Check if latest value breaches the threshold
        breached = False
        threshold = None
        if crit_key and crit_type:
            threshold = criteria.get(crit_key)
            if threshold is not None:
                if crit_type == "max" and latest > threshold:
                    breached = True
                elif crit_type == "min" and latest < threshold:
                    breached = True
            if breached:
                breaching += 1

        bl_val = baseline.get(bl_key) if bl_key else None
        delta_pct = (
            round(((latest - bl_val) / bl_val) * 100, 1)
            if bl_val not in (None, 0) else None
        )

        summaries[metric] = {
            "latest": latest, "min": mn, "max": mx, "mean": round(avg, 2),
            "trend": trend, "threshold": threshold, "breached": breached,
            "baseline": bl_val, "delta_from_baseline_pct": delta_pct,
        }

    health = "healthy" if breaching == 0 else ("degraded" if breaching <= 2 else "critical")
    return {
        "metric_summaries": summaries,
        "total_metrics": len(summaries),
        "breaching_count": breaching,
        "overall_health": health,
    }


def detect_anomalies(dashboard: dict, *, z_threshold: float = 2.0) -> dict[str, Any]:
    """Z-score anomaly detection across daily metric time-series."""
    daily = dashboard["daily_metrics"]
    anomalies = []

    # skip feature_funnel_completion_pct — no meaningful baseline for z-score on a new feature
    skip = {"feature_funnel_completion_pct"}

    for metric in METRIC_CONFIG:
        if metric in skip:
            continue

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
                anomalies.append({
                    "metric": metric, "day": day, "value": val,
                    "mean": round(mean, 2), "std_dev": round(std, 2),
                    "z_score": round(z, 2),
                    "direction": "spike" if val > mean else "drop",
                    "severity": "high" if z >= 3.0 else "medium",
                })

    anomalies.sort(key=lambda a: (-{"high": 2, "medium": 1}[a["severity"]], -a["z_score"]))
    return {"anomalies": anomalies, "total_anomalies": len(anomalies)}


def summarize_sentiment(dashboard: dict) -> dict[str, Any]:
    """Breaks down user feedback by channel, theme, and timeline."""
    feedback = dashboard["user_feedback"]

    by_channel = {}
    by_sentiment = {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0}
    themes = {}
    timeline = {}
    high_impact = []

    for entry in feedback:
        channel = entry.get("channel", "unknown")
        sentiment = entry.get("sentiment", "neutral")
        day = entry.get("day", 0)
        text = entry.get("feedback_text", "")

        if channel not in by_channel:
            by_channel[channel] = {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0, "total": 0}
        by_channel[channel][sentiment] = by_channel[channel].get(sentiment, 0) + 1
        by_channel[channel]["total"] += 1

        by_sentiment[sentiment] = by_sentiment.get(sentiment, 0) + 1

        if day not in timeline:
            timeline[day] = {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0}
        timeline[day][sentiment] = timeline[day].get(sentiment, 0) + 1

        # match feedback text against known themes
        text_lower = text.lower()
        for theme, keywords in THEME_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                themes[theme] = themes.get(theme, 0) + 1

        # flag high-impact items (viral risk channels + sensitive keywords)
        if any(sig in text_lower for sig in HIGH_IMPACT_SIGNALS):
            if channel in HIGH_IMPACT_CHANNELS:
                high_impact.append({
                    "day": day, "channel": channel,
                    "sentiment": sentiment, "snippet": text[:120],
                })

    total = max(len(feedback), 1)
    net_score = round((by_sentiment["positive"] - by_sentiment["negative"]) / total, 2)

    return {
        "total_feedback": len(feedback),
        "sentiment_breakdown": by_sentiment,
        "by_channel": by_channel,
        "top_themes": dict(sorted(themes.items(), key=lambda x: -x[1])),
        "sentiment_timeline": dict(sorted(timeline.items())),
        "high_impact_items": high_impact[:10],
        "net_sentiment_score": net_score,
    }


def compare_trends(dashboard: dict) -> dict[str, Any]:
    """Compares current metrics to baseline with velocity and recovery ETA."""
    daily = dashboard["daily_metrics"]
    baseline = dashboard["baseline_metrics"]

    comparisons = []
    improving = 0
    worsening = 0

    for metric_key, baseline_key, polarity in TREND_CONFIGS:
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

        if abs(velocity) < 0.01:
            direction = "stable"
        elif (velocity > 0 and polarity == "higher_is_better") or \
             (velocity < 0 and polarity == "lower_is_better"):
            direction = "improving"
            improving += 1
        else:
            direction = "worsening"
            worsening += 1

        # rough linear extrapolation: how many days to get back to baseline
        days_to_baseline = None
        if bl_val is not None and velocity != 0:
            gap = (latest - bl_val) if polarity == "lower_is_better" else (bl_val - latest)
            if gap > 0:
                daily_rate = abs(velocity) / 3
                if daily_rate > 0:
                    days_to_baseline = round(gap / daily_rate, 1)

        delta = round(latest - bl_val, 2) if bl_val is not None else None
        delta_pct = (
            round(((latest - bl_val) / bl_val) * 100, 1)
            if bl_val not in (None, 0) else None
        )

        comparisons.append({
            "metric": metric_key, "latest": latest, "baseline": bl_val,
            "delta": delta, "delta_pct": delta_pct, "direction": direction,
            "velocity_per_3d": velocity, "projected_days_to_baseline": days_to_baseline,
        })

    return {
        "comparisons": comparisons,
        "improving_count": improving,
        "worsening_count": worsening,
    }


# Registry used by BaseAgent._invoke_tools()
TOOL_REGISTRY = {
    "aggregate_metrics": {
        "fn": aggregate_metrics,
        "description": "Per-metric stats, trend direction, threshold breach flags, overall health.",
    },
    "detect_anomalies": {
        "fn": detect_anomalies,
        "description": "Z-score anomaly detection across daily metrics. Returns spikes/drops with severity.",
    },
    "summarize_sentiment": {
        "fn": summarize_sentiment,
        "description": "Channel/theme/timeline breakdown of user feedback with high-impact detection.",
    },
    "compare_trends": {
        "fn": compare_trends,
        "description": "Baseline deltas, 3-day velocity, direction, and recovery time estimate.",
    },
}
