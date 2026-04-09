# Tools that agents call before the LLM to process raw dashboard data.


# Each metric: (criteria_key, "max"|"min"|None, baseline_key, lower_is_better)
METRIC_CONFIG = {
    "crash_rate_pct":        ("max_crash_rate_pct",         "max", "crash_rate_pct",              True),
    "p95_latency_ms":        ("max_p95_latency_ms",         "max", "p95_latency_ms",              True),
    "signup_conversion_pct": ("min_signup_conversion_pct",  "min", "signup_conversion_pct",       False),
    "retention_d1_pct":      ("min_retention_d1_pct",       "min", "retention_d1_pct",            False),
    "retention_d7_pct":      ("min_retention_d7_pct",       "min", "retention_d7_pct",            False),
    "payment_success_pct":   ("min_payment_success_pct",    "min", "payment_success_pct",         False),
    "support_tickets":       ("max_support_tickets_per_day","max", "avg_support_tickets_per_day", True),
    "nps":                   ("min_nps",                    "min", "nps",                         False),
    "csat":                  ("min_csat",                   "min", "csat",                        False),
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


def aggregate_metrics(dashboard):
    """Summarizes each metric: latest value, trend, threshold breach, baseline delta."""
    daily = dashboard["daily_metrics"]
    baseline = dashboard["baseline_metrics"]
    criteria = dashboard["success_criteria"]

    summaries = {}
    breaching = 0

    for metric, (criteria_key, criteria_type, baseline_key, lower_is_better) in METRIC_CONFIG.items():
        values = [d[metric] for d in daily if d.get(metric) is not None]
        if not values:
            continue

        latest = values[-1]
        avg = sum(values) / len(values)

        # simple trend: compare first-3 vs last-3 day averages
        trend = "insufficient_data"
        if len(values) >= 6:
            early = sum(values[:3]) / 3
            late = sum(values[-3:]) / 3
            if abs(late - early) < 0.01:
                trend = "stable"
            elif (late < early) == lower_is_better:
                trend = "improving"
            else:
                trend = "worsening"

        # threshold breach check
        breached = False
        threshold = criteria.get(criteria_key) if criteria_key else None
        if threshold is not None:
            if criteria_type == "max" and latest > threshold:
                breached = True
            elif criteria_type == "min" and latest < threshold:
                breached = True
            if breached:
                breaching += 1

        baseline_val = baseline.get(baseline_key) if baseline_key else None
        delta_pct = None
        if baseline_val not in (None, 0):
            delta_pct = round(((latest - baseline_val) / baseline_val) * 100, 1)

        summaries[metric] = {
            "latest": latest,
            "min": min(values),
            "max": max(values),
            "mean": round(avg, 2),
            "trend": trend,
            "threshold": threshold,
            "breached": breached,
            "baseline": baseline_val,
            "delta_from_baseline_pct": delta_pct,
        }

    health = "healthy" if breaching == 0 else ("degraded" if breaching <= 2 else "critical")
    return {
        "metric_summaries": summaries,
        "total_metrics": len(summaries),
        "breaching_count": breaching,
        "overall_health": health,
    }


def summarize_sentiment(dashboard):
    """Counts sentiment by channel, extracts themes, flags high-impact items."""
    feedback = dashboard["user_feedback"]

    channel_counts = {}
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0}
    themes = {}
    high_impact = []

    for entry in feedback:
        channel = entry.get("channel", "unknown")
        sentiment = entry.get("sentiment", "neutral")
        text = entry.get("feedback_text", "")
        day = entry.get("day", 0)

        # count by channel
        if channel not in channel_counts:
            channel_counts[channel] = {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0, "total": 0}
        channel_counts[channel][sentiment] = channel_counts[channel].get(sentiment, 0) + 1
        channel_counts[channel]["total"] += 1
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

        # keyword theme matching
        text_lower = text.lower()
        for theme, keywords in THEME_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                themes[theme] = themes.get(theme, 0) + 1

        # flag stuff from high-reach channels that mentions sensitive topics
        if channel in HIGH_IMPACT_CHANNELS:
            if any(sig in text_lower for sig in HIGH_IMPACT_SIGNALS):
                high_impact.append({
                    "day": day, "channel": channel,
                    "sentiment": sentiment, "snippet": text[:120],
                })

    total = max(len(feedback), 1)
    net_score = round((sentiment_counts["positive"] - sentiment_counts["negative"]) / total, 2)

    return {
        "total_feedback": len(feedback),
        "sentiment_breakdown": sentiment_counts,
        "by_channel": channel_counts,
        "top_themes": dict(sorted(themes.items(), key=lambda x: -x[1])),
        "high_impact_items": high_impact[:10],
        "net_sentiment_score": net_score,
    }


def compare_trends(dashboard):
    """Simple baseline comparison: latest value vs pre-launch baseline for each metric."""
    daily = dashboard["daily_metrics"]
    baseline = dashboard["baseline_metrics"]

    comparisons = []
    improving = 0
    worsening = 0

    for metric, (_, _, baseline_key, lower_is_better) in METRIC_CONFIG.items():
        values = [d[metric] for d in daily if d.get(metric) is not None]
        if len(values) < 2:
            continue

        baseline_val = baseline.get(baseline_key)
        if baseline_val is None:
            continue

        latest = values[-1]
        delta = round(latest - baseline_val, 2)
        delta_pct = round((delta / baseline_val) * 100, 1) if baseline_val != 0 else None

        # is it getting better or worse vs baseline?
        if abs(delta) < 0.01:
            direction = "stable"
        elif (delta < 0) == lower_is_better:
            direction = "improving"
            improving += 1
        else:
            direction = "worsening"
            worsening += 1

        comparisons.append({
            "metric": metric,
            "latest": latest,
            "baseline": baseline_val,
            "delta": delta,
            "delta_pct": delta_pct,
            "direction": direction,
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
    "summarize_sentiment": {
        "fn": summarize_sentiment,
        "description": "Sentiment counts by channel, theme extraction, high-impact item flags.",
    },
    "compare_trends": {
        "fn": compare_trends,
        "description": "Latest value vs baseline for each metric with direction.",
    },
}
