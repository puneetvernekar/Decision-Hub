"""
War Room — Multi-Agent Launch Decision System
==============================================
Run:  python main.py [--model MODEL] [--offline] [--json]

--offline  runs a deterministic simulation without calling any LLM,
           useful for testing the pipeline end-to-end.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

from war_room.mock_dashboard import get_dashboard_snapshot
from war_room.models import AgentVerdict, Decision, WarRoomOutcome
from war_room.tools import (
    TOOL_REGISTRY,
    aggregate_metrics,
    compare_trends,
    detect_anomalies,
    summarize_sentiment,
)


# ── Pretty-print helpers ────────────────────────────────────────────────────

def _verdict_to_dict(v: AgentVerdict) -> dict:
    """Convert an AgentVerdict to a JSON-serialisable dict."""
    return {
        "agent": v.agent_name,
        "role": v.role,
        "decision": v.decision.value,
        "confidence": v.confidence,
        "rationale": v.rationale,
        "key_evidence": v.key_evidence,
        "recommended_actions": v.recommended_actions,
        "dissent_notes": v.dissent_notes,
    }


def _print_outcome(outcome: WarRoomOutcome) -> None:
    W = 64
    RESET = "\033[0m"
    decision_color = {
        Decision.PROCEED: "\033[92m",    # green
        Decision.PAUSE: "\033[93m",      # yellow
        Decision.ROLL_BACK: "\033[91m",  # red
    }
    color = decision_color.get(outcome.final_decision, "")

    print("\n" + "=" * W)
    print("  FINAL WAR-ROOM DECISION")
    print("=" * W)
    print(f"\n  Decision:  {color}{outcome.final_decision.value}{RESET}\n")
    print(f"  Rationale:\n{textwrap.fill(outcome.decision_rationale, W, initial_indent='    ', subsequent_indent='    ')}\n")

    # Phase 1 — Initial verdicts
    print("─" * W)
    print("  Phase 1 — Initial Verdicts")
    print("─" * W)
    for v in outcome.initial_verdicts:
        print(f"  • {v.agent_name:22s} → {v.decision.value:10s} (conf: {v.confidence:.0%})")
    print()

    # Phase 2a — Risk/Critic challenge
    print("─" * W)
    print("  Phase 2a — Risk/Critic Challenge")
    print("─" * W)
    c = outcome.critique
    print(f"  • {c.agent_name:22s} → {c.decision.value:10s} (conf: {c.confidence:.0%})")
    # Show first 200 chars of rationale
    wrapped = textwrap.fill(c.rationale[:200], W - 4, initial_indent="    ", subsequent_indent="    ")
    print(wrapped)
    print()

    # Phase 2b — Revised verdicts
    print("─" * W)
    print("  Phase 2b — Revised Verdicts (after deliberation)")
    print("─" * W)
    for v in outcome.revised_verdicts:
        init = next((i for i in outcome.initial_verdicts if i.role == v.role), None)
        change = ""
        if init and (init.decision != v.decision or abs(init.confidence - v.confidence) > 0.03):
            change = f"  ← was {init.decision.value} ({init.confidence:.0%})"
        print(f"  • {v.agent_name:22s} → {v.decision.value:10s} (conf: {v.confidence:.0%}){change}")
    print()

    # Action plan
    print("─" * W)
    print("  Action Plan")
    print("─" * W)
    for i, step in enumerate(outcome.action_plan, 1):
        print(f"  {i}. {step}")
    print()

    # Risks
    print("─" * W)
    print("  Risks & Mitigations")
    print("─" * W)
    for item in outcome.risks_and_mitigations:
        print(f"  • {item}")
    print()

    # Monitoring
    print("─" * W)
    print("  Follow-up Monitoring")
    print("─" * W)
    for item in outcome.follow_up_monitoring:
        print(f"  • {item}")
    print()

    # Dissent
    if outcome.dissenting_opinions:
        print("─" * W)
        print("  Dissenting Opinions")
        print("─" * W)
        for item in outcome.dissenting_opinions:
            print(f"  • {item}")
        print()

    print("=" * W)


# ── Offline simulation (no LLM needed) ─────────────────────────────────────

def _run_offline(dashboard: dict) -> WarRoomOutcome:
    """Deterministic offline simulation for testing without an API key."""
    print("\n⚡ Running OFFLINE simulation (no LLM calls)\n")

    # ── Invoke tools programmatically (same as real agents do) ──────────
    metrics_report = aggregate_metrics(dashboard)
    anomaly_report = detect_anomalies(dashboard)
    sentiment_report = summarize_sentiment(dashboard)
    trend_report = compare_trends(dashboard)

    print("─" * 64)
    print("  Tool Invocations")
    print("─" * 64)
    print(f"  🔧 Product Manager  invoked: aggregate_metrics, compare_trends")
    print(f"  🔧 Data Analyst     invoked: aggregate_metrics, detect_anomalies")
    print(f"  🔧 Marketing & Comms invoked: summarize_sentiment")
    print(f"  🔧 Risk / Critic    invoked: detect_anomalies, aggregate_metrics")
    print(f"\n  Results snapshot:")
    print(f"    aggregate_metrics  → health: {metrics_report['overall_health']}, "
          f"{metrics_report['breaching_count']}/{metrics_report['total_metrics']} metrics breaching")
    print(f"    detect_anomalies   → {anomaly_report['total_anomalies']} anomalies found "
          f"({sum(1 for a in anomaly_report['anomalies'] if a['severity'] == 'high')} high severity)")
    print(f"    summarize_sentiment → net score: {sentiment_report['net_sentiment_score']}, "
          f"{sentiment_report['sentiment_breakdown']['negative']} negative / "
          f"{sentiment_report['sentiment_breakdown']['positive']} positive")
    top_themes = list(sentiment_report['top_themes'].items())[:3]
    if top_themes:
        themes_str = ", ".join(f"{t[0]} ({t[1]})" for t in top_themes)
        print(f"                        top themes: {themes_str}")
    print(f"    compare_trends     → {trend_report['improving_count']} improving, "
          f"{trend_report['worsening_count']} worsening")
    print()

    # ── Extract data ────────────────────────────────────────────────────
    daily = dashboard["daily_metrics"]
    baseline = dashboard["baseline_metrics"]
    criteria = dashboard["success_criteria"]
    known_issues = dashboard["known_issues"]

    latest = daily[-1]  # Day 10
    n_days = len(daily)

    # Derived aggregates
    peak_crash = max(d["crash_rate_pct"] for d in daily)
    peak_latency = max(d["p95_latency_ms"] for d in daily)
    peak_tickets = max(d["support_tickets"] for d in daily)
    peak_churn = max(d["churn_cancellations"] for d in daily)
    total_cancellations = sum(d["churn_cancellations"] for d in daily)
    avg_churn_per_day = total_cancellations / n_days
    latest_crash = latest["crash_rate_pct"]
    latest_latency = latest["p95_latency_ms"]

    d7_values = [d["retention_d7_pct"] for d in daily if d["retention_d7_pct"] is not None]
    latest_d7 = d7_values[-1] if d7_values else None

    # Feedback analysis
    feedback = dashboard["user_feedback"]
    neg_count = sum(1 for f in feedback if f["sentiment"] == "negative")
    pos_count = sum(1 for f in feedback if f["sentiment"] == "positive")
    neu_count = len(feedback) - neg_count - pos_count

    # ════════════════════════════════════════════════════════════════════
    #  PHASE 1 — Initial independent verdicts (PM, Data, Marketing)
    # ════════════════════════════════════════════════════════════════════

    pm_initial = AgentVerdict(
        agent_name="Product Manager",
        role="pm",
        decision=Decision.PAUSE,
        confidence=0.72,
        rationale=(
            f"Day 10 signup conversion ({latest['signup_conversion_pct']}%) has recovered to the "
            f"threshold ({criteria['min_signup_conversion_pct']}%), but D1 retention at "
            f"{latest['retention_d1_pct']}% barely meets the {criteria['min_retention_d1_pct']}% "
            f"criterion. D7 retention at {latest_d7}% is below the {criteria['min_retention_d7_pct']}% "
            f"target — a concerning signal for long-term value. "
            f"Feature funnel completion at {latest['feature_funnel_completion_pct']}% is strong and exceeds the "
            f"{criteria['min_feature_funnel_completion_pct']}% target. "
            f"The missing opt-out toggle (KI-004) is a significant friction source. "
            f"Recommend pausing at 25% until the opt-out ships and D7 retention stabilises."
        ),
        key_evidence=[
            f"Signup conversion Day 10: {latest['signup_conversion_pct']}% (threshold: {criteria['min_signup_conversion_pct']}%)",
            f"D7 retention: {latest_d7}% — below {criteria['min_retention_d7_pct']}% target",
            f"Feature funnel completion: {latest['feature_funnel_completion_pct']}% — strong signal",
            f"Total cancellations over {n_days} days: {total_cancellations} ({avg_churn_per_day:.0f}/day vs {baseline['avg_churn_cancellations_per_day']}/day baseline)",
        ],
        recommended_actions=[
            "Freeze rollout at 25% — do not expand to the 50% cohort.",
            "Expedite the opt-out toggle (KI-004) before any further expansion.",
            "Run a D7 retention deep-dive to separate performance-driven churn from feature-driven churn.",
        ],
    )

    data_initial = AgentVerdict(
        agent_name="Data Analyst",
        role="data_analyst",
        decision=Decision.PAUSE,
        confidence=0.82,
        rationale=(
            f"Crash rate peaked at {peak_crash}% on Day 6 ({peak_crash/baseline['crash_rate_pct']:.1f}x "
            f"baseline, threshold {criteria['max_crash_rate_pct']}%). Post-hotfix recovery is clear: "
            f"Day 10 crash rate {latest_crash}% is above the {criteria['max_crash_rate_pct']}% threshold. "
            f"P95 latency peaked at {peak_latency}ms (Day 6), now {latest_latency}ms — within "
            f"the {criteria['max_p95_latency_ms']}ms target. "
            f"Payment success rate dipped to {min(d['payment_success_pct'] for d in daily)}% on Day 6 "
            f"(threshold: {criteria['min_payment_success_pct']}%). Now at {latest['payment_success_pct']}%. "
            f"Support tickets peaked at {peak_tickets}/day (Day 6) vs {baseline['avg_support_tickets_per_day']}/day "
            f"baseline. Now {latest['support_tickets']}/day — still elevated. "
            f"D7 retention ({latest_d7}%) is 5.8pp below baseline ({baseline['retention_d7_pct']}%) — "
            f"statistically significant. Trends are recovering but 3 of 9 metrics still breach thresholds."
        ),
        key_evidence=[
            f"Crash rate: peaked {peak_crash}% → now {latest_crash}% (threshold: {criteria['max_crash_rate_pct']}%)",
            f"P95 latency: peaked {peak_latency}ms → now {latest_latency}ms (threshold: {criteria['max_p95_latency_ms']}ms)",
            f"Payment success: dipped to {min(d['payment_success_pct'] for d in daily)}%, now {latest['payment_success_pct']}%",
            f"Support tickets: peaked {peak_tickets}/day, now {latest['support_tickets']}/day (baseline: {baseline['avg_support_tickets_per_day']})",
            f"D7 retention: {latest_d7}% vs {baseline['retention_d7_pct']}% baseline (-{baseline['retention_d7_pct'] - latest_d7:.1f}pp)",
            f"DAU recovered: {latest['dau']:,} (baseline-scaled: ~{baseline['avg_dau'] * latest['rollout_pct'] // 5:,})",
            f"Churn: peaked {peak_churn}/day (Day 6), now {latest['churn_cancellations']}/day (baseline: {baseline['avg_churn_cancellations_per_day']})",
        ],
        recommended_actions=[
            "Hold at 25% rollout until crash rate < 0.75% for 3 consecutive days.",
            "Set automated monitoring gate: all 9 metrics must be within thresholds for 48h before expansion.",
            "Run root-cause analysis on the Day 5-6 crash/latency spike correlation.",
            "Track D7 retention for the Day 3-4 cohort (data available Day 10-11) to confirm trend.",
        ],
    )

    marketing_initial = AgentVerdict(
        agent_name="Marketing & Comms",
        role="marketing_comms",
        decision=Decision.PAUSE,
        confidence=0.70,
        rationale=(
            f"Of 35 feedback entries: {pos_count} positive, {neg_count} negative, {neu_count} neutral. "
            f"Negative feedback is concentrated on Days 4-6 (the trouble window) and clusters around "
            f"3 recurring themes: (1) performance/lag — 6 mentions, (2) no opt-out — 5 mentions, "
            f"(3) draft data loss — 3 mentions. Two high-impact outliers: a user with a disability "
            f"praising the feature (#35) and a sales email auto-complete incident that could have "
            f"cost a deal (#34). App store ratings are polarised (1-star and 5-star). "
            f"Reddit thread about draft data loss (PSA post) has amplification risk. "
            f"Post-hotfix sentiment (Days 8-10) is more positive, but the missing opt-out toggle "
            f"is a persistent irritant across all channels."
        ),
        key_evidence=[
            f"Feedback: {pos_count} positive / {neg_count} negative / {neu_count} neutral across 6 channels",
            "Top repeated issue: performance/lag (6 mentions across in-app, support, Twitter)",
            "Top repeated issue: no opt-out toggle (5 mentions across in-app, support, Twitter, Reddit)",
            "Critical outlier: draft data loss creating PSA-style Reddit post (amplification risk)",
            "Critical outlier: sales email auto-complete incident — potential enterprise deal risk",
            "Positive outlier: accessibility user (#35) — strong advocacy potential if feature preserved",
            "App store: polarised 1-star / 5-star reviews",
        ],
        recommended_actions=[
            "Issue transparent status update: acknowledge performance issues, draft-loss bug, and opt-out gap.",
            "Respond personally to the Twitter auto-complete complaints with empathy and DM follow-up.",
            "Prepare accessibility-focused case study with user #35's permission — powerful positive narrative.",
            "Do NOT run any promotional campaigns for Smart Compose until opt-out toggle ships.",
            "Monitor Reddit thread on draft loss — if it trends, issue an official response.",
        ],
    )

    initial_verdicts = [pm_initial, data_initial, marketing_initial]

    # ════════════════════════════════════════════════════════════════════
    #  PHASE 2a — Risk/Critic challenge
    # ════════════════════════════════════════════════════════════════════

    critique = AgentVerdict(
        agent_name="Risk / Critic",
        role="risk_critic",
        decision=Decision.ROLL_BACK,
        confidence=0.68,
        rationale=(
            f"All three agents recommend Pause, but I challenge their shared assumption that "
            f"the recovery trend is sufficient to hold at 25%. "
            f"KI-001 (draft data loss) remains in 'investigating' status after 10 days — this is a "
            f"critical severity bug reported by 3 users and amplified on Reddit. At 25% rollout "
            f"({dashboard['total_user_base'] * 25 // 100:,} users exposed), the blast radius for a "
            f"data-loss bug is unacceptable. The PM's focus on funnel completion (28.5%) is an "
            f"anchoring bias — strong adoption means MORE users are exposed to the data-loss risk. "
            f"The Data Analyst notes 3 of 9 metrics still breach thresholds but still recommends Pause; "
            f"this conflates 'improving' with 'acceptable'. The Marketing agent correctly identifies "
            f"the amplification risk but underweights it — a single viral thread about data loss can "
            f"do more brand damage than all the performance complaints combined. "
            f"Cancellations averaged {avg_churn_per_day:.0f}/day over 10 days vs "
            f"{baseline['avg_churn_cancellations_per_day']}/day baseline — a "
            f"{avg_churn_per_day / baseline['avg_churn_cancellations_per_day']:.1f}x increase. "
            f"The recovery trend is only 3 days old and untested at higher rollout. "
            f"I recommend Roll Back to protect user trust while fixes are shipped."
        ),
        key_evidence=[
            f"KI-001 (critical): draft data loss — still 'investigating' after 10 days",
            f"Churn: {total_cancellations} cancellations in {n_days} days ({avg_churn_per_day:.0f}/day vs {baseline['avg_churn_cancellations_per_day']}/day baseline)",
            f"D7 retention: {latest_d7}% < {criteria['min_retention_d7_pct']}% target",
            f"Crash rate Day 10: {latest_crash}% — still above {criteria['max_crash_rate_pct']}% threshold",
            f"Payment success dip to {min(d['payment_success_pct'] for d in daily)}% signals infrastructure fragility",
            f"3 of 9 metrics still breaching thresholds on Day 10",
            f"PM's 28.5% funnel adoption = more users exposed to data-loss bug (anchoring risk)",
        ],
        recommended_actions=[
            "Roll back to pre-launch state for the 25% cohort — data-loss risk is too high.",
            "If rollback is rejected: freeze at 25%, mandate KI-001 fix within 48h, add opt-out toggle.",
            "Commission independent review of the inference API scaling before any expansion.",
            "Conduct blameless post-incident review covering the Day 5-6 degradation.",
        ],
        dissent_notes=(
            "The majority leans toward Pause, but pausing at 25% still exposes ~130k users "
            "to a data-loss bug (KI-001) with no confirmed fix. The recovery trend is only "
            "3 days old (Day 7 hotfix) and has not been stress-tested at higher rollout. "
            "The draft-disappearing bug alone warrants rollback — it is a trust-destroying event "
            "and 3 users have already reported it. If even 0.1% of the 130k cohort experiences "
            "data loss, that's 130 affected users with high churn and social amplification risk."
        ),
    )

    # ════════════════════════════════════════════════════════════════════
    #  PHASE 2b — Revised verdicts (after seeing peers + critique)
    # ════════════════════════════════════════════════════════════════════

    pm_revised = AgentVerdict(
        agent_name="Product Manager",
        role="pm",
        decision=Decision.PAUSE,
        confidence=0.65,
        rationale=(
            f"After reviewing the Risk/Critic's challenge, I acknowledge the KI-001 data-loss "
            f"concern is more urgent than my initial framing suggested. The 28.5% funnel adoption "
            f"IS a double-edged sword — high engagement means more users exposed to the bug. "
            f"However, I maintain Pause over Roll Back: the Day 7 hotfix shows the engineering "
            f"team can ship targeted fixes, and rolling back would lose the adoption signal and "
            f"create user confusion. My confidence is lower (65% vs 72%) because the D7 retention "
            f"decline ({latest_d7}% vs {criteria['min_retention_d7_pct']}% target) combined with "
            f"the unresolved KI-001 makes the risk profile worse than I initially weighted. "
            f"I now prioritise KI-001 fix as the #1 gate — above the opt-out toggle."
        ),
        key_evidence=[
            f"Adjusted: KI-001 data-loss risk elevated — 130k users exposed with no confirmed fix",
            f"D7 retention: {latest_d7}% still declining — cannot confirm feature-driven vs performance-driven",
            f"Funnel adoption 28.5% is strong but increases blast radius for data-loss bug",
            f"Data Analyst's evidence on 3/9 metric breaches reinforces caution",
        ],
        recommended_actions=[
            "Freeze rollout at 25% — KI-001 fix is now the #1 gate (above opt-out toggle).",
            "Mandate KI-001 fix within 48h as a hard prerequisite for maintaining current rollout.",
            "If KI-001 is not fixed in 48h, escalate to Roll Back discussion.",
            "Run a D7 retention deep-dive to separate performance-driven churn from feature-driven churn.",
        ],
    )

    data_revised = AgentVerdict(
        agent_name="Data Analyst",
        role="data_analyst",
        decision=Decision.PAUSE,
        confidence=0.80,
        rationale=(
            f"The Risk/Critic's challenge on KI-001 is valid — data-loss is a non-metric risk "
            f"that my quantitative analysis cannot fully capture. However, the metric trends "
            f"support Pause, not Roll Back: crash rate is on a clear downward trajectory "
            f"({peak_crash}% → {latest_crash}%), latency is within target ({latest_latency}ms "
            f"< {criteria['max_p95_latency_ms']}ms), and payment success has recovered to "
            f"{latest['payment_success_pct']}%. The critique that 'improving ≠ acceptable' is "
            f"fair — I've slightly lowered my confidence to 80%. The 3 breaching metrics "
            f"(crash rate, D7 retention, support tickets) need more time to confirm sustained "
            f"recovery. I agree with the PM that KI-001 should be the #1 gate."
        ),
        key_evidence=[
            f"Crash rate trajectory: {peak_crash}% (Day 6) → {latest_crash}% (Day 10) — improving but {latest_crash}% > {criteria['max_crash_rate_pct']}%",
            f"Latency: {latest_latency}ms — now within {criteria['max_p95_latency_ms']}ms target",
            f"Payment success: {latest['payment_success_pct']}% — recovered above {criteria['min_payment_success_pct']}% threshold",
            f"D7 retention: {latest_d7}% — still declining, need Day 3-4 cohort data (Days 10-11)",
            f"Risk/Critic's point accepted: KI-001 is a qualitative risk not captured in my metrics",
        ],
        recommended_actions=[
            "Hold at 25% rollout until crash rate < 0.75% for 3 consecutive days.",
            "Elevate KI-001 fix to P0 gate — no expansion until fix is confirmed.",
            "Set automated monitoring: all 9 metrics must be within thresholds for 48h before expansion.",
            "Track D7 retention for Day 3-4 cohort maturing on Days 10-11.",
        ],
    )

    marketing_revised = AgentVerdict(
        agent_name="Marketing & Comms",
        role="marketing_comms",
        decision=Decision.PAUSE,
        confidence=0.62,
        rationale=(
            f"The Risk/Critic's emphasis on the Reddit data-loss PSA and its amplification "
            f"potential has shifted my assessment. I underweighted the brand damage of a viral "
            f"data-loss thread — it IS more damaging than performance complaints. However, I "
            f"still favour Pause over Roll Back because: (1) rolling back generates its own "
            f"negative narrative ('they had to pull the feature'), (2) the accessibility user "
            f"(#35) represents a powerful positive story we'd lose, and (3) transparent "
            f"acknowledgement of the issue can convert critics into advocates if we fix fast. "
            f"My confidence has dropped from 70% to 62% — if KI-001 is not fixed within 48h, "
            f"I would shift to Roll Back to protect the brand."
        ),
        key_evidence=[
            f"Revised: Reddit data-loss PSA is higher risk than initially assessed — viral potential",
            f"Rollback has its own brand cost: 'they had to pull the feature' narrative",
            f"Accessibility user (#35) represents key positive narrative worth preserving",
            f"Post-hotfix sentiment (Days 8-10) is trending positive — fragile but real",
        ],
        recommended_actions=[
            "Issue transparent status update WITHIN 24h: acknowledge data-loss bug, performance issues, and opt-out gap.",
            "Respond personally to Twitter auto-complete complaints with empathy and DM follow-up.",
            "If KI-001 is not fixed in 48h, recommend shift to Roll Back to protect brand.",
            "Prepare dual comms plans: (a) 'we paused and fixed it' and (b) 'we rolled back to protect you'.",
            "Monitor Reddit thread on draft loss — if it trends, issue official response immediately.",
        ],
        dissent_notes=(
            "If KI-001 is not fixed within 48h, I would shift my recommendation to Roll Back. "
            "The brand cost of a prolonged data-loss exposure outweighs the brand cost of pulling "
            "the feature and re-launching cleanly."
        ),
    )

    revised_verdicts = [pm_revised, data_revised, marketing_revised]

    # ════════════════════════════════════════════════════════════════════
    #  PHASE 3 — Director / Senior PM synthesis
    # ════════════════════════════════════════════════════════════════════

    outcome = WarRoomOutcome(
        final_decision=Decision.PAUSE,
        decision_rationale=(
            "All three domain agents maintain Pause after deliberation, though with lower "
            "confidence (PM: 72%→65%, Data: 82%→80%, Marketing: 70%→62%). The Risk/Critic "
            "recommends Roll Back (68% confidence) citing KI-001 data-loss exposure. "
            "The strongest argument against Pause is the Risk/Critic's point that ~130,000 users "
            "remain exposed to a data-loss bug with no confirmed fix — this is a valid concern. "
            "However, the agents' revised positions converge on a conditional Pause with a hard "
            "48h gate on KI-001: if the fix is not shipped in 48h, the decision escalates to "
            "Roll Back. This conditional approach preserves the 28.5% adoption signal and avoids "
            "the brand cost of a full rollback, while setting a firm deadline that addresses the "
            "Risk/Critic's core concern. The deliberation shifted all agents to treat KI-001 as "
            "the #1 priority — above the opt-out toggle and expansion planning."
        ),
        initial_verdicts=initial_verdicts,
        critique=critique,
        revised_verdicts=revised_verdicts,
        action_plan=[
            "IMMEDIATE: Freeze rollout at 25% — block scheduled expansion to 50%.",
            "IMMEDIATE (P0): Assign 2 engineers to KI-001 (draft loss race condition); hard 48h deadline.",
            "WITHIN 24h: Add server-side draft backup as defence-in-depth for data loss.",
            "WITHIN 24h: Publish transparent status update across all channels (blog, in-app banner, social).",
            "WITHIN 48h: Ship KI-001 fix and verify with targeted regression tests.",
            "ESCALATION GATE: If KI-001 is NOT fixed by 48h, reconvene war room to decide Roll Back.",
            "WITHIN 48h: Run inference API load test simulating 50% user traffic.",
            "WITHIN 4 DAYS: Ship user-facing opt-out toggle (KI-004, fast-tracked from v4.12.1).",
            "WITHIN 1 WEEK: Conduct blameless post-incident review of the Day 5-6 degradation.",
            "RESUME GATE: Expand to 50% ONLY when ALL of these hold for 72h: "
            "crash_rate < 0.75%, p95_latency < 300ms, KI-001 fix confirmed, opt-out toggle live.",
        ],
        risks_and_mitigations=[
            "Risk: KI-001 data loss affects more users during pause → Mitigation: Server-side draft backup within 24h + P0 fix in 48h + escalation to Roll Back if missed.",
            "Risk: Crash rate re-spikes when expanding past 25% → Mitigation: Load test at 50% before expansion; automated rollback trigger at 1.5%.",
            "Risk: Negative sentiment compounds (Reddit PSA, app-store 1-stars) → Mitigation: Transparent comms within 24h + personal outreach to vocal critics.",
            "Risk: D7 retention continues declining → Mitigation: Deep-dive analysis separating performance-churn from feature-churn; opt-out toggle unblocks users who don't want the feature.",
            "Risk: Pause drags beyond 1 week, losing launch momentum → Mitigation: Hard 48h escalation gate on KI-001; clear 72h resume criteria with daily war-room check-ins.",
            "Risk: Payment success rate degrades again under load → Mitigation: Isolate payment service from inference API; independent scaling policy.",
        ],
        follow_up_monitoring=[
            "Crash rate — target: < 0.75% for 72 consecutive hours",
            "P95 latency — target: < 300ms for 72 consecutive hours",
            "D1 retention — target: ≥ 68% (currently at threshold)",
            "D7 retention — target: ≥ 42% (track Day 3-4 cohort maturing on Days 10-11)",
            "Support ticket volume — target: ≤ 80/day per 5% cohort-equivalent",
            "Payment success rate — target: ≥ 99.0%",
            "KI-001 recurrence — zero tolerance after fix deployment",
            "Churn/cancellations — target: ≤ 15/day (currently 31/day, baseline 10/day)",
            "Social media sentiment — daily scan; escalate if negative trend resumes",
        ],
        dissenting_opinions=[
            "Risk/Critic (Roll Back, 68% confidence): KI-001 (draft data loss) is still in "
            "'investigating' status after 10 days and has been reported by 3 users with Reddit "
            "amplification. At 25% rollout, ~130,000 users are exposed to a data-loss bug with "
            "no confirmed fix. The recovery trend is only 3 days old and untested at higher "
            "scale. Pausing still leaves users in a degraded state — a full rollback protects "
            "user trust while the team ships fixes and re-launches cleanly.",
            "Marketing & Comms (conditional): If KI-001 is not fixed within 48h, would shift "
            "recommendation to Roll Back. The brand cost of prolonged data-loss exposure "
            "outweighs the brand cost of pulling the feature.",
        ],
    )

    return outcome


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="War Room — Launch Decision System")
    parser.add_argument(
        "--model", default="gpt-4o", help="LLM model name (default: gpt-4o)"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run a deterministic offline simulation (no LLM / no API key needed)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also dump the full outcome as JSON to war_room_outcome.json",
    )
    args = parser.parse_args()

    dashboard = get_dashboard_snapshot()

    if args.offline:
        outcome = _run_offline(dashboard)
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY environment variable not set.")
            print("       Set it or use --offline for a demo run.")
            sys.exit(1)
        from openai import OpenAI
        from war_room.orchestrator import WarRoom

        client = OpenAI(api_key=api_key)
        war_room = WarRoom(client=client, model=args.model)
        outcome = war_room.run(dashboard)

    _print_outcome(outcome)

    if args.json:
        out_path = "war_room_outcome.json"
        # Always include tool analysis in JSON output
        tool_analysis = {
            "aggregate_metrics": aggregate_metrics(dashboard),
            "detect_anomalies": detect_anomalies(dashboard),
            "summarize_sentiment": summarize_sentiment(dashboard),
            "compare_trends": compare_trends(dashboard),
        }
        payload = {
            "final_decision": outcome.final_decision.value,
            "decision_rationale": outcome.decision_rationale,
            "tool_analysis": tool_analysis,
            "initial_verdicts": [_verdict_to_dict(v) for v in outcome.initial_verdicts],
            "critique": _verdict_to_dict(outcome.critique),
            "revised_verdicts": [_verdict_to_dict(v) for v in outcome.revised_verdicts],
            "action_plan": outcome.action_plan,
            "risks_and_mitigations": outcome.risks_and_mitigations,
            "follow_up_monitoring": outcome.follow_up_monitoring,
            "dissenting_opinions": outcome.dissenting_opinions,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n  📄 Full outcome written to {out_path}")


if __name__ == "__main__":
    main()
