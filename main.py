"""
War Room — Multi-Agent Launch Decision System
==============================================
Run:  python main.py [--model MODEL] [--offline]

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


# ── Pretty-print helpers ────────────────────────────────────────────────────

def _print_outcome(outcome: WarRoomOutcome) -> None:
    W = 64
    print("\n" + "=" * W)
    print("  FINAL WAR-ROOM DECISION")
    print("=" * W)

    decision_color = {
        Decision.PROCEED: "\033[92m",   # green
        Decision.PAUSE: "\033[93m",     # yellow
        Decision.ROLL_BACK: "\033[91m", # red
    }
    RESET = "\033[0m"
    color = decision_color.get(outcome.final_decision, "")
    print(f"\n  Decision:  {color}{outcome.final_decision.value}{RESET}\n")
    print(f"  Rationale:\n{textwrap.fill(outcome.decision_rationale, W, initial_indent='    ', subsequent_indent='    ')}\n")

    print("─" * W)
    print("  Individual Verdicts")
    print("─" * W)
    for v in outcome.individual_verdicts:
        print(f"  • {v.agent_name:22s} → {v.decision.value:10s} (conf: {v.confidence:.0%})")
    print()

    print("─" * W)
    print("  Action Plan")
    print("─" * W)
    for i, step in enumerate(outcome.action_plan, 1):
        print(f"  {i}. {step}")
    print()

    print("─" * W)
    print("  Risks & Mitigations")
    print("─" * W)
    for item in outcome.risks_and_mitigations:
        print(f"  • {item}")
    print()

    print("─" * W)
    print("  Follow-up Monitoring")
    print("─" * W)
    for item in outcome.follow_up_monitoring:
        print(f"  • {item}")
    print()

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

    # ── Extract data ────────────────────────────────────────────────────
    daily = dashboard["daily_metrics"]
    baseline = dashboard["baseline_metrics"]
    criteria = dashboard["success_criteria"]
    known_issues = dashboard["known_issues"]

    latest = daily[-1]  # Day 10
    n_days = len(daily)

    # Derive aggregates from daily metrics
    peak_crash = max(d["crash_rate_pct"] for d in daily)
    peak_latency = max(d["p95_latency_ms"] for d in daily)
    peak_tickets = max(d["support_tickets"] for d in daily)
    peak_churn = max(d["churn_cancellations"] for d in daily)
    total_tickets = sum(d["support_tickets"] for d in daily)
    total_cancellations = sum(d["churn_cancellations"] for d in daily)
    avg_churn_per_day = total_cancellations / n_days
    latest_crash = latest["crash_rate_pct"]
    latest_latency = latest["p95_latency_ms"]

    # D7 retention (available from Day 7 onward)
    d7_values = [d["retention_d7_pct"] for d in daily if d["retention_d7_pct"] is not None]
    latest_d7 = d7_values[-1] if d7_values else None

    # Recovery trend: compare Day 10 vs Day 6 (peak trouble)
    recovering = (
        daily[-1]["crash_rate_pct"] < daily[5]["crash_rate_pct"]
        and daily[-1]["p95_latency_ms"] < daily[5]["p95_latency_ms"]
    )

    # Feedback analysis
    feedback = dashboard["user_feedback"]
    neg_count = sum(1 for f in feedback if f["sentiment"] == "negative")
    pos_count = sum(1 for f in feedback if f["sentiment"] == "positive")
    neu_count = len(feedback) - neg_count - pos_count

    # Critical known issues still open
    critical_open = [ki for ki in known_issues if ki["severity"] == "critical" and "fix" not in ki["status"]]

    # ── Simulated agent verdicts ────────────────────────────────────────

    pm_verdict = AgentVerdict(
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

    data_verdict = AgentVerdict(
        agent_name="Data Analyst",
        role="data_analyst",
        decision=Decision.PAUSE,
        confidence=0.82,
        rationale=(
            f"Crash rate peaked at {peak_crash}% on Day 6 ({peak_crash/baseline['crash_rate_pct']:.1f}x "
            f"baseline, threshold {criteria['max_crash_rate_pct']}%). Post-hotfix recovery is clear: "
            f"Day 10 crash rate {latest_crash}% is {'above' if latest_crash > criteria['max_crash_rate_pct'] else 'at'} "
            f"the {criteria['max_crash_rate_pct']}% threshold. P95 latency peaked at {peak_latency}ms "
            f"(Day 6), now {latest_latency}ms — {'within' if latest_latency <= criteria['max_p95_latency_ms'] else 'still above'} "
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

    marketing_verdict = AgentVerdict(
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
            f"Sentiment is skewed negative in the Days 4-6 window. "
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
            f"App store: polarised 1-star / 5-star reviews",
        ],
        recommended_actions=[
            "Issue transparent status update: acknowledge performance issues, draft-loss bug, and opt-out gap.",
            "Respond personally to the Twitter auto-complete complaints with empathy and DM follow-up.",
            "Prepare accessibility-focused case study with user #35's permission — powerful positive narrative.",
            "Do NOT run any promotional campaigns for Smart Compose until opt-out toggle ships.",
            "Monitor Reddit thread on draft loss — if it trends, issue an official response.",
        ],
    )

    risk_verdict = AgentVerdict(
        agent_name="Risk / Critic",
        role="risk_critic",
        decision=Decision.ROLL_BACK,
        confidence=0.68,
        rationale=(
            f"KI-001 (draft data loss) remains in 'investigating' status after 10 days — this is a "
            f"critical severity bug that has been reported by 3 separate users and amplified on Reddit. "
            f"At 25% rollout ({dashboard['total_user_base'] * 25 // 100:,} users exposed), the blast "
            f"radius for a data-loss bug is unacceptable. "
            f"Cancellations averaged {avg_churn_per_day:.0f}/day over 10 days vs {baseline['avg_churn_cancellations_per_day']}/day "
            f"baseline — a {avg_churn_per_day / baseline['avg_churn_cancellations_per_day']:.1f}x increase that "
            f"would scale further at full rollout. D7 retention at {latest_d7}% is below the "
            f"{criteria['min_retention_d7_pct']}% threshold — the feature may be driving users away. "
            f"The Day 7 hotfix improved latency but crash rate ({latest_crash}%) still exceeds the "
            f"{criteria['max_crash_rate_pct']}% threshold. Extrapolating to 50% rollout carries "
            f"infrastructure risk given KI-002's partial fix status. The team may be anchored on "
            f"the recovery trend rather than the absolute breach of success criteria."
        ),
        key_evidence=[
            f"KI-001 (critical): draft data loss — still 'investigating' after 10 days",
            f"Churn: {total_cancellations} cancellations in {n_days} days ({avg_churn_per_day:.0f}/day vs {baseline['avg_churn_cancellations_per_day']}/day baseline)",
            f"D7 retention: {latest_d7}% < {criteria['min_retention_d7_pct']}% target",
            f"Crash rate Day 10: {latest_crash}% — still above {criteria['max_crash_rate_pct']}% threshold",
            f"Payment success dip to {min(d['payment_success_pct'] for d in daily)}% signals infrastructure fragility",
            f"3 of 9 metrics still breaching thresholds on Day 10",
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

    eng_verdict = AgentVerdict(
        agent_name="Engineering Lead",
        role="engineering_lead",
        decision=Decision.PAUSE,
        confidence=0.76,
        rationale=(
            f"The Day 7 hotfix materially improved performance: latency dropped from "
            f"{peak_latency}ms → {latest_latency}ms ({latest_latency / baseline['p95_latency_ms']:.1f}x "
            f"baseline, {'within' if latest_latency <= criteria['max_p95_latency_ms'] else 'near'} the "
            f"{criteria['max_p95_latency_ms']}ms target). Crash rate improved from {peak_crash}% → "
            f"{latest_crash}% but still exceeds the {criteria['max_crash_rate_pct']}% threshold. "
            f"KI-001 (draft loss race condition) is the highest-priority item — root cause is identified "
            f"(auto-save debounce vs suggestion overlay teardown) and a fix is feasible in 24-48h. "
            f"KI-002 (latency scaling) hotfix is deployed; further tuning requires load testing at 50% "
            f"traffic before expansion. KI-004 (opt-out toggle) is ~4 engineering days from ship. "
            f"Full rollback is costly (~6 engineer-hours + deployment risk + user confusion) and "
            f"would lose the 28.5% adoption signal. A pause + targeted fixes is the better path."
        ),
        key_evidence=[
            f"Latency: {peak_latency}ms (Day 6) → {latest_latency}ms (Day 10) — hotfix working",
            f"Crash rate: {peak_crash}% → {latest_crash}% — improving but still above {criteria['max_crash_rate_pct']}%",
            "KI-001 root cause identified: race condition in auto-save debounce — fix ETA 24-48h",
            "KI-002 hotfix deployed Day 7; needs load test at 50% before expansion",
            "KI-004 opt-out toggle: ~4 engineering days (aligns with v4.12.1 plan)",
            f"Rollback cost: ~6 engineer-hours + deployment risk + loss of {latest['feature_funnel_completion_pct']}% funnel adoption cohort",
            f"Payment success recovered: {latest['payment_success_pct']}% (correlated with latency fix)",
        ],
        recommended_actions=[
            "P0: Fix KI-001 (draft loss race condition) within 48h — add mutex on auto-save during overlay transitions.",
            "P1: Run load test simulating 50% traffic against inference API before any expansion.",
            "P1: Ship opt-out toggle (KI-004) — fast-track from v4.12.1 plan.",
            "Freeze rollout at 25% with automated gate: crash_rate < 0.75% AND latency < 300ms for 72h.",
            "Add server-side draft backup as defence-in-depth against future data-loss regressions.",
        ],
    )

    verdicts = [pm_verdict, data_verdict, marketing_verdict, risk_verdict, eng_verdict]

    # ── Synthesise ──────────────────────────────────────────────────────
    # Tally: 4 Pause, 1 Roll Back → Pause wins

    outcome = WarRoomOutcome(
        final_decision=Decision.PAUSE,
        decision_rationale=(
            "Four of five agents recommend Pause; one (Risk/Critic) recommends Roll Back. "
            "The 10-day data shows a clear pattern: a significant degradation on Days 5-6, "
            "followed by partial recovery after the Day 7 hotfix. By Day 10, crash rate "
            f"({latest_crash}%) and latency ({latest_latency}ms) are trending toward thresholds "
            f"but have not yet consistently met them. Three of nine metrics still breach success "
            f"criteria. The critical draft-loss bug (KI-001) remains open and poses unacceptable "
            f"data-integrity risk. Feature funnel completion ({latest['feature_funnel_completion_pct']}%) and "
            f"funnel completion ({latest['feature_funnel_completion_pct']}%) are strong positive "
            f"signals that argue against a full rollback. Pausing at 25% limits the blast radius "
            f"to ~{dashboard['total_user_base'] * 25 // 100:,} users while preserving the ability "
            f"to resume quickly. The pause is conditional on shipping the KI-001 fix and opt-out "
            f"toggle before any further expansion."
        ),
        individual_verdicts=verdicts,
        action_plan=[
            "IMMEDIATE: Freeze rollout at 25% — block scheduled expansion to 50%.",
            "IMMEDIATE (P0): Assign 2 engineers to KI-001 (draft loss race condition); target fix within 48h.",
            "WITHIN 24h: Add server-side draft backup as defence-in-depth for data loss.",
            "WITHIN 48h: Ship KI-001 fix and verify with targeted regression tests.",
            "WITHIN 48h: Run inference API load test simulating 50% user traffic.",
            "WITHIN 4 DAYS: Ship user-facing opt-out toggle (KI-004, fast-tracked from v4.12.1).",
            "WITHIN 24h: Publish transparent status update across all channels (blog, in-app banner, social).",
            "WITHIN 1 WEEK: Conduct blameless post-incident review of the Day 5-6 degradation.",
            "GATE: Resume expansion to 50% ONLY when ALL of these are true for 72h: "
            "crash_rate < 0.75%, p95_latency < 300ms, KI-001 fix confirmed, opt-out toggle live.",
        ],
        risks_and_mitigations=[
            "Risk: KI-001 data loss affects more users during pause → Mitigation: Server-side draft backup within 24h + P0 fix in 48h.",
            "Risk: Crash rate re-spikes when expanding past 25% → Mitigation: Load test at 50% before expansion; automated rollback trigger at 1.5%.",
            "Risk: Negative sentiment compounds (Reddit PSA, app-store 1-stars) → Mitigation: Transparent comms + personal outreach to vocal critics.",
            "Risk: D7 retention continues declining → Mitigation: Deep-dive analysis separating performance-churn from feature-churn; opt-out toggle unblocks users who don't want the feature.",
            "Risk: Pause drags beyond 1 week, losing launch momentum → Mitigation: Clear 72h gate criteria with daily war-room check-ins.",
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
            "NPS and CSAT — daily tracking; resume gate requires CSAT ≥ 3.8",
            "Social media sentiment — daily scan; escalate if negative trend resumes",
        ],
        dissenting_opinions=[
            "Risk/Critic (Roll Back, 68% confidence): KI-001 (draft data loss) is still in 'investigating' "
            "status after 10 days and has been reported by 3 users with Reddit amplification. At 25% rollout, "
            "~130,000 users are exposed to a data-loss bug with no confirmed fix. The recovery trend is only "
            "3 days old and untested at higher scale. Pausing still leaves users in a degraded state — a full "
            "rollback protects user trust while the team ships fixes and re-launches cleanly.",
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
        payload = {
            "final_decision": outcome.final_decision.value,
            "decision_rationale": outcome.decision_rationale,
            "individual_verdicts": [
                {
                    "agent": v.agent_name,
                    "role": v.role,
                    "decision": v.decision.value,
                    "confidence": v.confidence,
                    "rationale": v.rationale,
                    "key_evidence": v.key_evidence,
                    "recommended_actions": v.recommended_actions,
                    "dissent_notes": v.dissent_notes,
                }
                for v in outcome.individual_verdicts
            ],
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
