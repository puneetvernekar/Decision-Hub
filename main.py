"""
War Room — Multi-Agent Launch Decision System
==============================================
Run:  python main.py [--model MODEL] [--json]
"""

import argparse
import json
import os
import sys
import textwrap

from war_room.mock_dashboard import get_dashboard_snapshot
from war_room.models import AgentVerdict, Decision, WarRoomOutcome
from war_room.trace import get_trace


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
    print(f"\n  Decision:    {color}{outcome.final_decision.value}{RESET}")
    print(f"  Confidence:  {outcome.confidence_score:.0%}\n")
    print(f"  Rationale:\n{textwrap.fill(outcome.decision_rationale, W, initial_indent='    ', subsequent_indent='    ')}\n")

    # Confidence drivers
    if outcome.confidence_drivers:
        print("─" * W)
        print("  Confidence Drivers")
        print("─" * W)
        for item in outcome.confidence_drivers:
            print(f"  • {item}")
        print()

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

    # Action plan (structured with owners)
    print("─" * W)
    print("  Action Plan")
    print("─" * W)
    for i, step in enumerate(outcome.action_plan, 1):
        if isinstance(step, dict):
            tf = step.get("timeframe", "")
            owner = step.get("owner", "")
            action = step.get("action", "")
            print(f"  {i:2d}. [{tf}] {action}")
            print(f"      Owner: {owner}")
        else:
            print(f"  {i:2d}. {step}")
    print()

    # Risk register (structured)
    print("─" * W)
    print("  Risk Register")
    print("─" * W)
    for item in outcome.risks_and_mitigations:
        if isinstance(item, dict):
            risk = item.get("risk", "")
            lh = item.get("likelihood", "?")
            imp = item.get("impact", "?")
            mit = item.get("mitigation", "")
            print(f"  • {risk}")
            print(f"    Likelihood: {lh}  |  Impact: {imp}")
            print(f"    Mitigation: {mit}")
        else:
            print(f"  • {item}")
    print()

    # Communication plan
    if outcome.communication_plan:
        print("─" * W)
        print("  Communication Plan — Internal")
        print("─" * W)
        for item in outcome.communication_plan.get("internal", []):
            print(f"  • {item}")
        print()
        print("─" * W)
        print("  Communication Plan — External")
        print("─" * W)
        for item in outcome.communication_plan.get("external", []):
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

    # Trace summary
    trace_log = get_trace()
    if trace_log:
        print("─" * W)
        print(f"  Execution Trace ({len(trace_log)} steps)")
        print("─" * W)
        for entry in trace_log:
            print(f"  [{entry['step']:>3}] +{entry['elapsed_s']:6.2f}s  "
                  f"{entry['source']:20s}  {entry['event']:12s}  {entry['detail']}")
        print()

    print("=" * W)


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load .env file (if present) before reading any env vars
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed — fall back to raw env vars

    # Defaults come from .env → env vars → CLI overrides
    env_model = os.environ.get("LLM_MODEL", "gpt-4o")
    env_parallel = int(os.environ.get("MAX_PARALLEL_AGENTS", "3"))
    env_output = os.environ.get("OUTPUT_JSON_PATH", "war_room_outcome.json")

    parser = argparse.ArgumentParser(description="War Room — Launch Decision System")
    parser.add_argument(
        "--model", default=env_model,
        help=f"LLM model name (default from .env: {env_model})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also dump the full outcome as JSON to the output path",
    )
    args = parser.parse_args()

    dashboard = get_dashboard_snapshot()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL") or None
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        print("       Add it to .env or export it.")
        sys.exit(1)
    from openai import OpenAI
    from war_room.orchestrator import WarRoom

    client = OpenAI(api_key=api_key, base_url=base_url)
    war_room = WarRoom(client=client, model=args.model, max_parallel=env_parallel)
    outcome = war_room.run(dashboard)

    _print_outcome(outcome)

    if args.json:
        out_path = env_output
        payload = {
            "final_decision": outcome.final_decision.value,
            "decision_rationale": outcome.decision_rationale,
            "confidence_score": outcome.confidence_score,
            "confidence_drivers": outcome.confidence_drivers,
            "final_verdicts": [_verdict_to_dict(v) for v in outcome.revised_verdicts],
            "action_plan": outcome.action_plan,
            "risks_and_mitigations": outcome.risks_and_mitigations,
            "communication_plan": outcome.communication_plan,
            "follow_up_monitoring": outcome.follow_up_monitoring,
            "dissenting_opinions": outcome.dissenting_opinions,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n  📄 Full outcome written to {out_path}")


if __name__ == "__main__":
    main()
