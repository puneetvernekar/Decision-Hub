# War Room -- multi-agent launch decision system

import argparse
import json
import os
import sys
import textwrap

from war_room.mock_dashboard import get_dashboard_snapshot
from war_room.models import Decision
from war_room.trace import get_trace

W = 64  # print width
RESET = "\033[0m"
COLORS = {
    Decision.PROCEED: "\033[92m",
    Decision.PAUSE: "\033[93m",
    Decision.ROLL_BACK: "\033[91m",
}


def print_section(title):
    print("─" * W)
    print(f"  {title}")
    print("─" * W)


def print_items(items):
    for item in items:
        print(f"  - {item}")
    print()


def print_outcome(outcome):
    color = COLORS.get(outcome.final_decision, "")

    print("\n" + "=" * W)
    print("  FINAL WAR-ROOM DECISION")
    print("=" * W)
    print(f"\n  Decision:    {color}{outcome.final_decision.value}{RESET}")
    print(f"  Confidence:  {outcome.confidence_score:.0%}\n")
    print(textwrap.fill(outcome.decision_rationale, W,
                        initial_indent="    ", subsequent_indent="    "))
    print()

    if outcome.confidence_drivers:
        print_section("Confidence Drivers")
        print_items(outcome.confidence_drivers)

    # phase 1
    print_section("Phase 1 -- Initial Verdicts")
    for v in outcome.initial_verdicts:
        print(f"  - {v.agent_name:22s} -> {v.decision.value:10s} ({v.confidence:.0%})")
    print()

    # phase 2a
    print_section("Phase 2a -- Risk/Critic Challenge")
    c = outcome.critique
    print(f"  - {c.agent_name:22s} -> {c.decision.value:10s} ({c.confidence:.0%})")
    print(textwrap.fill(c.rationale[:200], W - 4,
                        initial_indent="    ", subsequent_indent="    "))
    print()

    # phase 2b
    print_section("Phase 2b -- Revised Verdicts")
    for v in outcome.revised_verdicts:
        init = next((i for i in outcome.initial_verdicts if i.role == v.role), None)
        change = ""
        if init and (init.decision != v.decision or abs(init.confidence - v.confidence) > 0.03):
            change = f"  (was {init.decision.value} {init.confidence:.0%})"
        print(f"  - {v.agent_name:22s} -> {v.decision.value:10s} ({v.confidence:.0%}){change}")
    print()

    # action plan
    print_section("Action Plan")
    for i, step in enumerate(outcome.action_plan, 1):
        if isinstance(step, dict):
            print(f"  {i}. [{step.get('timeframe', '')}] {step.get('action', '')}")
            print(f"     Owner: {step.get('owner', '')}")
        else:
            print(f"  {i}. {step}")
    print()

    # risks
    print_section("Risk Register")
    for item in outcome.risks_and_mitigations:
        if isinstance(item, dict):
            print(f"  - {item.get('risk', '')}")
            print(f"    {item.get('likelihood', '?')} likelihood / {item.get('impact', '?')} impact")
            print(f"    Mitigation: {item.get('mitigation', '')}")
        else:
            print(f"  - {item}")
    print()

    # comms plan
    if outcome.communication_plan:
        print_section("Comms -- Internal")
        print_items(outcome.communication_plan.get("internal", []))
        print_section("Comms -- External")
        print_items(outcome.communication_plan.get("external", []))

    print_section("Follow-up Monitoring")
    print_items(outcome.follow_up_monitoring)

    if outcome.dissenting_opinions:
        print_section("Dissenting Opinions")
        print_items(outcome.dissenting_opinions)

    # trace
    trace_log = get_trace()
    if trace_log:
        print_section(f"Execution Trace ({len(trace_log)} steps)")
        for e in trace_log:
            print(f"  [{e['step']:>3}] +{e['elapsed_s']:6.2f}s  "
                  f"{e['source']:20s}  {e['event']:12s}  {e['detail']}")
        print()

    print("=" * W)


def verdict_to_dict(v):
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


def main():
    # load .env if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    model = os.environ.get("LLM_MODEL", "gpt-4o")
    max_parallel = int(os.environ.get("MAX_PARALLEL_AGENTS", "3"))
    output_path = os.environ.get("OUTPUT_JSON_PATH", "decision_hub_output.json")

    parser = argparse.ArgumentParser(description="War Room -- Launch Decision System")
    parser.add_argument("--model", default=model, help=f"LLM model (default: {model})")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL") or None
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    from openai import OpenAI
    from war_room.orchestrator import WarRoom

    client = OpenAI(api_key=api_key, base_url=base_url)
    war_room = WarRoom(client=client, model=args.model, max_parallel=max_parallel)

    dashboard = get_dashboard_snapshot()
    outcome = war_room.run(dashboard)

    print_outcome(outcome)

    # write json output
    payload = {
        "final_decision": outcome.final_decision.value,
        "decision_rationale": outcome.decision_rationale,
        "confidence_score": outcome.confidence_score,
        "confidence_drivers": outcome.confidence_drivers,
        "final_verdicts": [verdict_to_dict(v) for v in outcome.revised_verdicts],
        "action_plan": outcome.action_plan,
        "risks_and_mitigations": outcome.risks_and_mitigations,
        "communication_plan": outcome.communication_plan,
        "follow_up_monitoring": outcome.follow_up_monitoring,
        "dissenting_opinions": outcome.dissenting_opinions,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Output written to {output_path}")


if __name__ == "__main__":
    main()
