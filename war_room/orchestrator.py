"""
War-room orchestrator.

Coordinates all agents, collects individual verdicts, runs a synthesis
round, and produces the final WarRoomOutcome.
"""

from __future__ import annotations

import json
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from openai import OpenAI

from .agents import ALL_AGENTS, BaseAgent
from .models import AgentVerdict, Decision, WarRoomOutcome


class WarRoom:
    """Orchestrates the multi-agent war-room session."""

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o",
        agent_classes: Optional[list[type[BaseAgent]]] = None,
        max_parallel: int = 5,
    ):
        self.client = client
        self.model = model
        self.agent_classes = agent_classes or ALL_AGENTS
        self.max_parallel = max_parallel

    # ── Phase 1: Individual agent analysis ──────────────────────────────

    def _run_agents(self, dashboard: dict) -> list[AgentVerdict]:
        """Run all agents in parallel and collect their verdicts."""
        agents = [cls(self.client, self.model) for cls in self.agent_classes]
        verdicts: list[AgentVerdict] = []

        with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            futures = {pool.submit(agent.analyze, dashboard): agent for agent in agents}
            for future in as_completed(futures):
                agent = futures[future]
                try:
                    verdict = future.result()
                    verdicts.append(verdict)
                except Exception as exc:
                    print(f"  ⚠  {agent.name} failed: {exc}")

        return verdicts

    # ── Phase 2: Critique round ─────────────────────────────────────────

    def _critique_round(
        self, dashboard: dict, verdicts: list[AgentVerdict]
    ) -> str:
        """Ask the Risk/Critic agent to review all other verdicts."""
        verdicts_summary = "\n\n".join(
            f"**{v.agent_name}** ({v.role}) → {v.decision.value} "
            f"(confidence {v.confidence:.0%})\n"
            f"Rationale: {v.rationale}\n"
            f"Evidence: {'; '.join(v.key_evidence)}"
            for v in verdicts
        )
        prompt = textwrap.dedent(f"""\
            You are the Risk Analyst in a product-launch war room.
            All agents have submitted their individual verdicts.

            Here are the verdicts:
            {verdicts_summary}

            Dashboard data:
            ```json
            {json.dumps(dashboard, indent=2)}
            ```

            Review the verdicts critically. Identify:
            1. Agreements and disagreements between agents.
            2. Blind spots or assumptions that are not backed by data.
            3. Any additional risks the group may be overlooking.

            Return your critique as plain text (2-4 paragraphs).
        """)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.4,
            messages=[
                {"role": "system", "content": "You are a senior risk analyst providing a critique round."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    # ── Phase 3: Synthesis & final decision ─────────────────────────────

    def _synthesize(
        self,
        dashboard: dict,
        verdicts: list[AgentVerdict],
        critique: str,
    ) -> WarRoomOutcome:
        """Synthesise all inputs into a final structured decision."""
        verdicts_json = json.dumps(
            [
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
                for v in verdicts
            ],
            indent=2,
        )

        schema = textwrap.dedent("""\
        {
          "final_decision": "Proceed | Pause | Roll Back",
          "decision_rationale": "...",
          "action_plan": ["step 1", "step 2", ...],
          "risks_and_mitigations": ["risk → mitigation", ...],
          "follow_up_monitoring": ["metric to watch", ...],
          "dissenting_opinions": ["...", ...]
        }""")

        prompt = textwrap.dedent(f"""\
            You are the War-Room Facilitator synthesising the final launch decision.

            ## Individual Verdicts
            ```json
            {verdicts_json}
            ```

            ## Critique Round
            {critique}

            ## Dashboard Snapshot (for reference)
            ```json
            {json.dumps(dashboard, indent=2)}
            ```

            Using all of the above, produce the FINAL war-room decision.
            Rules:
            • The decision must be exactly one of: Proceed, Pause, Roll Back.
            • Weight higher-confidence verdicts and data-backed reasoning more.
            • If agents are evenly split, lean toward caution (Pause > Proceed).
            • The action plan must be concrete, sequenced, and assignable.
            • Capture any dissenting opinions faithfully.

            Respond ONLY with a JSON object matching this schema:
            {schema}
        """)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert facilitator making the final launch decision for the war room.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        raw = response.choices[0].message.content
        start = raw.find("{")
        end = raw.rfind("}") + 1
        payload = json.loads(raw[start:end])

        return WarRoomOutcome(
            final_decision=Decision(payload["final_decision"]),
            decision_rationale=payload["decision_rationale"],
            individual_verdicts=verdicts,
            action_plan=payload["action_plan"],
            risks_and_mitigations=payload["risks_and_mitigations"],
            follow_up_monitoring=payload["follow_up_monitoring"],
            dissenting_opinions=payload.get("dissenting_opinions", []),
        )

    # ── Public API ──────────────────────────────────────────────────────

    def run(self, dashboard: dict, verbose: bool = True) -> WarRoomOutcome:
        """Execute the full war-room session and return the outcome."""
        if verbose:
            print("=" * 60)
            print("  WAR ROOM SESSION — Launch Decision")
            print("=" * 60)
            print(f"\n  Feature : {dashboard.get('feature_name', 'N/A')}")
            print(f"  Rollout : {dashboard.get('current_rollout_percentage', '?')}% "
                  f"({dashboard.get('rollout_schedule', '')})")
            print(f"  Window  : {len(dashboard.get('daily_metrics', []))} days of data\n")

        # Phase 1
        if verbose:
            print("─" * 60)
            print("  Phase 1 — Individual Agent Analysis")
            print("─" * 60)
        verdicts = self._run_agents(dashboard)
        if verbose:
            for v in verdicts:
                print(f"\n  [{v.agent_name}]  →  {v.decision.value}  "
                      f"(confidence: {v.confidence:.0%})")
                print(f"    Rationale: {v.rationale[:120]}...")

        # Phase 2
        if verbose:
            print(f"\n{'─' * 60}")
            print("  Phase 2 — Critique Round")
            print("─" * 60)
        critique = self._critique_round(dashboard, verdicts)
        if verbose:
            print(f"\n{critique[:500]}...\n" if len(critique) > 500 else f"\n{critique}\n")

        # Phase 3
        if verbose:
            print("─" * 60)
            print("  Phase 3 — Synthesis & Final Decision")
            print("─" * 60)
        outcome = self._synthesize(dashboard, verdicts, critique)
        return outcome
