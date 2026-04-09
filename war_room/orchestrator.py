"""
War-room orchestrator — 3-phase coordinated decision process.

Phase 1 — Independent analysis by PM, Data Analyst, Marketing/Comms (parallel)
Phase 2 — Risk/Critic challenges (2a) + agents revise their verdicts (2b, parallel)
Phase 3 — Director / Senior PM synthesises the final go/no-go decision
"""

import json
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed

from .agents import (
    PHASE1_AGENTS, RiskCriticAgent, BaseAgent,
    format_verdicts_summary,
)
from .models import AgentVerdict, Decision, WarRoomOutcome
from .trace import trace, reset_trace, get_trace


class WarRoom:
    """Orchestrates the multi-agent war-room session."""

    def __init__(self, client, model="gpt-4o", max_parallel=3):
        self.client = client
        self.model = model
        self.max_parallel = max_parallel

    # ── Phase 1: Independent analysis ───────────────────────────────────

    def _phase1_analyze(self, dashboard):
        """Run PM, Data Analyst, Marketing/Comms.

        Uses parallel execution when max_parallel > 1, otherwise runs
        sequentially (safer for rate-limited free-tier APIs).

        Returns the verdicts **and** agent instances (so the caller
        can inspect ``agent.last_tool_results``).
        """
        agents = [cls(self.client, self.model) for cls in PHASE1_AGENTS]
        verdicts = []
        agent_map = []

        if self.max_parallel <= 1:
            # Sequential — avoids rate-limit collisions on free tiers
            for agent in agents:
                try:
                    verdict = agent.analyze(dashboard)
                    verdicts.append(verdict)
                    agent_map.append(agent)
                except Exception as exc:
                    print(f"  ⚠  {agent.name} failed: {exc}")
        else:
            with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
                futures = {pool.submit(agent.analyze, dashboard): agent for agent in agents}
                for future in as_completed(futures):
                    agent = futures[future]
                    try:
                        verdict = future.result()
                        verdicts.append(verdict)
                        agent_map.append(agent)
                    except Exception as exc:
                        print(f"  ⚠  {agent.name} failed: {exc}")

        return verdicts, agent_map

    # ── Phase 2a: Risk/Critic challenge ─────────────────────────────────

    def _phase2a_critique(self, dashboard, verdicts):
        """Risk/Critic reviews all Phase 1 verdicts and produces a challenge.

        Returns the critique verdict **and** the agent instance.
        """
        critic = RiskCriticAgent(self.client, self.model)
        return critic.challenge(dashboard, verdicts), critic

    # ── Phase 2b: Agent revision ────────────────────────────────────────

    def _phase2b_revise(self, dashboard, initial_verdicts, critique):
        """Each Phase 1 agent revises after seeing peers + critique.

        Sequential when max_parallel <= 1, parallel otherwise.
        """
        agents = [cls(self.client, self.model) for cls in PHASE1_AGENTS]
        revised = []

        if self.max_parallel <= 1:
            for agent in agents:
                own_verdict = next(v for v in initial_verdicts if v.role == agent.role)
                peer_verdicts = [v for v in initial_verdicts if v.role != agent.role]
                try:
                    verdict = agent.revise(dashboard, own_verdict, peer_verdicts, critique)
                    revised.append(verdict)
                except Exception as exc:
                    print(f"  ⚠  {agent.name} revision failed: {exc}")
        else:
            with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
                futures = {}
                for agent in agents:
                    own_verdict = next(v for v in initial_verdicts if v.role == agent.role)
                    peer_verdicts = [v for v in initial_verdicts if v.role != agent.role]
                    futures[pool.submit(
                        agent.revise, dashboard, own_verdict, peer_verdicts, critique
                    )] = agent

                for future in as_completed(futures):
                    agent = futures[future]
                    try:
                        verdict = future.result()
                        revised.append(verdict)
                    except Exception as exc:
                        print(f"  ⚠  {agent.name} revision failed: {exc}")

        return revised

    # ── Phase 3: Director / Senior PM synthesis ─────────────────────────

    def _phase3_synthesize(self, initial_verdicts, critique, revised_verdicts):
        """Director / Senior PM makes the final go/no-go decision."""

        initial_summary = format_verdicts_summary(initial_verdicts)
        critique_summary = format_verdicts_summary([critique])
        revised_summary = format_verdicts_summary(revised_verdicts)

        # Detect who changed position during deliberation
        changes = []
        for init in initial_verdicts:
            for rev in revised_verdicts:
                if init.role == rev.role:
                    if init.decision != rev.decision or abs(init.confidence - rev.confidence) > 0.03:
                        changes.append(
                            f"{init.agent_name}: {init.decision.value} ({init.confidence:.0%}) "
                            f"→ {rev.decision.value} ({rev.confidence:.0%})"
                        )
        changes_text = "\n".join(changes) if changes else "No agents changed their position."

        schema = textwrap.dedent("""\
        {
          "final_decision": "Proceed | Pause | Roll Back",
          "decision_rationale": "... (reference specific metrics and feedback)",
          "confidence_score": 0.0-1.0,
          "confidence_drivers": [
            "What would INCREASE confidence: ...",
            "What would DECREASE confidence: ..."
          ],
          "action_plan": [
            {"action": "...", "owner": "Engineering | PM | Marketing | Support | Leadership", "timeframe": "IMMEDIATE | WITHIN 24h | WITHIN 48h | ..."}
          ],
          "risks_and_mitigations": [
            {"risk": "...", "likelihood": "high|medium|low", "impact": "high|medium|low", "mitigation": "..."}
          ],
          "communication_plan": {
            "internal": ["message / action for internal stakeholders", ...],
            "external": ["message / action for users / public", ...]
          },
          "follow_up_monitoring": ["metric to watch", ...],
          "dissenting_opinions": ["...", ...]
        }""")

        prompt = textwrap.dedent(f"""\
            You are the Director of Product / Senior PM making the final launch
            decision in a war-room session.  You have authority to make the
            go/no-go call after hearing all perspectives.

            ## Phase 1 — Initial Verdicts
            {initial_summary}

            ## Phase 2a — Risk/Critic's Challenges
            {critique_summary}

            ## Phase 2b — Revised Verdicts (after deliberation)
            {revised_summary}

            ## Position Changes During Deliberation
            {changes_text}

            Using all of the above, produce the FINAL war-room decision.
            Rules:
            • The decision must be exactly one of: Proceed, Pause, Roll Back.
            • Weight higher-confidence verdicts and data-backed reasoning more.
            • If agents are evenly split, lean toward caution (Pause > Proceed).
            • Before deciding, identify the strongest argument AGAINST the
              majority position and explain why it does or doesn't change
              your conclusion.
            • The decision_rationale MUST reference specific metric values
              and feedback themes that drove the decision.
            • confidence_score: your overall confidence in the decision (0–1).
            • confidence_drivers: list what evidence would raise or lower
              your confidence (e.g., "KI-001 fix confirmed → +0.15").
            • Each action_plan item must have action, owner, and timeframe.
            • risks_and_mitigations: structured as objects with risk,
              likelihood, impact, and mitigation.
            • communication_plan: separate internal (engineering, leadership,
              support) and external (users, press, social) messaging guidance.
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
                    "content": (
                        "You are a senior Director of Product making the final "
                        "launch decision.  You are impartial, data-driven, and "
                        "prioritise user trust and business sustainability."
                    ),
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
            confidence_score=float(payload.get("confidence_score", 0.5)),
            confidence_drivers=payload.get("confidence_drivers", []),
            initial_verdicts=initial_verdicts,
            critique=critique,
            revised_verdicts=revised_verdicts,
            action_plan=payload["action_plan"],
            risks_and_mitigations=payload["risks_and_mitigations"],
            communication_plan=payload.get("communication_plan", {"internal": [], "external": []}),
            follow_up_monitoring=payload["follow_up_monitoring"],
            dissenting_opinions=payload.get("dissenting_opinions", []),
        )

    # ── Public API ──────────────────────────────────────────────────────

    @staticmethod
    def _log_tools(agents):
        """Print which tools each agent invoked (verbose helper)."""
        for agent in agents:
            if agent.last_tool_results:
                tool_names = ", ".join(agent.last_tool_results.keys())
                print(f"    🔧 {agent.name} invoked: {tool_names}")

    def run(self, dashboard, verbose=True):
        """Execute the full 3-phase war-room session and return the outcome."""
        reset_trace()
        trace("orchestrator", "session", "War-room session started")
        if verbose:
            print("=" * 60)
            print("  WAR ROOM SESSION — Launch Decision")
            print("=" * 60)
            print(f"\n  Feature : {dashboard.get('feature_name', 'N/A')}")
            print(f"  Rollout : {dashboard.get('current_rollout_percentage', '?')}% "
                  f"({dashboard.get('rollout_schedule', '')})")
            print(f"  Window  : {len(dashboard.get('daily_metrics', []))} days of data\n")

        # Phase 1 — Independent Analysis
        trace("orchestrator", "phase_start", "Phase 1 — Independent Agent Analysis")
        if verbose:
            print("─" * 60)
            print("  Phase 1 — Independent Agent Analysis")
            print("─" * 60)
        initial_verdicts, p1_agents = self._phase1_analyze(dashboard)
        trace("orchestrator", "phase_end", f"Phase 1 complete — {len(initial_verdicts)} verdicts collected")
        if verbose:
            self._log_tools(p1_agents)
            for v in initial_verdicts:
                print(f"\n  [{v.agent_name}]  →  {v.decision.value}  "
                      f"(confidence: {v.confidence:.0%})")
                print(f"    Rationale: {v.rationale[:120]}...")

        # Phase 2a — Risk/Critic Challenge
        trace("orchestrator", "phase_start", "Phase 2a — Risk/Critic Challenge")
        if verbose:
            print(f"\n{'─' * 60}")
            print("  Phase 2a — Risk/Critic Challenge")
            print("─" * 60)
        critique, critic_agent = self._phase2a_critique(dashboard, initial_verdicts)
        trace("orchestrator", "phase_end", f"Phase 2a complete — critique: {critique.decision.value}")
        if verbose:
            self._log_tools([critic_agent])
            print(f"\n  [Risk / Critic]  →  {critique.decision.value}  "
                  f"(confidence: {critique.confidence:.0%})")
            print(f"    Rationale: {critique.rationale[:200]}...")

        # Phase 2b — Revision
        trace("orchestrator", "phase_start", "Phase 2b — Agent Revision (after deliberation)")
        if verbose:
            print(f"\n{'─' * 60}")
            print("  Phase 2b — Agent Revision (after deliberation)")
            print("─" * 60)
        revised_verdicts = self._phase2b_revise(dashboard, initial_verdicts, critique)
        trace("orchestrator", "phase_end", f"Phase 2b complete — {len(revised_verdicts)} revised verdicts")
        if verbose:
            for v in revised_verdicts:
                init = next((i for i in initial_verdicts if i.role == v.role), None)
                change = ""
                if init and (init.decision != v.decision or abs(init.confidence - v.confidence) > 0.03):
                    change = f"  ← was {init.decision.value} ({init.confidence:.0%})"
                print(f"\n  [{v.agent_name}]  →  {v.decision.value}  "
                      f"(confidence: {v.confidence:.0%}){change}")
                print(f"    Rationale: {v.rationale[:120]}...")

        # Phase 3 — Director Synthesis
        trace("orchestrator", "phase_start", "Phase 3 — Director / Senior PM Final Decision")
        if verbose:
            print(f"\n{'─' * 60}")
            print("  Phase 3 — Director / Senior PM Final Decision")
            print("─" * 60)
        outcome = self._phase3_synthesize(initial_verdicts, critique, revised_verdicts)
        trace("orchestrator", "phase_end", f"Phase 3 complete — decision: {outcome.final_decision.value}")
        trace("orchestrator", "session", "War-room session complete")
        return outcome
