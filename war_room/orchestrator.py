"""
War-room orchestrator — 3-phase coordinated decision process.

Phase 1 — Independent analysis by PM, Data Analyst, Marketing/Comms (parallel)
Phase 2 — Risk/Critic challenges (2a) + agents revise their verdicts (2b, parallel)
Phase 3 — Director / Senior PM synthesises the final go/no-go decision
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from .agents import (
    PHASE1_AGENTS, RiskCriticAgent, DirectorAgent, BaseAgent,
)
from .trace import trace, reset_trace, get_trace, log_verdict, log_outcome


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

    # -- Phase 3: Director synthesis --

    def _phase3_synthesize(self, initial_verdicts, critique, revised_verdicts):
        """Director makes the final go/no-go decision."""
        director = DirectorAgent(self.client, self.model)
        return director.synthesize(initial_verdicts, critique, revised_verdicts)

    # -- Public API --

    @staticmethod
    def _log_tools(agents):
        """Print which tools each agent invoked (verbose helper)."""
        for agent in agents:
            if agent.last_tool_results:
                tool_names = ", ".join(agent.last_tool_results.keys())
                print(f"    {agent.name} invoked: {tool_names}")

    def run(self, dashboard, verbose=True):
        """Execute the full 3-phase war-room session and return the outcome."""
        reset_trace()
        if verbose:
            print("=" * 60)
            print("  DECISION-HUB")
            print("=" * 60)
        trace("orchestrator", "session", "Session started")

        # Phase 1 — Independent Analysis
        trace("orchestrator", "phase_start", "Phase 1 — Independent Agent Analysis")
        if verbose:
            print("─" * 60)
            print("  Phase 1 — Independent Agent Analysis")
            print("─" * 60)
        initial_verdicts, _ = self._phase1_analyze(dashboard)
        trace("orchestrator", "phase_end", f"Phase 1 complete — {len(initial_verdicts)} verdicts collected")
        for v in initial_verdicts:
            log_verdict(f"Phase 1 — {v.agent_name}", v)
        if verbose:
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
        critique, _ = self._phase2a_critique(dashboard, initial_verdicts)
        trace("orchestrator", "phase_end", f"Phase 2a complete — critique: {critique.decision.value}")
        log_verdict("Phase 2a — Risk/Critic", critique)
        if verbose:
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
        for v in revised_verdicts:
            log_verdict(f"Phase 2b — {v.agent_name} (revised)", v)
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
        log_outcome(outcome)
        return outcome
