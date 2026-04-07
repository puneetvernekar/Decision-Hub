"""
Agent definitions for the war-room multi-agent system.

Each agent receives the full dashboard snapshot and produces an AgentVerdict.
The LLM is called via the OpenAI-compatible chat completions API so it works
with OpenAI, Azure OpenAI, or any compatible provider.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any

from openai import OpenAI

from .models import AgentVerdict, Decision

# ── Shared JSON extraction helper ───────────────────────────────────────────

_VERDICT_SCHEMA = textwrap.dedent("""\
{
  "decision": "Proceed | Pause | Roll Back",
  "confidence": 0.0-1.0,
  "rationale": "...",
  "key_evidence": ["...", "..."],
  "recommended_actions": ["...", "..."],
  "dissent_notes": "... or null"
}""")


def _parse_verdict(raw: str, agent_name: str, role: str) -> AgentVerdict:
    """Extract a JSON verdict from the LLM response text."""
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in {agent_name} response")
    payload = json.loads(raw[start:end])
    return AgentVerdict(
        agent_name=agent_name,
        role=role,
        decision=Decision(payload["decision"]),
        confidence=float(payload["confidence"]),
        rationale=payload["rationale"],
        key_evidence=payload["key_evidence"],
        recommended_actions=payload["recommended_actions"],
        dissent_notes=payload.get("dissent_notes"),
    )


# ── Base agent ──────────────────────────────────────────────────────────────

class BaseAgent:
    """Base class for war-room agents."""

    name: str = "BaseAgent"
    role: str = "base"
    system_prompt: str = ""

    def __init__(self, client: OpenAI, model: str = "gpt-4o"):
        self.client = client
        self.model = model

    def _build_user_message(self, dashboard: dict) -> str:
        return (
            "Here is the current launch dashboard data (10-day daily metrics, "
            "baseline comparisons, aggregate KPIs, 35 user feedback entries, "
            "success criteria, release notes, and known issues):\n\n"
            f"```json\n{json.dumps(dashboard, indent=2)}\n```\n\n"
            f"Produce your verdict as a JSON object with this exact schema:\n{_VERDICT_SCHEMA}"
        )

    def analyze(self, dashboard: dict) -> AgentVerdict:
        """Call the LLM and return a structured AgentVerdict."""
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self._build_user_message(dashboard)},
            ],
        )
        raw = response.choices[0].message.content
        return _parse_verdict(raw, self.name, self.role)


# ── Concrete agents ─────────────────────────────────────────────────────────

class ProductManagerAgent(BaseAgent):
    name = "Product Manager"
    role = "pm"
    system_prompt = textwrap.dedent("""\
        You are the Product Manager in a product-launch war room.
        Your responsibilities:
        • Define and evaluate success criteria for the launch.
        • Assess user impact — both positive adoption signals and negative friction.
        • Frame the go / no-go decision from a product perspective.
        • Weigh short-term pain against long-term strategic value.
        • Evaluate the feature adoption funnel, retention (D1/D7), and churn trends.
        • Consider the release notes and known issues in your assessment.

        Be data-driven but also consider qualitative user sentiment.
        Compare current daily metrics against the pre-defined success criteria.
        Your decision must be one of: Proceed, Pause, or Roll Back.
        Respond ONLY with the JSON verdict object.
    """)


class DataAnalystAgent(BaseAgent):
    name = "Data Analyst"
    role = "data_analyst"
    system_prompt = textwrap.dedent("""\
        You are the Data Analyst in a product-launch war room.
        Your responsibilities:
        • Perform quantitative analysis of the 10-day daily metric trends.
        • Analyse all 9 metric dimensions: signup conversion, DAU/WAU, D1/D7
          retention, crash rate, p95 latency, payment success rate, support
          ticket volume, feature funnel completion, and churn/cancellations.
        • Identify anomalies, inflection points, and statistically meaningful shifts.
        • Compare current metrics to baseline and success criteria thresholds.
        • Assess whether trends are worsening, stabilising, or recovering.
        • Note any correlation between metrics (e.g., latency ↔ crash rate).
        • Quantify confidence in your assessment.

        Focus on numbers, trends, and statistical reasoning.
        Clearly state which thresholds are breached and by how much.
        Your decision must be one of: Proceed, Pause, or Roll Back.
        Respond ONLY with the JSON verdict object.
    """)


class MarketingCommsAgent(BaseAgent):
    name = "Marketing & Comms"
    role = "marketing_comms"
    system_prompt = textwrap.dedent("""\
        You are the Marketing & Communications lead in a product-launch war room.
        Your responsibilities:
        • Assess public perception from 35 user feedback entries across 6 channels
          (in-app, support, Twitter, Reddit, app-store, outliers).
        • Identify repeated themes, sentiment trends over the 10-day window, and
          high-impact outliers (e.g., accessibility praise, embarrassing auto-completes).
        • Evaluate brand risk including app-store rating impact and social virality.
        • Recommend proactive communication actions (blog posts, in-app banners,
          social responses, press statements).
        • Consider whether messaging can mitigate user frustration or if product
          changes are needed first.
        • Factor in the known issues list and the missing opt-out toggle.

        Balance brand protection with the opportunity of positive buzz.
        Your decision must be one of: Proceed, Pause, or Roll Back.
        Respond ONLY with the JSON verdict object.
    """)


class RiskCriticAgent(BaseAgent):
    name = "Risk / Critic"
    role = "risk_critic"
    system_prompt = textwrap.dedent("""\
        You are the Risk Analyst and Devil's Advocate in a product-launch war room.
        Your responsibilities:
        • Challenge optimistic assumptions made by other perspectives.
        • Highlight worst-case scenarios and tail risks across all 9 metrics.
        • Flag the critical known issue (KI-001: draft data loss) and assess
          whether it alone warrants rollback.
        • Assess cascading failure risks (e.g., latency → crashes → churn → revenue).
        • Evaluate whether extrapolating from 25% rollout to 50-100% is safe.
        • Identify gaps in the data or areas where more evidence is needed.
        • Call out if the team may be suffering from sunk-cost or confirmation bias.
        • Consider payment failure rate trends and revenue impact.

        Be constructively critical. Your job is to stress-test the decision.
        Your decision must be one of: Proceed, Pause, or Roll Back.
        Respond ONLY with the JSON verdict object.
    """)


class EngineeringLeadAgent(BaseAgent):
    name = "Engineering Lead"
    role = "engineering_lead"
    system_prompt = textwrap.dedent("""\
        You are the Engineering Lead in a product-launch war room.
        Your responsibilities:
        • Assess technical health: crash rate, p95 latency, payment success rate,
          and their 10-day trends.
        • Evaluate the known issues list (especially KI-001 draft loss and KI-002
          latency scaling) and their fix status.
        • Determine whether the Day 7 hotfix is working based on post-fix metrics.
        • Assess infrastructure readiness for expanding from 25% → 50% rollout.
        • Evaluate whether a targeted hotfix vs. full rollback is feasible.
        • Consider the cost and risk of rolling back vs. pausing vs. proceeding.
        • Factor in the missing user-facing toggle (KI-004) and its engineering ETA.

        Focus on system reliability, SLAs, and engineering feasibility.
        Your decision must be one of: Proceed, Pause, or Roll Back.
        Respond ONLY with the JSON verdict object.
    """)


# ── Registry ────────────────────────────────────────────────────────────────

ALL_AGENTS: list[type[BaseAgent]] = [
    ProductManagerAgent,
    DataAnalystAgent,
    MarketingCommsAgent,
    RiskCriticAgent,
    EngineeringLeadAgent,
]
