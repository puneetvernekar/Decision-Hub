import json
import re
import textwrap
import time

from .models import AgentVerdict, Decision
from .tools import TOOL_REGISTRY
from .trace import trace

# ── Retry helper for rate-limited APIs ──────────────────────────────────────

_MAX_RETRIES = 5


def _llm_call_with_retry(client, **kwargs):
    """Call ``client.chat.completions.create`` with automatic retry on 429.

    Parses the retry delay from the error message when available and falls
    back to exponential backoff otherwise.
    """
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            err_str = str(exc)
            if "429" not in err_str and "rate_limit" not in err_str.lower():
                raise  # not a rate-limit error — propagate immediately

            # Try to parse the suggested wait from the error message
            match = re.search(r"try again in ([\d.]+)s", err_str, re.IGNORECASE)
            wait = float(match.group(1)) + 1.0 if match else min(2 ** attempt, 30)

            if attempt == _MAX_RETRIES:
                raise  # exhausted retries

            trace("rate_limit", "retry",
                  f"429 hit — waiting {wait:.1f}s before retry {attempt}/{_MAX_RETRIES}")
            time.sleep(wait)

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
    confidence = float(payload["confidence"])
    if confidence > 1.0:
        confidence = confidence / 100.0
    confidence = max(0.0, min(1.0, confidence))
    return AgentVerdict(
        agent_name=agent_name,
        role=role,
        decision=Decision(payload["decision"]),
        confidence=confidence,
        rationale=payload["rationale"],
        key_evidence=payload["key_evidence"],
        recommended_actions=payload["recommended_actions"],
        dissent_notes=payload.get("dissent_notes"),
    )


def format_verdicts_summary(verdicts: list[AgentVerdict]) -> str:
    """Format a list of verdicts into a readable summary for LLM prompts."""
    return "\n\n".join(
        f"**{v.agent_name}** ({v.role}) → {v.decision.value} "
        f"(confidence {v.confidence:.0%})\n"
        f"Rationale: {v.rationale}\n"
        f"Evidence: {'; '.join(v.key_evidence)}\n"
        f"Recommended actions: {'; '.join(v.recommended_actions)}"
        + (f"\nDissent: {v.dissent_notes}" if v.dissent_notes else "")
        for v in verdicts
    )


# ── Base agent ──────────────────────────────────────────────────────────────

class BaseAgent:
    """Base class for war-room agents."""

    name: str = "BaseAgent"
    role: str = "base"
    system_prompt: str = ""
    tools: list[str] = []   # tool names from TOOL_REGISTRY

    def __init__(self, client, model="gpt-4o"):
        self.client = client
        self.model = model
        self.last_tool_results = {}  # filled by _invoke_tools

    # ── Tool invocation machinery ───────────────────────────────────────

    def _invoke_tools(self, dashboard):
        """Programmatically call every tool listed in ``self.tools``.

        Returns a dict mapping tool name → structured result.
        Also stores results in ``self.last_tool_results`` so the
        orchestrator can log which tools were invoked.
        """
        results = {}
        for tool_name in self.tools:
            entry = TOOL_REGISTRY[tool_name]
            results[tool_name] = entry["fn"](dashboard)
        self.last_tool_results = results
        return results

    @staticmethod
    def _format_tool_outputs(tool_results) -> str:
        """Render tool outputs as labelled JSON blocks for the LLM prompt."""
        if not tool_results:
            return ""
        sections = []
        for name, data in tool_results.items():
            desc = TOOL_REGISTRY[name]["description"]
            sections.append(
                f"### Tool: {name}\n{desc}\n"
                f"```json\n{json.dumps(data, indent=2)}\n```"
            )
        return "\n\n".join(sections)

    # ── Phase 1: independent analysis ───────────────────────────────────

    def _build_user_message(self, dashboard: dict, tool_outputs: str = "") -> str:
        parts = [
            "Here is the current launch dashboard data (10-day daily metrics, "
            "baseline comparisons, aggregate KPIs, 35 user feedback entries, "
            "success criteria, release notes, and known issues):\n\n"
            f"```json\n{json.dumps(dashboard, indent=2)}\n```",
        ]
        if tool_outputs:
            parts.append(
                "\n\nBelow are pre-computed analysis results from the tools "
                "you invoked.  Use these as the primary basis for your "
                "assessment — they contain aggregated stats, anomalies, "
                "sentiment breakdowns, and/or trend comparisons that have "
                "already been computed from the raw data above.\n\n"
                + tool_outputs
            )
        parts.append(
            f"\n\nProduce your verdict as a JSON object with this exact schema:\n{_VERDICT_SCHEMA}"
        )
        return "\n".join(parts)

    def analyze(self, dashboard: dict) -> AgentVerdict:
        """Phase 1: Invoke tools, then call the LLM with enriched context."""
        trace(self.name, "tool_call", f"Invoking {len(self.tools)} tool(s): {', '.join(self.tools)}")
        tool_results = self._invoke_tools(dashboard)
        trace(self.name, "tool_done", f"{len(tool_results)} tool result(s) received")
        tool_text = self._format_tool_outputs(tool_results)

        trace(self.name, "llm_call", "Sending enriched prompt to LLM")
        response = _llm_call_with_retry(
            self.client,
            model=self.model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self._build_user_message(dashboard, tool_text)},
            ],
        )
        raw = response.choices[0].message.content
        verdict = _parse_verdict(raw, self.name, self.role)
        trace(self.name, "verdict", f"{verdict.decision.value} (confidence: {verdict.confidence:.0%})")
        return verdict

    # ── Phase 2b: revision after deliberation ───────────────────────────

    def revise(
        self,
        dashboard: dict,
        own_verdict: AgentVerdict,
        peer_verdicts: list[AgentVerdict],
        critique: AgentVerdict,
    ) -> AgentVerdict:
        """Revise verdict after seeing peers' verdicts and the Risk/Critic's
        challenge.  The agent may adjust its decision, confidence, or
        rationale — or hold firm with stronger justification."""

        peers_summary = format_verdicts_summary(peer_verdicts)
        critique_summary = format_verdicts_summary([critique])

        prompt = textwrap.dedent(f"""\
            You previously submitted this verdict:
            Decision: {own_verdict.decision.value} (confidence: {own_verdict.confidence:.0%})
            Rationale: {own_verdict.rationale}
            Evidence: {'; '.join(own_verdict.key_evidence)}

            Here are the other agents' verdicts:
            {peers_summary}

            Here is the Risk/Critic's challenge:
            {critique_summary}

            Based on the other agents' perspectives and the Risk/Critic's
            challenges, reconsider your position:
            • Have any new arguments changed your view?
            • You may adjust your decision, confidence, rationale, or actions.
            • If you stand firm, strengthen your justification and address
              the challenges directly.
            • Use dissent_notes to flag any disagreements with the group.

            Produce your REVISED verdict as a JSON object:
            {_VERDICT_SCHEMA}
        """)

        trace(self.name, "llm_call", "Revising verdict after deliberation")
        response = _llm_call_with_retry(
            self.client,
            model=self.model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        raw = response.choices[0].message.content
        verdict = _parse_verdict(raw, self.name, self.role)
        trace(self.name, "revised", f"{verdict.decision.value} (confidence: {verdict.confidence:.0%})")
        return verdict


# ── Concrete Phase-1 agents ────────────────────────────────────────────────

class ProductManagerAgent(BaseAgent):
    name = "Product Manager"
    role = "pm"
    tools = ["aggregate_metrics", "compare_trends"]
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
    tools = ["aggregate_metrics"]
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
    tools = ["summarize_sentiment"]
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


# ── Risk / Critic (Phase 2a agent) ─────────────────────────────────────────

class RiskCriticAgent(BaseAgent):
    name = "Risk / Critic"
    role = "risk_critic"
    tools = ["aggregate_metrics"]
    system_prompt = textwrap.dedent("""\
        You are the Risk Analyst and Devil's Advocate in a product-launch war room.
        Your responsibilities:
        • Challenge optimistic assumptions made by the other agents.
        • Highlight worst-case scenarios and tail risks across all metrics.
        • Flag critical known issues and assess whether they alone warrant rollback.
        • Assess cascading failure risks (e.g., latency → crashes → churn → revenue).
        • Evaluate whether extrapolating from current rollout to full rollout is safe.
        • Identify gaps in the data or areas where more evidence is needed.
        • Call out if the team may be suffering from sunk-cost or confirmation bias.
        • Consider payment failure rate trends and revenue impact.

        Be constructively critical. Your job is to stress-test the decision.
        Your decision must be one of: Proceed, Pause, or Roll Back.
        Respond ONLY with the JSON verdict object.
    """)

    def challenge(self, dashboard: dict, verdicts: list[AgentVerdict]) -> AgentVerdict:
        """Phase 2a: Invoke tools, then review all Phase 1 verdicts critically."""
        trace(self.name, "tool_call", f"Invoking {len(self.tools)} tool(s): {', '.join(self.tools)}")
        tool_results = self._invoke_tools(dashboard)
        trace(self.name, "tool_done", f"{len(tool_results)} tool result(s) received")
        tool_text = self._format_tool_outputs(tool_results)

        verdicts_summary = format_verdicts_summary(verdicts)

        tool_section = ""
        if tool_text:
            tool_section = (
                "\n\n## Pre-Computed Analysis (from your tools)\n"
                "Use these results to ground your critique in hard data.\n\n"
                + tool_text
            )

        prompt = textwrap.dedent(f"""\
            All domain agents have submitted their independent verdicts for the
            launch decision.  Review their assessments critically.

            ## Agent Verdicts
            {verdicts_summary}

            ## Dashboard Data
            ```json
            {json.dumps(dashboard, indent=2)}
            ```
            {tool_section}

            Your job:
            1. Challenge the assumptions and reasoning of each agent.
            2. Identify blind spots, biases (sunk-cost, anchoring, confirmation),
               and risks the group may be overlooking.
            3. Highlight worst-case scenarios and tail risks.
            4. Request additional evidence where claims are unsubstantiated.
            5. Produce your own verdict — you may agree or disagree with the
               majority.

            Produce your verdict as a JSON object:
            {_VERDICT_SCHEMA}
        """)

        trace(self.name, "llm_call", "Reviewing Phase 1 verdicts critically")
        response = _llm_call_with_retry(
            self.client,
            model=self.model,
            temperature=0.4,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        raw = response.choices[0].message.content
        verdict = _parse_verdict(raw, self.name, self.role)
        trace(self.name, "verdict", f"{verdict.decision.value} (confidence: {verdict.confidence:.0%})")
        return verdict


# ── Registry ────────────────────────────────────────────────────────────────

PHASE1_AGENTS: list[type[BaseAgent]] = [
    ProductManagerAgent,
    DataAnalystAgent,
    MarketingCommsAgent,
]
