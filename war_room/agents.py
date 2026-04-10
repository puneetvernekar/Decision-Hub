import json
import textwrap

from .models import AgentVerdict, Decision
from .tools import TOOL_REGISTRY
from .trace import trace


VERDICT_SCHEMA = """\
{
  "decision": "Proceed | Pause | Roll Back",
  "confidence": 0.0-1.0,
  "rationale": "...",
  "key_evidence": ["...", "..."],
  "recommended_actions": ["...", "..."],
  "dissent_notes": "... or null"
}"""


def parse_verdict(raw, agent_name, role):
    """Pull the JSON verdict out of the LLM's response."""
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in {agent_name} response")

    data = json.loads(raw[start:end])

    conf = float(data["confidence"])
    if conf > 1.0:
        conf = conf / 100.0
    conf = max(0.0, min(1.0, conf))

    return AgentVerdict(
        agent_name=agent_name,
        role=role,
        decision=Decision(data["decision"]),
        confidence=conf,
        rationale=data["rationale"],
        key_evidence=data["key_evidence"],
        recommended_actions=data["recommended_actions"],
        dissent_notes=data.get("dissent_notes"),
    )


def format_verdicts_summary(verdicts):
    """Turn a list of verdicts into readable text for LLM prompts."""
    parts = []
    for v in verdicts:
        block = (
            f"**{v.agent_name}** ({v.role}) -> {v.decision.value} "
            f"(confidence {v.confidence:.0%})\n"
            f"Rationale: {v.rationale}\n"
            f"Evidence: {'; '.join(v.key_evidence)}\n"
            f"Recommended actions: {'; '.join(v.recommended_actions)}"
        )
        if v.dissent_notes:
            block += f"\nDissent: {v.dissent_notes}"
        parts.append(block)
    return "\n\n".join(parts)


#--Base agent--

class BaseAgent:
    """Base class all war-room agents inherit from."""

    name = "BaseAgent"
    role = "base"
    system_prompt = ""
    tools = []

    def __init__(self, client, model="gpt-4o"):
        self.client = client
        self.model = model
        self.last_tool_results = {}

    def run_tools(self, dashboard):
        """Call each tool in self.tools and return {name: result}."""
        results = {}
        for name in self.tools:
            results[name] = TOOL_REGISTRY[name]["fn"](dashboard)
        self.last_tool_results = results
        return results

    def _format_tool_text(self, tool_results):
        """Format tool results as markdown+json blocks for the prompt."""
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

    def _build_prompt(self, dashboard, tool_text=""):
        """Build the user message with dashboard data + tool outputs."""
        msg = (
            "Here is the current launch dashboard data (10-day daily metrics, "
            "baselines, user feedback, success criteria, release notes, "
            "and known issues):\n\n"
            f"```json\n{json.dumps(dashboard, indent=2)}\n```"
        )
        if tool_text:
            msg += (
                "\n\nBelow are pre-computed analysis results from the tools. "
                "Use these as the primary basis for your assessment:\n\n"
                + tool_text
            )
        msg += f"\n\nProduce your verdict as JSON matching this schema:\n{VERDICT_SCHEMA}"
        return msg

    def analyze(self, dashboard):
        """Phase 1: run tools, then ask the LLM for an independent verdict."""
        trace(self.name, "tool_call", f"Running tools: {', '.join(self.tools)}")
        tool_results = self.run_tools(dashboard)
        tool_text = self._format_tool_text(tool_results)

        trace(self.name, "llm_call", "Sending prompt to LLM")
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self._build_prompt(dashboard, tool_text)},
            ],
        )
        raw = resp.choices[0].message.content
        verdict = parse_verdict(raw, self.name, self.role)
        trace(self.name, "verdict", f"{verdict.decision.value} ({verdict.confidence:.0%})")
        return verdict

    def revise(self, dashboard, own_verdict, peer_verdicts, critique):
        """Phase 2b: reconsider after seeing peers + the critic's challenge."""
        peers_text = format_verdicts_summary(peer_verdicts)
        critique_text = format_verdicts_summary([critique])

        prompt = textwrap.dedent(f"""\
            You previously submitted this verdict:
            Decision: {own_verdict.decision.value} (confidence: {own_verdict.confidence:.0%})
            Rationale: {own_verdict.rationale}
            Evidence: {'; '.join(own_verdict.key_evidence)}

            Other agents' verdicts:
            {peers_text}

            Risk/Critic's challenge:
            {critique_text}

            Reconsider your position. You can adjust your decision,
            confidence, rationale, or actions. If you stand firm, address
            the challenges directly. Use dissent_notes for disagreements.

            Produce your REVISED verdict as JSON:
            {VERDICT_SCHEMA}
        """)

        trace(self.name, "llm_call", "Revising verdict")
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content
        verdict = parse_verdict(raw, self.name, self.role)
        trace(self.name, "revised", f"{verdict.decision.value} ({verdict.confidence:.0%})")
        return verdict


# -- Phase 1 agents --

class ProductManagerAgent(BaseAgent):
    name = "Product Manager"
    role = "pm"
    tools = ["aggregate_metrics", "compare_trends"]
    system_prompt = textwrap.dedent("""\
        You are the Product Manager in a product-launch war room.
        Define and evaluate success criteria for the launch. Assess user
        impact -- both positive adoption and negative friction. Weigh
        short-term pain against long-term strategic value. Look at
        retention (D1/D7), NPS, CSAT trends, release notes, and known issues.

        Be data-driven but factor in qualitative sentiment too.
        Decision must be one of: Proceed, Pause, or Roll Back.
        Respond ONLY with the JSON verdict object.
    """)


class DataAnalystAgent(BaseAgent):
    name = "Data Analyst"
    role = "data_analyst"
    tools = ["aggregate_metrics"]
    system_prompt = textwrap.dedent("""\
        You are the Data Analyst in a product-launch war room.
        Quantitatively analyse the 10-day metric trends across all 9
        dimensions: crash rate, p95 latency, signup conversion, D1/D7
        retention, payment success, support tickets, NPS, and CSAT.
        Flag anomalies, inflection points, and threshold breaches.
        Note correlations (e.g. latency vs crash rate).

        Focus on numbers and statistical reasoning.
        Decision must be one of: Proceed, Pause, or Roll Back.
        Respond ONLY with the JSON verdict object.
    """)


class MarketingCommsAgent(BaseAgent):
    name = "Marketing & Comms"
    role = "marketing_comms"
    tools = ["summarize_sentiment"]
    system_prompt = textwrap.dedent("""\
        You are the Marketing & Comms lead in a product-launch war room.
        Assess public perception from user feedback across all channels
        (in-app, support, Twitter, Reddit, app-store). Identify recurring
        themes, high-impact outliers, and brand risk. Recommend comms
        actions (blog posts, in-app banners, social responses).
        Factor in known issues and the missing opt-out toggle.

        Balance brand protection with positive buzz opportunity.
        Decision must be one of: Proceed, Pause, or Roll Back.
        Respond ONLY with the JSON verdict object.
    """)


# -- Risk / Critic (Phase 2a) --

class RiskCriticAgent(BaseAgent):
    name = "Risk / Critic"
    role = "risk_critic"
    tools = ["aggregate_metrics"]
    system_prompt = textwrap.dedent("""\
        You are the Risk Analyst and Devil's Advocate in a launch war room.
        Challenge optimistic assumptions. Highlight worst-case scenarios
        and tail risks. Flag critical known issues. Assess cascading
        failure risks (latency -> crashes -> churn -> revenue). Evaluate
        if current rollout trends are safe to extrapolate to full rollout.
        Watch for sunk-cost or confirmation bias.

        Be constructively critical -- your job is to stress-test the decision.
        Decision must be one of: Proceed, Pause, or Roll Back.
        Respond ONLY with the JSON verdict object.
    """)

    def challenge(self, dashboard, verdicts):
        """Phase 2a: review all Phase 1 verdicts as devil's advocate."""
        trace(self.name, "tool_call", f"Running tools: {', '.join(self.tools)}")
        tool_results = self.run_tools(dashboard)
        tool_text = self._format_tool_text(tool_results)

        verdicts_text = format_verdicts_summary(verdicts)

        tool_section = ""
        if tool_text:
            tool_section = (
                "\n\n## Pre-Computed Analysis\n"
                "Use these to ground your critique in data.\n\n"
                + tool_text
            )

        prompt = textwrap.dedent(f"""\
            All agents have submitted their independent verdicts.
            Review them critically.

            ## Agent Verdicts
            {verdicts_text}

            ## Dashboard Data
            ```json
            {json.dumps(dashboard, indent=2)}
            ```
            {tool_section}

            Your job:
            1. Challenge each agent's assumptions and reasoning.
            2. Identify blind spots, biases, and overlooked risks.
            3. Highlight worst-case scenarios.
            4. Produce your own verdict (you may agree or disagree).

            Produce your verdict as JSON:
            {VERDICT_SCHEMA}
        """)

        trace(self.name, "llm_call", "Critiquing Phase 1 verdicts")
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.4,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content
        verdict = parse_verdict(raw, self.name, self.role)
        trace(self.name, "verdict", f"{verdict.decision.value} ({verdict.confidence:.0%})")
        return verdict


PHASE1_AGENTS = [
    ProductManagerAgent,
    DataAnalystAgent,
    MarketingCommsAgent,
]
