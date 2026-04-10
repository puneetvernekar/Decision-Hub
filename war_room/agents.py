import json
import textwrap

from .models import AgentVerdict, Decision, WarRoomOutcome
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

                        Important calibration rules:
                        - Treat the Risk/Critic output as one input, not authority.
                        - Do NOT change your decision just to align with the group.
                                                - Do not downgrade from Proceed to Pause unless at least
                                                    one threshold is breached OR a high/critical issue is
                                                    unresolved.
                        - Change decision only if the new evidence materially outweighs
                            your role-specific priorities.
                        - It is acceptable for agents to disagree after revision when
                            trade-offs remain unresolved.
                        - Product Manager should prioritize adoption/retention trade-offs,
                            Data Analyst should prioritize threshold integrity and trend
                            confidence, and Marketing & Comms should prioritize trust and
                            brand risk.

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
    tools = ["aggregate_metrics", "compare_trends", "summarize_sentiment"]
    system_prompt = textwrap.dedent("""\
        You are the Risk Analyst and Devil's Advocate in a launch war room.
        Challenge optimistic assumptions. Highlight worst-case scenarios
        and tail risks. Flag critical known issues. Assess cascading
        failure risks (latency -> crashes -> churn -> revenue). Evaluate
        if current rollout trends are safe to extrapolate to full rollout.
        Watch for sunk-cost or confirmation bias.

        Be constructively critical -- your job is to stress-test the decision,
        not to force a veto.

        Grounding rules you MUST follow:
        - Do not claim a threshold breach unless latest value is actually
          beyond the stated threshold in the provided tool output.
        - Distinguish clearly between "worsening trend" and "breach".
        - If major metrics are within thresholds, sentiment is net positive,
          and known issues are low/medium with active mitigation, favor a
          conditional Proceed or low-confidence Pause over high-confidence
          Pause/Roll Back.
        - Include both the strongest risk argument and the strongest
          proceed argument before concluding.

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

                        Calibration rules:
                        - Use only provided numbers; do not invent or exaggerate.
                        - Explicitly separate: breached thresholds vs negative trends.
                        - If your decision is Pause or Roll Back, include at least one
                            specific gating condition that would justify Proceed.
                        - Keep confidence proportional to evidence quality.

            Produce your verdict as JSON:
            {VERDICT_SCHEMA}
        """)

        trace(self.name, "llm_call", "Critiquing Phase 1 verdicts")
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
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


# -- Director (Phase 3) --

OUTCOME_SCHEMA = """\
{
  "final_decision": "Proceed | Pause | Roll Back",
  "decision_rationale": "... (reference specific metrics and feedback)",
  "confidence_score": 0.0-1.0,
  "confidence_drivers": ["What would INCREASE confidence: ...", "What would DECREASE confidence: ..."],
  "action_plan": [{"action": "...", "owner": "Engineering | PM | Marketing | Support | Leadership", "timeframe": "IMMEDIATE | WITHIN 24h | WITHIN 48h | ..."}],
  "risks_and_mitigations": [{"risk": "...", "likelihood": "high|medium|low", "impact": "high|medium|low", "mitigation": "..."}],
  "communication_plan": {"internal": ["..."], "external": ["..."]},
  "follow_up_monitoring": ["metric to watch", ...],
  "dissenting_opinions": ["...", ...]
}"""


class DirectorAgent(BaseAgent):
    name = "Director"
    role = "director"
    tools = []
    system_prompt = textwrap.dedent("""\
        You are a senior Director of Product making the final launch
        decision. You are impartial, data-driven, and prioritise user
        trust and business sustainability.
    """)

    def synthesize(self, initial_verdicts, critique, revised_verdicts):
        """Phase 3: make the final go/no-go call from all verdicts."""
        initial_text = format_verdicts_summary(initial_verdicts)
        critique_text = format_verdicts_summary([critique])
        revised_text = format_verdicts_summary(revised_verdicts)

        # detect who changed position
        changes = []
        for init in initial_verdicts:
            for rev in revised_verdicts:
                if init.role == rev.role:
                    if init.decision != rev.decision or abs(init.confidence - rev.confidence) > 0.03:
                        changes.append(
                            f"{init.agent_name}: {init.decision.value} ({init.confidence:.0%}) "
                            f"-> {rev.decision.value} ({rev.confidence:.0%})"
                        )
        changes_text = "\n".join(changes) if changes else "No agents changed their position."

        prompt = textwrap.dedent(f"""\
            You are making the final launch decision after hearing all
            perspectives in a war-room session.

            ## Phase 1 -- Initial Verdicts
            {initial_text}

            ## Phase 2a -- Risk/Critic's Challenges
            {critique_text}

            ## Phase 2b -- Revised Verdicts (after deliberation)
            {revised_text}

            ## Position Changes During Deliberation
            {changes_text}

            Produce the FINAL war-room decision. Rules:
            - Decision must be exactly: Proceed, Pause, or Roll Back.
            - Weight higher-confidence, data-backed verdicts more.
            - If agents are split, lean toward caution (Pause > Proceed).
            - Identify the strongest argument AGAINST the majority and
              explain why it does or doesn't change your conclusion.
            - decision_rationale must reference specific metrics and themes.
            - Each action_plan item needs action, owner, and timeframe.
            - Capture any dissenting opinions faithfully.

            Respond ONLY with JSON matching this schema:
            {OUTCOME_SCHEMA}
        """)

        trace(self.name, "llm_call", "Synthesizing final decision")
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )

        raw = resp.choices[0].message.content
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])

        return WarRoomOutcome(
            final_decision=Decision(data["final_decision"]),
            decision_rationale=data["decision_rationale"],
            confidence_score=float(data.get("confidence_score", 0.5)),
            confidence_drivers=data.get("confidence_drivers", []),
            initial_verdicts=initial_verdicts,
            critique=critique,
            revised_verdicts=revised_verdicts,
            action_plan=data["action_plan"],
            risks_and_mitigations=data["risks_and_mitigations"],
            communication_plan=data.get("communication_plan", {"internal": [], "external": []}),
            follow_up_monitoring=data["follow_up_monitoring"],
            dissenting_opinions=data.get("dissenting_opinions", []),
        )
