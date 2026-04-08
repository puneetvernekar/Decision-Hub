"""Data models used across the war-room system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Decision(Enum):
    PROCEED = "Proceed"
    PAUSE = "Pause"
    ROLL_BACK = "Roll Back"


@dataclass
class AgentVerdict:
    """A single agent's assessment."""
    agent_name: str
    role: str
    decision: Decision
    confidence: float          # 0.0 – 1.0
    rationale: str
    key_evidence: list[str]
    recommended_actions: list[str]
    dissent_notes: Optional[str] = None


@dataclass
class WarRoomOutcome:
    """Final synthesised outcome of the war-room session."""
    final_decision: Decision
    decision_rationale: str
    confidence_score: float                    # 0.0 – 1.0 overall confidence
    confidence_drivers: list[str]              # what would increase/decrease confidence
    initial_verdicts: list[AgentVerdict]       # Phase 1: PM, Data, Marketing
    critique: AgentVerdict                     # Phase 2a: Risk/Critic challenge
    revised_verdicts: list[AgentVerdict]       # Phase 2b: revised after deliberation
    action_plan: list[dict]                    # [{action, owner, timeframe}, ...]
    risks_and_mitigations: list[dict]          # [{risk, likelihood, impact, mitigation}, ...]
    communication_plan: dict                   # {internal: [...], external: [...]}
    follow_up_monitoring: list[str]
    dissenting_opinions: list[str] = field(default_factory=list)
