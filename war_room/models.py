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
    individual_verdicts: list[AgentVerdict]
    action_plan: list[str]
    risks_and_mitigations: list[str]
    follow_up_monitoring: list[str]
    dissenting_opinions: list[str] = field(default_factory=list)
