# War Room — Multi-Agent Launch Decision System

A multi-agent system that simulates a cross-functional "war room" during a
product launch. It analyses a mock dashboard (metrics + user feedback) and
produces a structured launch decision: **Proceed / Pause / Roll Back**, along
with a concrete action plan.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   War Room Orchestrator              │
│                                                     │
│  Phase 1 — Individual Analysis (parallel)           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │    PM    │ │  Data    │ │Marketing │            │
│  │  Agent   │ │ Analyst  │ │ & Comms  │            │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘            │
│  ┌────┴─────┐ ┌────┴─────┐                         │
│  │  Risk /  │ │  Engg    │                         │
│  │  Critic  │ │  Lead    │                         │
│  └────┬─────┘ └────┬─────┘                         │
│       │             │                               │
│  Phase 2 — Critique Round                           │
│       Risk/Critic reviews all verdicts              │
│       │                                             │
│  Phase 3 — Synthesis                                │
│       Facilitator LLM produces final decision       │
└─────────────────────────────────────────────────────┘
```

### Agents

| Agent | Role |
|---|---|
| **Product Manager** | Success criteria, user impact, go/no-go framing |
| **Data Analyst** | Quantitative trends, anomalies, confidence levels |
| **Marketing & Comms** | Sentiment, brand risk, communication actions |
| **Risk / Critic** | Devil's advocate, worst-case scenarios, bias checks |
| **Engineering Lead** | Technical health, hotfix feasibility, rollback cost |

### Three-Phase Process

1. **Individual Analysis** — All 5 agents analyse the dashboard in parallel and
   submit independent verdicts (decision + confidence + evidence + actions).
2. **Critique Round** — The Risk/Critic agent reviews all verdicts, identifies
   blind spots, and challenges weak reasoning.
3. **Synthesis** — A facilitator LLM weighs all verdicts and the critique to
   produce the final decision, action plan, and risk mitigations.

## Mock Scenario

**Feature:** Smart Compose 2.0 — AI-powered email completion  
**Rollout:** 15% of users over 72 hours  
**Status:** Error rate spiked to 2.4% (threshold: 1.0%), latency peaked at
420ms, but both are recovering. User feedback is polarised — strong adoption
signals alongside performance complaints and a public auto-complete incident.

## Quick Start

### Offline Demo (no API key needed)

```bash
pip install -r requirements.txt
python main.py --offline
```

### With OpenAI API

```bash
export OPENAI_API_KEY="sk-..."
pip install -r requirements.txt
python main.py --model gpt-4o
```

### JSON Output

Add `--json` to also write the full structured outcome to `war_room_outcome.json`:

```bash
python main.py --offline --json
```

## Project Structure

```
├── main.py                    # Entry point + offline simulation + output formatting
├── requirements.txt
├── README.md
└── war_room/
    ├── __init__.py
    ├── mock_dashboard.py      # Mock metrics, KPIs, user feedback, success criteria
    ├── models.py              # AgentVerdict, WarRoomOutcome, Decision enum
    ├── agents.py              # 5 agent classes with role-specific system prompts
    └── orchestrator.py        # 3-phase war-room coordination logic
```

## Output

The system produces:

- **Final Decision**: Proceed / Pause / Roll Back
- **Decision Rationale**: Why this decision was reached
- **Individual Verdicts**: Each agent's independent assessment
- **Action Plan**: Sequenced, assignable steps
- **Risks & Mitigations**: Identified risks with countermeasures
- **Follow-up Monitoring**: Metrics and gates to track
- **Dissenting Opinions**: Faithfully captured minority views
