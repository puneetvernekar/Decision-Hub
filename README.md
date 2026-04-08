# War Room — Multi-Agent Launch Decision System

A multi-agent system that simulates a cross-functional "war room" during a
product launch. It analyses a mock dashboard (metrics + user feedback) and
produces a structured launch decision: **Proceed / Pause / Roll Back**, along
with a concrete action plan.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    War Room Orchestrator                  │
│                                                          │
│  Phase 1 — Independent Analysis (parallel)               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │    PM    │  │  Data    │  │Marketing │               │
│  │  Agent   │  │ Analyst  │  │ & Comms  │               │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘               │
│       │              │             │                      │
│       └──────────────┼─────────────┘                      │
│                      ▼                                    │
│  Phase 2a — Risk/Critic Challenge                        │
│  ┌──────────────────────────────────────┐                │
│  │  Risk/Critic reviews all 3 verdicts  │                │
│  │  → challenges + own verdict          │                │
│  └──────────────────┬───────────────────┘                │
│                     ▼                                     │
│  Phase 2b — Deliberation & Revision (parallel)           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ PM revise│  │DA revise │  │MC revise │               │
│  │ (sees    │  │ (sees    │  │ (sees    │               │
│  │ peers +  │  │ peers +  │  │ peers +  │               │
│  │ critique)│  │ critique)│  │ critique)│               │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘               │
│       │              │             │                      │
│       └──────────────┼─────────────┘                      │
│                      ▼                                    │
│  Phase 3 — Director / Senior PM Synthesis                │
│  ┌──────────────────────────────────────┐                │
│  │  Receives initial + critique +       │                │
│  │  revised verdicts → final decision   │                │
│  └──────────────────────────────────────┘                │
└──────────────────────────────────────────────────────────┘
```

### Agents

| Agent | Phase | Role |
|---|---|---|
| **Product Manager** | 1 & 2b | Success criteria, user impact, go/no-go framing |
| **Data Analyst** | 1 & 2b | Quantitative trends, anomalies, confidence levels |
| **Marketing & Comms** | 1 & 2b | Sentiment, brand risk, communication actions |
| **Risk / Critic** | 2a | Devil's advocate — challenges assumptions, highlights risks |
| **Director / Senior PM** | 3 | Final authority — synthesises all inputs into go/no-go decision |

### Three-Phase Process

1. **Independent Analysis** — PM, Data Analyst, and Marketing/Comms analyse the
   dashboard in parallel and submit independent verdicts.
2. **Deliberation** —
   - **(2a)** The Risk/Critic reviews all Phase 1 verdicts, challenges
     assumptions, and produces its own verdict.
   - **(2b)** Each Phase 1 agent sees the other agents' verdicts plus the
     Risk/Critic's challenges, and submits a **revised** verdict — adjusting
     decision, confidence, or rationale as warranted.
3. **Director Synthesis** — A Director / Senior PM receives the initial verdicts,
   the Risk/Critic's challenge, and the revised verdicts.  They make the final
   go/no-go call with an action plan, risk mitigations, and monitoring gates.

### Agent Tools

Agents invoke programmatic tools **before** calling the LLM.  Tool outputs
(aggregated stats, anomalies, sentiment breakdowns, trend comparisons) are
injected into the prompt so the model reasons over processed data, not raw JSON.

| Tool | Called by | Description |
|---|---|---|
| `aggregate_metrics` | PM, Data Analyst, Risk/Critic | Per-metric stats (min/max/mean/latest), trend direction, threshold breach flags, overall health score |
| `detect_anomalies` | Data Analyst, Risk/Critic | Z-score anomaly detection across all daily metric time-series — returns spikes/drops with severity |
| `summarize_sentiment` | Marketing & Comms | Channel/theme/timeline breakdown of user feedback with high-impact item detection |
| `compare_trends` | PM | Baseline deltas, 3-day velocity, direction (improving/worsening/stable), linear recovery ETA |

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
    ├── agents.py              # 4 agent classes (PM, Data, Marketing, Risk/Critic)
    ├── tools.py               # 4 programmatic tools (aggregate, anomaly, sentiment, trends)
    └── orchestrator.py        # 3-phase war-room orchestration with feedback loop
```

## Output

The system produces:

- **Final Decision**: Proceed / Pause / Roll Back
- **Decision Rationale**: Why this decision was reached
- **Initial Verdicts**: Phase 1 independent assessments (PM, Data, Marketing)
- **Critique**: Risk/Critic's challenge verdict with identified blind spots
- **Revised Verdicts**: Phase 2b assessments after deliberation (showing what changed)
- **Action Plan**: Sequenced, assignable steps
- **Risks & Mitigations**: Identified risks with countermeasures
- **Follow-up Monitoring**: Metrics and gates to track
- **Dissenting Opinions**: Faithfully captured minority views
