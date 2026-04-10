# Decision Hub

Decision Hub is a multi-agent launch war-room simulator.
It ingests mock product metrics and user feedback, runs a 3-phase agent process,
and outputs a final decision:

- Proceed
- Pause
- Roll Back

## Project Structure

```text
main.py
requirements.txt
README.md
data/
  daily_metrics.csv
  user_feedback.csv
  known_issues.csv
  release_notes.md
war_room/
  __init__.py
  agents.py
  mock_dashboard.py
  models.py
  orchestrator.py
  tools.py
  trace.py
```

## Current Program Flow

1. Phase 1: Independent analysis
- Product Manager
- Data Analyst
- Marketing and Comms

2. Phase 2a: Risk and Critic challenge
- Reviews Phase 1 verdicts and provides a stress-test verdict

3. Phase 2b: Agent revision
- Phase 1 agents revise after seeing peers, Risk and Critic

4. Phase 3: Director synthesis
- Produces final decision, rationale, action plan, risks, comms plan, and monitoring list

## Setup Instructions

1. Install Python 3.10+.
2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Create environment file.

```bash
cp .env.example .env
```

4. Update .env values for your provider and model.

## Environment Variables

Required:

- OPENAI_API_KEY: API key for your OpenAI-compatible endpoint(I had used an Locally hosted model(gemma3:12b))

Optional but commonly used:

- OPENAI_BASE_URL: Base URL for non-OpenAI providers (for example Ollama or Groq)
- LLM_MODEL: Model name used for all agent calls
- MAX_PARALLEL_AGENTS: Parallelism for Phase 1 and Phase 2b (use 1 for stability)
- OUTPUT_JSON_PATH: File path for final JSON output


Example .env values:

```dotenv
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=http://localhost:11434/v1
LLM_MODEL=gemma3:12b
MAX_PARALLEL_AGENTS=1
OUTPUT_JSON_PATH=decision_hub_output.json
```

## Run End-to-End

From project root:

```bash
python main.py
```

What happens end-to-end:

- Console prints session phases and final decision summary
- Trace log is written to decision-hub.log
- Final JSON payload is written to path in OUTPUT_JSON_PATH


## Output Files

- decision-hub.log: phase-by-phase execution trace and per-agent JSON verdict dumps
- decision_hub_output.json: final structured result consumed by downstream workflows


