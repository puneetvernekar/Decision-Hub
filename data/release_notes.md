# Smart Compose 2.0 — Release Notes  (v4.12.0)

## What's New
- **AI-powered sentence completion** in the email compose window.
  As you type, Smart Compose predicts the rest of your sentence and displays
  a ghost-text suggestion you can accept with Tab or dismiss by continuing
  to type.
- **Tone-matching engine** — suggestions adapt to the detected formality of
  the email thread (casual / professional / formal).
- **Multi-language support** — launch includes English, Spanish, French,
  German, and Portuguese.
- **Inline synonym picker** — long-press a suggested word to see
  alternatives.

## How It Works
- The Smart Compose model runs server-side.  Each keystroke in the compose
  window sends a lightweight context payload to the inference API, which
  returns a completion within a target latency of < 200 ms (p95).
- Suggestions are generated only for the active compose window; no email
  content is stored beyond the session.

## Rollout Plan
- **Day 1–2:** 5 % of users (internal + beta opt-in).
- **Day 3–6:** 10 % random sample.
- **Day 7–10:** 25 % with monitoring gate.
- **Day 11+:** Expand to 50 % → 100 % pending war-room approval.

## Changes to Existing Behaviour
- Compose window now includes a suggestion overlay layer.
- Slightly increased network usage (~2 KB per keystroke payload).
- New `smart_compose_enabled` user preference flag (default: ON for cohort
  users; no user-facing toggle shipped in v4.12.0 — planned for v4.12.1).
