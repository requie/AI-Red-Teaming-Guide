# Model/System Security Card

One card per production AI system. Keep it short, current, and linked from the system's repo. It is the first thing an incident responder, auditor, or new team member should read.

## System Identity
- Name and version:
- Owner (team + person):
- Environment(s):
- Underlying model(s) and provider(s), with versions:
- Last updated:

## Intended and Prohibited Use
- Intended use:
- Prohibited use:
- Users (internal, customers, public) and expected volume:

## Architecture and Attack Surface
- Interfaces (API / UI / voice / CI):
- Data sources the model reads (RAG corpora, web, email, repo content):
- Trust boundaries (where untrusted text enters):
- High-value assets (data, credentials, actions):

## Agent, Tool & Protocol Inventory
| Agent / tool / MCP server / A2A peer | Version (pinned?) | Credentials & scope | Can take real-world actions? | Human approval required? |
|---|---|---|---|---|
| | | | | |

- Kill switch (how to stop it, who can, target time-to-stop):
- Agent registry entry / identity:

## Controls
- Preventive (input handling, tool allowlists, least-privilege credentials, signed tool/agent metadata):
- Detective (logging of prompts and tool calls, anomaly alerts, egress monitoring):
- Corrective (kill switch, credential rotation, memory quarantine, rollback):

## Test Coverage
- Categories tested (OWASP LLM / Agentic Top 10 IDs):
- Benchmarks run and scores (e.g. AgentDojo, InjecAgent), with dates:
- Latest red-team engagement and date:
- Known coverage gaps:

## Open Risks
| Risk | Severity | Compensating controls | Owner | Target date |
|------|----------|-----------------------|-------|-------------|
| | | | | |

## Compliance Evidence
- Regulatory regime(s) (e.g. EU AI Act risk class / GPAI, state laws):
- Where adversarial-testing evidence is stored:
- Serious-incident reporting owner and timeline:

## Incident Readiness
- On-call owner:
- Escalation path:
- Runbook link:

---

## Worked Example (filled, abbreviated)

- **System:** SupportAgent v3.2 · Owner: Support Platform (A. Rivera) · Prod + staging · Model: hosted frontier LLM (pinned version) · Updated 2026-10-01
- **Intended use:** answer customer support questions and open tickets. **Prohibited:** refunds, account changes, legal or medical advice.
- **Untrusted inputs:** customer messages, uploaded documents, help-center RAG corpus.
- **Tool inventory:**

| Tool | Version | Credentials | Real-world action? | Approval? |
|---|---|---|---|---|
| `create_ticket` | v2.1 (pinned) | Ticketing API, project-scoped | Yes | No |
| `send_email` | v1.3 (pinned) | Mail API, allowlisted domains | Yes | Yes, external domains |
| `docs-search` MCP | 1.4.2 (hash-pinned) | Read-only | No | No |

- **Kill switch:** feature flag `support_agent_enabled`; platform on-call; target ≤ 5 min.
- **Coverage:** ASI01, ASI02, ASI04, ASI06 tested; AgentDojo-style injection suite on every release. **Gap:** voice channel untested.
- **Open risk:** low-resource-language refusal gap (Medium) — compensating control: human review queue for flagged languages; owner Safety team; due 2026-11-15.
- **Compliance:** EU customers — evidence in `security-evals/reports/`; incident reporting owner: Trust & Safety lead.
