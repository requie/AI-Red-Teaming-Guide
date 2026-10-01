# Stakeholder Readout Outline (AI Red Teaming)

Aim for a 1-page executive summary plus a short deck or doc. Lead with decisions you need, not with techniques. Be precise about scope and residual risk — readouts can become evidence for auditors and regulators, so never claim more assurance than the testing supports.

## 1. Executive Summary (one page)
- **Bottom line** (one sentence: is it safe to ship / keep running, and under what conditions?):
- Top 3 risks, in business terms:
- Risk trend vs last engagement (better / same / worse, and why):
- **Decisions needed from leadership** (with a recommended option and a deadline):

## 2. Engagement Scope
- Systems, model versions, agents, and tools tested:
- Timeframe, access level (black/gray/white box), and constraints:
- Threat assumptions (who the attacker is, what they want):
- **What was NOT tested** (state this explicitly):

## 3. Key Findings
- Critical/high findings (one line each: what, impact, status):
- Notable exploit chains (show the chain, not just the first step):
- Framework mapping (OWASP LLM / Agentic Top 10 IDs, MITRE ATLAS techniques):
- Residual risk after mitigations:

## 4. Metrics Dashboard
| Metric | This engagement | Previous | Target |
|--------|-----------------|----------|--------|
| ASR by category (with sample size) | | | |
| Exploit recurrence after fix | | | |
| Median time-to-fix (Critical/High) | | | |
| Kill-switch time-to-stop (agents) | | | |
| Control coverage of high-risk abuse paths | | | |

## 5. Action Plan
- Immediate (0–30 days):
- Near-term (31–90 days):
- Strategic (90+ days):
- Owners and dates for each item:

## 6. Compliance & Assurance Notes
- Evidence produced (reports, logs, regression results) and where it is stored:
- Regulatory relevance (e.g. EU AI Act GPAI adversarial testing, serious-incident reporting readiness):
- Limits of these results (what a reader should not conclude):

## 7. Appendix
- Methodology and tools
- Evidence quality and confidence for each major claim
- Open questions

---

## Worked Example (executive summary, filled)

**Bottom line:** SupportAgent v3.2 is **not ready for general availability**. It can ship to the 5% pilot once the two criticals below are fixed and re-tested (target 14 Oct).

**Top risks:**
1. A document uploaded by any customer can make the agent email other customers' data to an attacker (indirect prompt injection + over-trusted email tool). *Critical, fix in progress.*
2. The agent will follow instructions hidden in a pinned MCP server's responses after several calls; install-time review missed it. *Critical.*
3. Safety refusals are weaker in Swahili and Tagalog than in English (ASR 18% vs 3%). *Medium.*

**Trend:** worse than Q3 — new email and MCP integrations added attack surface faster than controls.

**Decision needed:** approve a 2-week pilot delay **or** ship the pilot with the email tool disabled. Recommendation: ship without the email tool. Decision by 8 Oct.

**Not tested:** the voice channel and the production RAG corpus.
