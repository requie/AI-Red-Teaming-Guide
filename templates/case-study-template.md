# AI Red Team Case Study Template

Use this to document an incident or engagement so others can learn from it. Every factual claim needs a source or an evidence tag. Mark press-reported details as such.

## Header
- Title (what happened + month/year):
- Case ID:
- Date of event / date of disclosure:
- Author and review date:

## Context
- System description (model, agent, tools, protocols such as MCP/A2A):
- Business criticality:
- Deployment context (production, internal eval, CI, etc.):

## Attack Vector & Framework Mapping
- Vector (e.g. indirect prompt injection via retrieved content):
- OWASP LLM / Agentic Top 10 IDs:
- MITRE ATLAS technique(s):
- Attacker: ☐ external ☐ insider ☐ none — agent acted on its own

## Attack Chain
Write each step as *actor → action → result*. Include the step where a control should have stopped it.
1.
2.
3.

## Finding Details
- Vulnerability class:
- Root cause:
- Controls bypassed (and why they failed):
- Severity (CVSS + AI modifiers: autonomy, blast radius, recoverability):

## Impact and Cost
- User/business impact:
- Data or systems affected:
- Estimated remediation effort:

## Detection & Response
- How it was detected and how long it took:
- Time from detection to containment (**time-to-stop**):
- Notifications made (customers, regulators):

## Evidence and Confidence
- Evidence quality: Evidence-backed / Expert guidance
- Confidence level: High / Medium / Low
- Sources (primary first; label press-reported details):

## Remediation and Validation
- Immediate mitigation:
- Long-term control improvements:
- Regression test coverage added (test IDs):

## Lessons Learned
- What changed in process/architecture:
- What other teams should test for:

---

## Worked Example (filled, abbreviated)

- **Title:** Runtime-gated MCP poisoning in a contributed "productivity" server (Aug 2026) · **Case ID:** CS-2026-07
- **Context:** Internal coding assistant with five MCP servers; one added via an external contributor's PR.
- **Vector:** Agentic supply chain → tool-metadata poisoning. **OWASP:** ASI04, ASI02. **Attacker:** external.
- **Attack chain:**
  1. Contributor → opens PR adding `productivity-suite` MCP server → merged after a code-style review only.
  2. Server → returns normal results for the first 3 tool calls → passes manual smoke test.
  3. Server → on call 4, returns metadata telling the agent to read `~/.ssh` and `~/.aws` and hide it → *control gap: no metadata diffing across calls*.
  4. Agent → reads files and sends them in a tool argument → caught by egress allowlist (blocked), alert raised.
- **Root cause:** tool metadata treated as trusted; review covered install time only.
- **Detection:** egress alert after 6 minutes; agent disabled 11 minutes later (time-to-stop 17 min).
- **Evidence:** Evidence-backed (internal logs). Confidence: High. Pattern matches the public Deadbugz campaign.
- **Remediation:** removed server; pinned tool schemas with hashes; added metadata-diff check every call; new regression tests `mcp-poison-006`, `mcp-gated-007`.
- **Lessons:** treat PRs adding agent tools as high-risk changes; test tools over long sessions.
