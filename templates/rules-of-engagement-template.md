# Red Team Rules of Engagement (Template)

Sign this **before** testing starts. It protects testers, system owners, and the organization, and it is the scope evidence regulators and auditors ask for (e.g. EU AI Act GPAI adversarial-testing documentation).

## 1. Engagement Identity
- Engagement ID / name:
- Dates and testing windows (incl. time zone):
- Red team lead:
- System owner:

## 2. Scope
- In scope (systems, model versions, endpoints, environments):
- Out of scope:
- **Agents and tools in scope** (list each agent, MCP server, plugin, A2A peer, and the credentials/permissions it holds):
- **Third-party models/APIs touched** (confirm their terms allow testing):
- Environments: ☐ dedicated test ☐ staging ☐ production (requires extra sign-off)

## 3. Authorized Techniques
- Allowed (e.g. prompt injection, jailbreaks, tool-misuse attempts, MCP/plugin poisoning in test registries, RAG poisoning in test corpora):
- Prohibited (e.g. DoS, social engineering of staff, testing third-party services outside their published bounty scope, real customer data):
- Automated tooling allowed (name + max request rate):

## 4. Agent-Specific Guardrails
- Agents run with **test credentials only**, scoped to the minimum needed; list them:
- Network egress for agents under test: ☐ blocked ☐ allowlist only (list):
- Actions that always need human approval during testing (payments, emails to real recipients, deletes, merges):
- **Kill switch**: who can stop a run, how, and the maximum acceptable time from alert to stop:
- Real money, real messages, or real infrastructure changes: ☐ never ☐ only with listed approvals

## 5. Safety Guardrails & Stop Conditions
- No export of production data
- Rate limits and resource/cost ceilings:
- Stop immediately if: real user data is exposed · an agent acts outside the test environment · unexpected harm to a third party · cost ceiling reached
- Who is notified on a stop:

## 6. Escalation and Notification
| Severity | Notify | Within |
|----------|--------|--------|
| Critical | Security lead + system owner (phone) | Immediately |
| High | Security lead | 24 hours |
| Medium/Low | Findings report | Weekly / end of engagement |

- Security contact:
- Legal/compliance contact:
- Regulatory reporting owner (e.g. EU AI Act serious incidents):

## 7. Data Handling
- Data classes used (synthetic / anonymized / production):
- Where findings and transcripts are stored (encrypted, access-controlled):
- Retention period:
- Deletion and evidence-preservation procedure:

## 8. Authorization & Safe Harbor
> The organization authorizes the named testers to perform the techniques in Section 3 against the systems in Section 2 during the windows in Section 1. Good-faith testing within this scope is authorized and will not be treated as a policy violation. Testers will stop and report if they go outside scope by accident. This authorization does not cover third-party systems unless their owners have agreed in writing.

*(Have legal review this wording for your jurisdiction.)*

## 9. Sign-off
- Red Team Lead: ____________ Date: ______
- System Owner: ____________ Date: ______
- Security Lead: ____________ Date: ______
- Legal/Compliance: ____________ Date: ______

---

## Worked Example (filled, abbreviated)

- **Engagement:** RT-2026-Q4-02 · 6–17 Oct 2026, 09:00–18:00 UTC · Lead: J. Okafor · Owner: Support Platform team
- **In scope:** SupportAgent v3.2 (staging), its RAG index (test corpus only), `send_email` and `create_ticket` tools, `docs-search` MCP server v1.4.2 (pinned)
- **Out of scope:** production tenants, the payment service, third-party LLM provider infrastructure
- **Agent credentials:** test Gmail sandbox account; ticketing API key scoped to the `RT-TEST` project
- **Egress:** allowlist only — LLM provider API + internal staging hosts
- **Human approval required:** any `send_email` to a non-`@example.test` address
- **Kill switch:** platform on-call disables the agent feature flag; target ≤ 5 minutes from alert to stop
- **Stop conditions:** any production record seen, any outbound email to a real domain, LLM spend > $500
- **Signed:** all four parties, 3 Oct 2026
