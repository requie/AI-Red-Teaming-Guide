# Changelog

All notable changes to this guide should be documented in this file.

## [v1.2.0] - 2026-10 — Q4 refresh: agent incidents, coding agents, A2A, frontier capability

### Added
- **New sections:** AI Coding-Agent & CI/CD Security; Agent-to-Agent (A2A) & Agent Identity; Frontier Capability & AI-Accelerated Vulnerability Discovery (Claude Mythos / Project Glasswing, rogue-agent incidents).
- **New case studies:** D — OpenAI frontier agent reached Australia's Medicare statistics portal during internal evaluation (Jun 2026, disclosed Sep); E — "Comment and Control" prompt injection against Claude Code, Gemini CLI, and Copilot coding agents in CI (Apr 2026); F — Deadbugz runtime-gated MCP supply-chain campaign (Aug 2026).
- **Regulatory:** EU **Digital Omnibus on AI** (high-risk obligations moved to 2 Dec 2027 / 2 Aug 2028); FTC probe of OpenAI, Anthropic, and METR (Sep 2026).
- **Frameworks:** OWASP 2026 LLM Top 10, Agent Control Standard, AI Red Teaming Landscape; MITRE ATLAS v5.x.
- **Tools:** MiDojo (asago / Red Hat). **Benchmarks:** InjecAgent, HarmBench, JailbreakBench, CyberSecEval. **Learning:** HackAPrompt and vendor AI bug bounty programs.
- Filled-in, agent-aware versions of the rules-of-engagement, stakeholder-readout, case-study, and model/system security-card templates.

### Changed
- EU AI Act and NIST items rewritten for October 2026 status (GPAI enforcement in force; Cyber AI Profile still a draft; COSAiS agent overlays in development); 2023 US EO marked historical.
- Update Watchlist re-validated 2026-10-01; badge and footer dates updated.
- Table of contents now covers every section; comparison matrix and Resources include every listed tool; case-study headings normalized.
- `ai-redteam-regression.yml` skips cleanly without an API key instead of masking failures.
- Spanish, Chinese, and French translations re-synced with the English v1.2.0 edition (incl. DeepKeep and Darkmoon).

## [2026-09-30] - Darkmoon

### Added

- Added **Darkmoon** (GPL-3.0, [ASCIT31/Dark-Moon](https://github.com/ASCIT31/Dark-Moon)) to Open-Source Tools: self-hosted, LLM-orchestrated multi-agent penetration testing over MCP with real-exploit validation (contributed by @MBK-fr in #23).

## [2026-09-27] - DeepKeep AI Security Platform

### Added

- Added **DeepKeep AI Security Platform** to Commercial Platforms, the comparison matrix, and commercial resources.
- Summarized vendor-described automated AI red teaming for continuous coverage, regression testing, and compliance evidence.
- Added Vibe AI Red Teaming as human-steered adaptive testing for business-impact vulnerabilities and agentic multi-step attack paths.
- Recorded the DeepKeep product-source review in `resources-validation.md`.

## [2026-09-07] - Featured commercial platform: AVERSYN

### Added

- Featured **AVERSYN by Cogensec** in the tools overview and at the top of Commercial Platforms, with a direct table-of-contents link.
- Added Aversyn to the comparison matrix and commercial resources, linking to https://cogensec.com/aversyn.
- Described autonomous adversarial validation, reproducible evidence, remediation, operator controls, and engineering integrations from Cogensec's product page.
- Made commercial/proprietary status, invitation-based frontier access, and maintainer affiliation explicit; left pricing and learning curve unassessed.
- Synchronized the Aversyn addition across English, Spanish, French, and Simplified Chinese and recorded the product-source review date.

## [v1.1.0] - 2026-07 — New tools + multi-language support
### Added
- **New red-teaming tools surfaced:**
  - Scenario (LangWatch) — open-source, simulation-based multi-turn agent red teaming (added to tools, comparison matrix, resources).
  - General Analysis — commercial agentic + tool/MCP red teaming with CI/CD gates (promoted to a full Commercial Platforms entry).
  - Haize Labs — commercial large-scale automated LLM stress-testing.
- **Multi-language support:** full translations `README.es.md` (Spanish), `README.zh.md` (Chinese, Simplified), and `README.fr.md` (French), each with a sync/source-of-truth note.
- **Language bar** switcher added to the top of every README variant (English · Español · 中文 · Français).
- **Translations** contribution note added to the Contributing section.

## [2026-06-10] - Agentic-era refresh
### Added
- **New attack-surface sections** in README:
  - MCP & Tool-Protocol Security (tool/schema poisoning, server compromise, credential theft, namespace collisions)
  - Computer-Use & Browser Agent Attacks (visual hijacking, OCR spoofing, pixel adversarial inputs)
  - RAG Attack Taxonomy (source poisoning, retrieval manipulation, citation spoofing, context exhaustion)
  - Voice, Audio & Multimodal Attacks (speaker cloning, audio adversarial, ultrasonic, cross-modal)
  - Fine-Tuning & Model Supply-Chain Security (backdoors, malicious LoRA, poisoned checkpoints)
  - AI-on-AI Red Teaming (agent-orchestrated assessment, judge-model pitfalls)
  - AI Incident Response (agent containment, escalation logic, EU serious-incident reporting)
- **Frameworks**: OWASP Top 10 for Agentic Applications 2026 (ASI01–ASI10) and Microsoft Agentic Failure-Mode Taxonomy v2.0.
- **Three new agentic attack trees**: Goal Hijack, Agentic Supply Chain Compromise, Rogue Agents; all trees tagged with OWASP ASI IDs.
- **Runnable Evaluation Harness**: YAML policy, Python scorer, and release-gate runner replacing prior pseudocode.
- **Three current case studies** (2025–2026): AI-orchestrated state intrusion, OpenClaw framework, GitHub Copilot RCE; older cases regrouped as Historical.
- **EU AI Act enforcement mapping**: GPAI systemic-risk obligations (Aug 2 2026), Article→evidence table.
- Filled examples added to vulnerability-report, test-case-library, and threat-modeling-workshop templates; agentic checks added to the PR checklist.

### Changed
- Tools section updated for 2026 (PyRIT v0.11/repo move, Garak→NVIDIA v0.14, promptfoo→OpenAI acquisition, multi-turn orchestration shift, validation dates).
- Added **Redamon** (samugit83) to the open-source tools list and comparison matrix; credited @samugit83 in a new Contributors subsection.
- Added a "Join the Global Red Teaming Network" banner (top of README + Contributing section) linking to the Cogensec network.
- 2025–2026 incident list and industry-impact statistics in "Why It Matters".
- Update Watchlist re-validated to 2026-06-10 with NIST Cyber AI Profile, COSAiS overlays, and critical-infrastructure profile.
- Badge and freshness messaging updated to June 2026; removed stale `--break-system-packages` pip guidance.

## [2026-02] - Source governance refresh
### Added
- README refresh for 2026 source governance:
  - Updated freshness messaging and badge to 2026
  - Added a date-stamped “Latest Update Watchlist” with official EU AI Act, OWASP Agentic Top 10, and NIST update triggers
  - Expanded Regulatory Compliance and Resources sections with current references
- `resources-validation.md` expanded with 2026-04-27 validation dates and additional standards/regulatory rows.
- Operational implementation sections in README:
  - Implementation Quickstart (30/60/90)
  - Evaluation Harness (Reference Implementation)
  - Agentic AI Attack Trees + Controls Mapping
  - AI Harm Severity and Triage Model
  - Secure SDLC Integration Artifacts
  - Defensive Architecture Patterns
  - Multilingual & Cultural Safety Playbook
  - Data Governance for Red Teaming
  - Metrics That Matter (and Anti-Metrics)
  - Purple Team Operations
  - Common Implementation Pitfalls
  - Case Study Quality Bar
  - Model & System Cards for Security Posture
  - Source Hygiene & Update Governance
  - Practitioner Appendices
- New templates in `templates/`:
  - threat-modeling-workshop.md
  - ai-security-pr-checklist.md
  - rules-of-engagement-template.md
  - vulnerability-report-template.md
  - test-case-library-starter.md
  - stakeholder-readout-outline.md
  - model-system-security-card.md
  - case-study-template.md
- `resources-validation.md` to track external source freshness.
- `.github/workflows/ai-redteam-regression.yml` baseline CI workflow.
