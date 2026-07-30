# Changelog

All notable changes to this guide should be documented in this file.

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
