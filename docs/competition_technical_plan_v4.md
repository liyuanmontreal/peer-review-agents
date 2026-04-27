# Technical Plan for the Koala Science ICML 2026 Agent Review Competition (v4)
## 0. Current Implementation Status (v4.1 Alignment)

This section aligns the technical plan with the current codebase reality, so future implementation follows the actual project state rather than an idealized clean-room phase order.

### 0.1 Already Implemented or Partially Implemented

The following components already exist in the current codebase:

- **GSR artifact workflow** has been added, including local smoke validation.
- **Reactive analysis Phase 5A** is implemented:
  - external comment → claim extraction
  - paper retrieval + verification via GSR
  - recommendation (`react` / `skip` / `unclear`)
  - draft generation (dry-run)
  - persistence into Koala-side DB tables
- **Reactive persistence tables** exist:
  - `koala_extracted_claims`
  - `koala_claim_verifications`
  - `koala_reactive_drafts`
- **Heat-band / Goldilocks model** is implemented at the tool layer:
  - `paper_heat_band()`
  - `crowding_score()`
  - `sort_seed_papers_by_crowding()`
- **Competition rule updates** already reflected in code and strategy:
  - verdict requires **3** distinct other agents (not 5)
  - contradiction verdict aliases are normalized (`refuted`, `contradicted`, `contradiction`)
- **Sync robustness patches** already exist:
  - comment text refresh on re-sync
  - increased default comment fetch limit
  - basic observability for Phase 5A stats

### 0.2 In Progress

The following is the immediate next implementation focus:

- **Phase 5B runtime integration**
  - choose the best reactive candidate at runtime
  - apply heat-band / Goldilocks preference in live selection
  - route selected candidates through artifact + preflight
  - keep posting **dry-run by default**

### 0.3 Not Yet Implemented

The following remain future work:

- full proactive seed-comment runtime
- live reactive posting enabled by default
- full verdict assembly and live submission path
- novelty-checker public surfacing
- severe-issue public surfacing
- final rubric-anchored calibrated scorer
- 
## 1. Project Goal

The goal of this project is to build a competitive **GSR Meta-Review Agent** for the Koala Science ICML 2026 Agent Review Competition by leveraging and extending our existing **GSR (Good Samaritan Review)** system. This agent will read papers, monitor multi-agent discussions, publish high-quality comments during the review phase, and submit a calibrated verdict during the verdict phase when the competition conditions are satisfied. The primary objective is to produce decisions that are as close as possible to the actual ICML 2026 accept/reject outcomes while also generating useful, evidence-grounded contributions that other agents may cite.

This v4 plan updates the prior design to explicitly incorporate:
- the **global competition timeline** (competition open/close dates and the rolling 72-hour paper lifecycle),
- the **karma economy** (initial 100 karma, explicit action costs, no-bankruptcy policy),
- the **verdict eligibility rules** (discussion-before-verdict, 3 distinct other agents, correct time window),
- the **sparse-participation reality** (approximately 21 active agents in current competition),
- and the need to turn strategy into **hard operational rules** rather than soft heuristics.

The design principle remains unchanged:

> **GSR remains the grounding engine; the Koala-specific orchestration, strategy, and rule-hardening layers convert GSR into a competition-grade meta-review agent.**


## 2. Competition Rule Model and Timeline Assumptions

This section formalizes the competition constraints as system-level assumptions and hard requirements.

### 2.1 Global Competition Timeline

The competition-wide timeline is:

- **Competition opens:** Friday **2026-04-24, 12:00 PM ET**
- **Competition closes:** Thursday **2026-04-30 AoE**
- **Final verdict windows complete:** approximately **72 hours after the last paper is released**
- **Leaderboard published:** after **ICML 2026 decisions are public**

This means the system must reason over **two simultaneous time axes**:

1. **Global competition window**  
   The platform accepts new paper releases only during the competition-wide release period.

2. **Per-paper rolling lifecycle**  
   Each paper still has its own **72-hour lifecycle**:
   - **0–48h:** Review / discussion phase
   - **48–72h:** Verdict phase

### 2.2 Critical Timeline Consequence

Because papers may be released late in the competition-wide window, the agent **must not assume all verdicts end on April 30**. Instead:

- **Paper release may stop at competition close (Apr 30 AoE),**
- but **verdict submission may continue for ~72h after the last paper release**, depending on when the last papers appeared.

Therefore, the system must track:

- `competition_release_open`
- `competition_release_closed`
- `paper_open_time`
- `paper_review_end_time = paper_open_time + 48h`
- `paper_verdict_end_time = paper_open_time + 72h`
- `global_last_seen_release_time`
- `global_post_release_tail_active`

### 2.3 Karma Rules (Hard Constraints)

Every agent starts with **100 karma**.

Actions cost karma as follows:

- **First comment or first thread on a paper:** `1.0 karma`
- **Each subsequent comment or thread on the same paper:** `0.1 karma`
- **Submitting a verdict:** `0.0 karma`

If the agent does not have enough karma to cover an action, it **cannot** take that action.

### 2.4 Verdict Eligibility Rules (Hard Constraints)

A verdict is only attempted if all of the following are satisfied:

1. The agent has **already participated in discussion** on that paper.
2. The paper is currently inside its **48–72h verdict window**.
3. There are at least **3 distinct other agents** whose comments are **citable** for the final verdict.
4. The required **audit artifact** exists and its `github_file_url` is available.
5. Internal score confidence exceeds the minimum threshold for submission.

### 2.5 Transparency / Audit Rule

Every external action (comment or verdict) must include a `github_file_url` pointing to a public transparency artifact in the audit repository. This is treated as a **hard preflight requirement**, not a best-effort behavior.


## 3. Agent Design (Updated v4)

The GSR Meta-Review Agent is designed as an **evidence-grounded, literature-aware, decision-calibrated, rule-hardened meta-reviewer** rather than a general-purpose reviewer. Its two main goals are:

1. Publish short, concrete, verifiable, citation-friendly comments during the review phase.
2. Submit a more accurate 0–10 verdict during the verdict phase by integrating:
   - paper content,
   - multi-agent discussion,
   - novelty signal,
   - severe-issue signal,
   - discussion structure,
   - and the implicit standards of ICML reviewing.

Internally, the v4 agent consists of **eight** modules:

1. **Evidence Checker**
2. **Paper-First Evidence Scout**
3. **Novelty Checker**
4. **Severe Issue Scanner**
5. **Decision Calibrator**
6. **Opportunity Manager**
7. **Citation & Eligibility Engine**
8. **Action Preflight / Moderation Guardrail**

### 3.1 Evidence Checker

The **Evidence Checker** reads comments from other agents, extracts verifiable claims from those comments, and uses the existing GSR evidence retrieval and verification pipeline to determine whether those claims are supported, contradicted, or insufficiently supported by the paper.

Its main role is to generate grounded:
- corrections,
- confirmations,
- clarifications,
- or scope-calibration comments.

### 3.2 Paper-First Evidence Scout (New, Required for Sparse Fields)

The **Paper-First Evidence Scout** is the major v4 addition required by sparse participation.

In low-participation settings, the agent cannot rely on waiting for others to post first. Therefore, when a paper has no meaningful discussion yet, this module proactively scans the paper itself to find **one concrete, verifiable, discussion-seeding point** that is:

- low-risk,
- evidence-grounded,
- useful to later agents,
- and likely to remain citable.

Typical outputs include:

- a strong claim with only partial evidence support,
- missing baseline / incomplete comparison coverage,
- a scope mismatch between claim and experiment,
- a novelty calibration question,
- or a high-confidence severe issue.

This converts the system from a purely reactive fact-checker into a **discussion initializer** while preserving the factual-evidence identity of GSR.

### 3.3 Novelty Checker

The **Novelty Checker** evaluates whether the paper’s claimed contribution is genuinely novel, moderately incremental, or likely overstated relative to prior work. This module is especially important because other agents may systematically overestimate novelty.

It uses the **Lacuna API** as the primary literature search backend to retrieve related work and compare the current paper against relevant prior papers. If needed, additional literature sources such as Semantic Scholar, arXiv, or Crossref can be used as fallback or supplemental sources.

This module is not intended to perform exhaustive literature review; it is intended to detect whether the current discussion is materially over- or under-estimating novelty in a way that could change the final verdict.

### 3.4 Severe Issue Scanner

The **Severe Issue Scanner** is a dedicated phase that explicitly looks for decision-critical or desk-reject-level failures that general LLM reasoning often misses unless prompted separately. This includes issues such as:

- missing bibliography,
- missing experiments,
- lack of empirical evidence for empirical claims,
- absent or fundamentally inadequate baselines,
- broken or suspicious citations,
- major methodological omissions,
- severe mismatch between claimed contributions and presented evidence.

The purpose is not to enumerate ordinary weaknesses, but to identify flaws that should strongly affect the final verdict.

### 3.5 Decision Calibrator

The **Decision Calibrator** estimates whether identified issues are actually decision-critical. It does not simply penalize every weakness.

Instead, it integrates:

- strengths,
- weaknesses,
- novelty signal,
- severe-issue signal,
- external discussion consensus,
- disagreement structure,
- and citation value of discussion threads,

into a structured internal reasoning state.

The v4 design continues to use a **two-stage scoring mechanism**:

1. An **instruct-tuned model** produces structured reasoning and qualitative assessments.
2. A **base model (non-instruct-tuned)** maps that reasoning into the final numerical 0–10 score.

This is intended to improve calibration and reduce rhetoric-driven overconfidence.

### 3.6 Opportunity Manager

The **Opportunity Manager** decides which papers and threads are worth participating in, controls karma usage, estimates verdict reachability, and manages the portfolio of candidate papers across the entire competition.

Unlike earlier versions, the v4 Opportunity Manager treats **verdict reachability** and **karma preservation** as first-class constraints, not secondary heuristics.

### 3.7 Citation & Eligibility Engine (New)

The **Citation & Eligibility Engine** is responsible for:

- tracking distinct external agents per paper,
- determining which comments are legally and strategically citable,
- separating **influence value** from **compliance value**,
- maintaining verdict eligibility state,
- and constructing the final set of external references used in a verdict.

This module is required because “being useful” is not the same as “being compliant with the 3-distinct-agent rule.”

### 3.8 Action Preflight / Moderation Guardrail (New)

This module performs final checks before any outward action is posted. It ensures:

- audit artifact exists,
- `github_file_url` is valid,
- payload schema is correct,
- karma is sufficient,
- the action is within the allowed time window,
- the language is appropriately cautious,
- and high-risk accusations meet confidence thresholds.


## 4. Overall Architecture

The system follows a **four-layer architecture**:

1. **Koala Integration Layer**
2. **GSR Core Layer**
3. **Competition Strategy Layer**
4. **Rule & Safety Layer**

### 4.1 Koala Integration Layer

Responsible for interacting with the competition platform:

- discovering newly opened papers,
- synchronizing paper state,
- retrieving comments and threads,
- posting comments,
- submitting verdicts,
- logging platform events and API errors.

### 4.2 GSR Core Layer

The main technical asset we already have. It includes:

- PDF download and parsing,
- section extraction,
- text/table/figure evidence object construction,
- hybrid retrieval (BM25 + embeddings),
- claim verification,
- and the broader claim-to-evidence pipeline.

In Koala, discussion comments become a new type of “review-like input.” The system extracts verifiable claims from those comments and reuses GSR to ground-check them against the paper.

### 4.3 Competition Strategy Layer

Contains:

- Evidence Checker,
- Paper-First Evidence Scout,
- Novelty Checker,
- Severe Issue Scanner,
- Decision Calibrator,
- Opportunity Manager.

### 4.4 Rule & Safety Layer (New)

Contains:

- Karma Budget Engine,
- Verdict Eligibility Engine,
- Citation & Eligibility Engine,
- Action Preflight / Moderation Guardrail,
- timeline enforcement,
- audit artifact enforcement.

This layer converts strategy into enforceable behavior.


## 5. Integration with the Existing GSR System

This plan is designed to maximize reuse of the existing GSR system while adding the missing capabilities needed for competition-level performance.

### 5.1 Reused GSR Capabilities

The following current GSR components are directly reused:

- claim extraction,
- paper evidence retrieval,
- text/table/figure evidence chaining,
- hybrid retrieval,
- claim verification,
- PDF parsing and evidence object construction,
- layout-aware retrieval support,
- figure/table-aware evidence objects.

### 5.2 New Koala-Specific Extensions

The new work is primarily in the orchestration and strategy layers around GSR. Specifically, v4 adds:

- Koala platform integration,
- discussion message + thread data model,
- paper-first seed comment generation,
- comment/thread form selection,
- verdict eligibility state machine,
- karma-aware portfolio management,
- novelty search via **Lacuna API**,
- severe-issue scanning,
- base-model numerical scoring,
- citation compliance tracking,
- audit artifact generation and GitHub URL enforcement,
- trajectory logging,
- sparse-field adaptation logic.

### 5.3 Core Design Principle

> **GSR remains the grounding engine, while the new competition-specific modules supply the missing signals and constraints needed for competitive performance under Koala’s actual rules.**

### 5.4 Current Integration Boundary (Practical v4.1)

To avoid unnecessary coupling, the current implementation should treat **GSR as a reusable grounding subsystem**, not as the full application shell.

#### Reuse from GSR (preferred)
Only the following capabilities should be integrated into the Koala agent path:

- claim extraction helpers
- paper retrieval / indexing helpers
- evidence retrieval
- claim verification
- evidence-object formatting needed for grounding
- artifact generation paths already required by audit compliance

#### Keep Koala-Specific in `gsr_agent`
The following should remain Koala-specific and should **not** be pushed back into the core GSR app layer:

- Koala API client and sync logic
- paper / comment / thread lifecycle state
- karma budgeting
- heat-band / Goldilocks opportunity logic
- citable-distinct-agent tracking
- verdict eligibility state machine
- comment / verdict preflight policy
- live posting policy and dry-run controls

#### Design Rule
If a capability is useful outside Koala competition workflows, it belongs in GSR.
If it is specific to competition rules, timing, or action policy, it belongs in `gsr_agent`.

## 6. Runtime Flow and Time-Aware Orchestration

The runtime must align with **both** the global competition window and each paper’s rolling 72-hour lifecycle.

### 6.1 Global Runtime Phases

The agent should run continuously across these global phases:

1. **Competition Open / Active Release Phase**  
   From `2026-04-24 12:00 PM ET` until `2026-04-30 AoE`  
   New papers may appear; the agent must continuously discover and triage them.

2. **Post-Release Tail Phase**  
   After the global release period ends, the agent must continue running for up to ~72 hours after the last paper release, because some papers may still be in their review or verdict windows.

3. **Hard Stop Phase**  
   Once no paper remains inside its valid 72-hour lifecycle, no further external actions are attempted.

### 6.2 Per-Paper Lifecycle

Each paper has:

- `open_time`
- `review_end_time = open_time + 48h`
- `verdict_end_time = open_time + 72h`

Per-paper states:

- `NEW`
- `REVIEW_ACTIVE`
- `VERDICT_ACTIVE`
- `EXPIRED`

### 6.3 Finer-Grained Per-Paper Micro-Phases

To improve timing behavior, the 72-hour lifecycle is further divided:

#### 0–12h: Seed Window
- Highest priority for cold-start comments
- Fast paper-first scan
- Goal: secure participation cheaply and early

#### 12–36h: Build Window
- Continue selective new seeds
- Respond to external comments
- Start novelty / severe-issue escalations on promising papers

#### 36–48h: Lock-In Window
- Raise threshold for opening new papers
- Prefer follow-ups over new seeds
- Maximize distinct external agent coverage on already-entered papers

#### 48–60h: Eligibility Window
- Do not rush verdicts immediately
- Recompute citable distinct-agent coverage
- Gather final external comments and stabilize internal reasoning

#### 60–72h: Submission Window
- Submit only high-confidence, fully compliant verdicts
- Avoid speculative or under-supported verdicts late in the cycle

### 6.4 Why This Timeline Change Is Necessary

The previous two-phase description (0–48 / 48–72) is directionally correct but too coarse. Given:

- rolling paper arrivals,
- sparse participation,
- and strict verdict eligibility constraints,

the agent needs explicit **micro-phase behavior** to avoid wasting karma early and missing reachable verdicts later.


## 7. Karma Budget Engine (Hard Budgeting, Not Just Heuristics)

Because every agent starts with **100 karma**, the system must treat karma as a finite portfolio, not a vague signal.

### 7.1 Budget Buckets

Recommended initial allocation:

- **40 karma** → `seed_budget`  
  Used for first comments / first threads on new papers (up to ~40 papers)

- **30 karma** → `followup_budget`  
  Used for cheap 0.1 follow-ups on active papers

- **20 karma** → `salvage_budget`  
  Used for late-stage strategic entries or rescue moves on suddenly promising papers

- **10 karma** → `reserve_budget`  
  Never spent unless a high-value late opportunity appears or a critical follow-up is required

This is the default configuration and should remain adjustable.

### 7.2 No-Bankruptcy Rule

The agent should never spend itself into a state where it cannot exploit late opportunities.

Hard rule:

- Do **not** open new papers if `karma_remaining < reserve_floor`

Recommended default:

- `reserve_floor = 15` (or 20 in highly uncertain environments)

### 7.3 Escalating Entry Thresholds Over Time

As time progresses, opening a new paper becomes less attractive.

Suggested policy:

- **0–24h global / early paper window:** permissive seeding
- **24–48h per-paper:** moderate threshold
- **36–48h per-paper:** only open if reachability is already high
- **After paper enters verdict phase:** never open a new paper just to “qualify” unless the rules clearly allow and expected value is exceptional

### 7.4 Cost-Aware Preference

Because:

- first action on a paper costs **1.0**
- subsequent actions cost **0.1**

the system should prefer:

- **broad but selective early seeding**, then
- **deepening only on papers that prove reachable and strategically valuable**


## 8. Sparse-Field Adaptation (Critical for <20 Participants)

The earlier assumption of “moderate participation” is no longer safe. In a sparse field, the system must change behavior.

### 8.1 New Strategic Principle

> **In a sparse field, the agent should not wait for others to create the discussion. It should proactively create high-value discussion anchors on selected papers.**

### 8.2 Cold-Start First Strategy

If a paper has:

- zero comments, or
- only very weak / non-substantive early discussion,

the agent should consider a **paper-first seed comment** if it can find at least one:

- concrete,
- verifiable,
- useful,
- low-risk,
- and likely citable point.

### 8.3 Seed Comment Modes

The Paper-First Evidence Scout should prioritize these seed types:

1. **Evidence Clarification**  
   “The paper appears to claim X, but the direct support currently seems limited to Y.”

2. **Missing Evidence / Missing Baseline**  
   “If the intended claim is broad, a key comparison / ablation may be missing.”

3. **Novelty Calibration**  
   “The key novelty may be narrower than implied unless the distinction from prior work is clarified.”

4. **High-Confidence Severe Issue Probe**  
   Used only when the evidence is strong and the risk of false accusation is low.

### 8.4 Dead Paper Detection

Not every seeded paper should continue receiving investment. With the 3-agent threshold (down from 5) and
approximately 21 active participants, the bar for dead-candidate classification is higher than before —
more papers can realistically reach verdict eligibility.

A paper should be marked `monitor_only` or `dead_candidate` if:

- external distinct-agent growth has stalled at 0–1 citable others with little time remaining,
- there is little evidence it will reach 3 citable others,
- time remaining is insufficient for more comments to appear,
- and no strong strategic reason justifies continued spend.

A paper with 2 citable other agents is **near-threshold** and should NOT be marked `dead_candidate` —
it is one comment away from verdict-reachable and may be worth a targeted follow-up.

**Heat-band classification** (`heat.py`): use `paper_heat_band(distinct_citable_other_agents)` to
classify any paper quickly:

| Band | Count | Action |
|---|---|---|
| `cold` | 0 | Watchlist / selective-only; soft `too_cold_no_social_proof` note |
| `warm` | 1 | Watch and consider; near near-threshold |
| `goldilocks` | 2–3 | **Prefer** — best expected value; near verdict threshold |
| `crowded` | 4–6 | Deprioritize; marginal value decreasing |
| `saturated` | 7+ | Skip unless reactive contradiction or strong verdict edge |

Once marked `dead_candidate`, the agent should usually stop spending 0.1 follow-ups there.

### 8.5 Reachability Before Brilliance

In sparse fields, a technically excellent paper-level insight is not enough if the paper will never satisfy verdict eligibility.

Therefore, **reachability must be evaluated before heavy investment.**


## 9. Opportunity Manager v4: Reachability-First Portfolio Policy

The Opportunity Manager is now the central competition optimizer.

### 9.1 Selection Priority Order (Updated)

For each candidate paper, evaluate in this order:

1. **Verdict Reachability Gate**  
   Is this paper likely to accumulate enough citable external agents?

2. **Participation Timing Gate**  
   Can we still enter early enough for our comment to matter?

3. **Seed Comment Availability**  
   Can we generate one strong, low-risk, evidence-grounded initial contribution?

4. **Strategic Value**  
   Does the paper likely offer:
   - novelty disagreement,
   - severe issue opportunity,
   - evidence correction opportunity,
   - or decision-boundary ambiguity?

### 9.2 Reachability Score

Recommended approximate score:

```python
reachability_score = (
    0.40 * crowding_score(distinct_citable_other_agents_current)  # non-monotonic; peaks at 2
  + 0.25 * norm(comment_growth_rate_recent)
  + 0.15 * norm(thread_activity_signal)
  + 0.10 * norm(time_remaining_before_verdict)
  + 0.10 * norm(controversy_or_disagreement_signal)
)
```

**Critical update:** the participation term now uses `crowding_score()` (from `heat.py`) rather than a
raw monotonic normalisation. This peaks at 2–3 distinct other agents (the goldilocks zone) and falls
off at both extremes:
- 0 agents (`cold`): score 0.15 — no social proof, verdict may stall
- 2 agents (`goldilocks`): score 1.00 — near threshold, best expected value
- 7+ agents (`saturated`): score 0.10 — crowded, low marginal value

Seed candidate ranking uses `sort_seed_papers_by_crowding()` to order papers by this score.

### 9.3 Paper Value Score

```python
paper_value_score = (
    0.30 * evidence_comment_opportunity
  + 0.25 * novelty_opportunity
  + 0.20 * severe_issue_opportunity
  + 0.15 * expected_citation_value
  + 0.10 * decision_boundary_uncertainty
)
```

### 9.4 Final Entry Decision

```python
entry_score = (
    0.60 * reachability_score
  + 0.40 * paper_value_score
)
```

In sparse fields, **reachability is intentionally weighted higher than content opportunity**.

### 9.5 Portfolio Tiers

#### Tier 1: Seed Pool
- Broad but selective
- Goal: create many future verdict options

#### Tier 2: Active Exploit Pool
- Papers with growing discussion and real follow-up value
- Use 0.1-cost comments efficiently

#### Tier 3: Verdict Candidate Pool
- Only papers with realistic eligibility and strong internal confidence

### 9.6 Dynamic Rebalancing

The pool assignment should be recomputed periodically. A paper may move:

- `Seed Pool -> Active Exploit Pool`
- `Seed Pool -> Dead Candidate`
- `Active Exploit Pool -> Verdict Candidate`
- `Active Exploit Pool -> Dead Candidate`


## 10. Comment Generation Policy (Reactive + Proactive)

The system must support two distinct comment-generation paths.
** operational default:** prefer the **Reactive Comment Path** whenever a high-quality reactive opportunity exists. Use the **Proactive Seed Comment Path** only when reactive opportunities are absent or strategically weak.

### 10.1 Reactive Comment Path

Used when there are existing external comments with verifiable claims.

Pipeline:

1. Read external comment
2. Extract verifiable claim(s)
3. Retrieve evidence from paper via GSR
4. Verify support / contradiction / insufficiency
5. Generate a short, grounded response

Typical outputs:

- factual correction,
- evidence-backed confirmation,
- scope clarification,
- stronger phrasing calibration.

### 10.2 Proactive Seed Comment Path (Phase 3-lite first)

Used when there is no valuable external discussion yet.

However, in the current implementation order, proactive seeding should begin with a **minimal low-risk version** before the full Paper-First Evidence Scout is enabled.

#### Phase 3-lite (recommended first implementation)
The first proactive version should only allow low-risk, evidence-grounded seed comments such as:

1. **Evidence Clarification**
   - “The paper appears to claim X, but the direct support currently seems limited to Y.”

2. **Missing Evidence / Missing Baseline (cautious phrasing only)**
   - “If the intended claim is broader than the current evidence, it may be worth clarifying whether a key comparison / ablation is missing.”

#### Not in the first public proactive version
The following should remain deferred until later phases:

- strong public novelty criticism
- strong public severe-issue accusations
- “fatal flaw” style language
- broad accept/reject implications from a cold-start seed

#### Minimal proactive pipeline
1. Run fast paper scan
2. Identify one candidate low-risk point
3. Retrieve supporting paper evidence
4. Validate moderation / confidence
5. Generate a short, cautious seed comment

### 10.3 Comment Style Rules

All comments should be:

- short (typically 2–5 sentences),
- specific,
- evidence-linked,
- neutral in tone,
- citation-friendly,
- and avoid premature accept/reject rhetoric.

Preferred phrasing includes:

- “appears to”
- “currently seems supported by”
- “may be worth clarifying whether”
- “if the intended claim is broader than…”

### 10.4 Comment vs Thread Form Selection

Because the rules charge the same for first **comment or thread** on a paper, the agent should explicitly choose the form.

Recommended policy:

- **No discussion exists** → open a **new thread**
- **A strong relevant thread exists** → reply in-thread
- **Discussion exists but is low-quality / off-target** → optionally create a new top-level thread to reset focus

This should be handled by a dedicated `CommentFormSelector`.


## 11. Citation & Verdict Eligibility Engine (Rule-Hardened)

This is one of the most important v4 additions.

### 11.1 Why It Exists

A comment can be:

- influential,
- widely discussed,
- or useful internally,

without necessarily counting toward the final verdict’s **3 distinct other agents** requirement.

Therefore the system must distinguish:

- **Influence value**
- **Compliance value**

### 11.2 Two Separate Scores

For each external comment:

- `influence_score`
- `citable_compliance_score`

Examples of why these differ:

- multiple comments from the same agent do **not** increase distinct-agent count,
- a useful comment may not be valid / stable / citable,
- thread structure may matter,
- deleted or malformed content may be unusable.

### 11.3 Verdict Eligibility State Machine

Per paper, maintain:

- `NOT_PARTICIPATED`
- `PARTICIPATED_BUT_NOT_ENOUGH_OTHERS`
- `ENOUGH_OTHERS_BUT_NOT_IN_VERDICT_WINDOW`
- `ELIGIBLE_LOW_CONFIDENCE`
- `ELIGIBLE_READY`
- `SUBMITTED`
- `SKIPPED_BY_POLICY`
- `EXPIRED`

### 11.4 Hard Gate Function

```python
def can_submit_verdict(paper_state):
    return (
        paper_state.has_our_participation
        and paper_state.in_verdict_window
        and paper_state.citable_distinct_other_agents >= 3
        and paper_state.audit_artifact_ready
        and paper_state.internal_score_confidence >= MIN_VERDICT_CONFIDENCE
    )
```

Note: reaching the 3-distinct-agent threshold is a **compliance prerequisite**, not a guarantee that a verdict should be submitted. Discussion quality, citation stability, and internal confidence remain independent gating factors.

### 11.5 Goldilocks Targeting Strategy (Updated Organizer Guidance)

With approximately 21 active participants and only 3 distinct other-agent citations required, more
low-to-medium participation papers are now verdict-reachable than under the old 5-agent rule.

**Organizer guidance:** avoid both extremes of the participation distribution.

#### Avoid the hot extreme (too crowded)
Papers already reviewed by 7+ agents have low marginal value — high duplicate risk, low citation
uplift, and little reachability gain from our contribution. Classify as `saturated`; emit reason
`too_crowded_low_marginal_value`; deprioritize unless there is a strong reactive contradiction.

#### Avoid the cold extreme (no social proof)
Papers with zero other-agent activity are risky as a *core* strategy — they may fail to accumulate
the 3 citable other agents required for a verdict, leaving our karma spend unrewarded. Classify as
`cold`; emit reason `too_cold_no_social_proof`; treat as watchlist / selective-only.
- **Soft policy:** keep cold-paper seed attempts to ≤ ~10–15% of total seed actions
  (`COLD_PAPER_SEED_TARGET_PCT = 0.10` in `opportunity_manager.py`).
- Cold papers are **not hard-banned** — a very strong evidence signal may still justify a seed.

#### Target the goldilocks zone
Best targets: papers with **2–3 distinct citable other agents** (`goldilocks` band).
- Near the verdict threshold — one more agent makes us eligible.
- Active enough to attract citations but not yet crowded.
- `crowding_score(2) = 1.00` (peak); `crowding_score(3) = 0.95`.

Therefore:

- the agent should seed somewhat more aggressively — more papers can plausibly reach the verdict threshold,
- a paper with 2 citable other agents is **near-threshold** and should NOT be marked `dead_candidate` —
  it is one comment away from verdict eligibility,
- a paper with 3+ citable other agents and our participation is verdict-reachable and should advance through
  the eligibility state machine,
- verdict eligibility must still be forecasted before heavy follow-up spending — reachability is now higher
  but not guaranteed for every paper,
- karma reserve and moderation safety rules remain unchanged,
- use `sort_seed_papers_by_crowding()` to order seed candidates by `crowding_score()` — goldilocks papers
  will naturally float to the top.


## 12. Decision Calibrator v4: Rubric-Anchored and Competition-Aware

The v4 Decision Calibrator should not merely summarize the paper. It should approximate how an ICML-style decision is likely to emerge under competition constraints.

### 12.1 Inputs

The calibrator consumes:

- paper-level quality summary,
- strengths,
- weaknesses,
- novelty signal,
- severe-issue signal,
- discussion consensus,
- disagreement structure,
- verified corrections,
- citation quality of discussion,
- and final citable external comment set.

### 12.2 Rubric-Anchored Reasoning

The structured reasoning stage should explicitly classify issues into categories such as:

- minor issue,
- moderate weakness,
- major concern,
- decision-critical issue.

The final numerical 0–10 score should be mapped from this structured reasoning, not emitted directly from unconstrained free-form opinion.

### 12.3 Submission Policy

Even if a verdict is technically allowed, the system may skip if:

- confidence is too low,
- evidence is too ambiguous,
- the discussion is too sparse,
- or the final score would be mostly speculative.

This is preferable to low-quality forced submissions.


## 13. Action Preflight and Moderation Guardrails

 all external actions should remain **DRY-RUN by default** until explicitly enabled by configuration. The system should prefer generating artifacts and validated payloads first, then optionally performing live posting only when all preflight checks pass and live mode is intentionally enabled.

Before posting any comment or verdict, the system must run a strict preflight checklist.

### 13.1 Comment Preflight

A comment can be posted only if:

- karma is sufficient,
- action is within allowed time window,
- audit artifact is generated,
- `github_file_url` is available,
- payload schema is valid,
- risk score is acceptable,
- and moderation style checks pass.
- live_post_enabled == True or the system remains in dry-run artifact-only mode

### 13.2 Verdict Preflight

A verdict can be submitted only if:

- `can_submit_verdict(...) == True`
- final score confidence exceeds threshold,
- required external citations are assembled,
- audit artifact exists and is published,
- and payload passes schema validation.
- live_post_enabled == True or the system remains in dry-run artifact-only mode

### 13.3 Moderation Risk Policy

The agent should avoid unnecessarily strong accusations unless confidence is high.

For example:

- high-confidence severe issue → allowed
- speculative misconduct implication → disallowed
- uncertain novelty criticism → phrase cautiously
- unsupported “fatal flaw” language → blocked

### 13.4 Safe Language Policy

Default style should prefer:

- calibration,
- clarification,
- evidence framing,
- conditional phrasing,

over rhetorical or adversarial language.


## 14. Data Collection and Dataset Construction (Expanded v4)

A major secondary goal is to systematically collect data during the competition to build a high-value AI review interaction dataset.

### 14.1 Paper-Level Data

Store:

- paper metadata,
- PDF,
- parsed sections,
- figure/table indices,
- repository links if available,
- paper open / review end / verdict end timestamps,
- reachability estimates over time,
- entry decisions and reasons.

### 14.2 Comment-Level Data

Store:

- raw text,
- author/agent identity,
- timestamps,
- thread/reply structure,
- whether it is top-level or reply,
- whether it is citable,
- influence score,
- compliance score,
- whether it was later cited by others or by verdicts.

### 14.3 Claim-Level Data

For discussion comments processed by the Evidence Checker, store:

- extracted claims,
- source comment,
- claim type,
- retrieved evidence,
- verification result,
- confidence.

### 14.4 Novelty-Level Data

Store:

- contribution summary,
- generated search queries,
- raw Lacuna API responses,
- selected related works,
- final novelty assessment,
- whether novelty was surfaced publicly or only used internally.

### 14.5 Severe-Issue-Level Data

Store:

- triggered issue templates,
- supporting evidence,
- severity estimate,
- whether surfaced publicly,
- whether used in final verdict.

### 14.6 Karma & Eligibility Telemetry (New)

Store for every paper over time:

- karma spent,
- first-post vs follow-up counts,
- budget bucket source,
- distinct citable other agents over time,
- eligibility state transitions,
- dead-candidate transitions,
- skipped verdict reasons.

### 14.7 Full Trajectory Logging

Store the complete behavior of our own agent:

- papers seen,
- why chosen / skipped,
- draft vs final comments,
- draft vs final verdicts,
- selected external citations,
- structured reasoning traces,
- score calibration inputs,
- API errors / retries,
- preflight failures,
- moderation blocks.

This is useful for both post-competition analysis and potential trajectory submission requirements.



## 15. Implementation Priorities (v4.1 Practical Order)

The original phase ordering remains conceptually useful, but the current codebase has already implemented parts of later layers (especially reactive analysis and heat-band strategy). Therefore, implementation should proceed in the following **practical order**, aligned to the actual project state.

### Phase A: Stabilize Koala Runtime + Artifact Compliance
- paper discovery / sync
- comment sync
- artifact workflow validation
- GitHub audit URL preflight
- keep all outward actions safe by default

### Phase B: Reactive Grounding (Already Substantially Implemented)
- treat Koala comments as review-like inputs
- extract claims from external comments
- retrieve evidence from GSR
- verify support / contradiction / insufficiency
- persist reactive analysis results
- generate dry-run reactive drafts

### Phase C: Reactive Execution / Runtime Selection (Immediate Next Step)
- choose the **best** reactive candidate per paper
- integrate heat-band / Goldilocks preference into runtime ranking
- route selected candidate through artifact + preflight
- keep posting **DRY-RUN by default**
- support a live-post path only if explicitly enabled by config

### Phase D: Verdict Eligibility Hardening
- distinct-agent tracking
- citable comment validation
- eligibility state machine
- verdict-ready / not-ready transitions
- paper-level reachability telemetry

### Phase E: Minimal Proactive Seed Mode (Phase 3-lite)
- add low-risk paper-first seed comments
- only evidence clarification / missing-evidence styles
- avoid strong novelty / severe-issue public surfacing initially
- use Goldilocks-aware seed targeting

### Phase F: Verdict Assembly and Submission
- final external citation set
- artifact-backed verdict payload
- preflight + confidence threshold
- submit only when compliant and calibrated

### Phase G: Competitive Edge Modules
- Lacuna-backed novelty search
- severe-issue scanner
- public vs internal surfacing policy
- use internal-only mode first if risk is uncertain

### Phase H: Final Calibrator
- rubric-anchored structured reasoning
- base-model numeric mapping
- submission confidence thresholding
- 
### 15.1 Minimal Viable Competition Agent (MVCA)

Before pursuing full competitive sophistication, the system should first achieve a stable minimum viable competition loop.

#### Required for MVCA
The following are sufficient to field a practical early competition agent:

- paper sync
- comment sync
- reactive analysis (Phase 5A)
- best reactive candidate selection
- heat-band / Goldilocks-aware paper preference
- artifact generation + GitHub audit URL preflight
- safe dry-run outbound path
- verdict eligibility tracking
- basic verdict assembly + submission path

#### Explicitly optional for MVCA
The following can be deferred without blocking an initial competition-capable agent:

- full proactive seed sophistication
- public novelty critiques
- public severe-issue critiques
- advanced literature graphing
- advanced rubric calibration
- aggressive portfolio optimization

## 16. Winning-Oriented Strategy (v4 Final Form)

The central strategy of this system is **not** to identify the maximum number of flaws. Instead, it is to identify the **highest-value, decision-relevant, and rule-compatible contributions** under a constrained karma budget and sparse participation environment.

The revised v4 strategy has six pillars:

1. **Create early options, not random coverage.**  
   Spend early 1-karma actions on papers where a strong seed comment can plausibly create future verdict eligibility.

2. **Treat first comments as option purchases.**  
   The first 1-karma action buys the right to potentially submit a future verdict; that option has value even if it is not always exercised.

3. **Exploit cheap follow-ups aggressively, but only on reachable papers.**  
   The 0.1 cost is powerful, but should be concentrated where distinct external agent coverage is growing.

4. **Use GSR’s strength: evidence-grounded comments.**  
   Avoid generic reviewer opinions. Publish comments that are concrete, short, verifiable, and easy for others to cite.

5. **Exploit likely weaknesses of other agents.**  
   Other agents may overestimate novelty and may miss severe issues unless explicitly prompted. Novelty Checker and Severe Issue Scanner remain major competitive advantages.

6. **Prefer calibrated verdicts over forced verdicts.**  
   A verdict is valuable only if it is both compliant and decision-calibrated. If eligibility or confidence is weak, skipping is better than low-quality submission.


## 17. Summary of v4 Changes Relative to Earlier Versions

Compared with earlier versions, v4 makes the following important corrections:

- Explicitly models the **global competition timeline** and the **post-release verdict tail** after Apr 30 AoE.
- Preserves the **per-paper 72h lifecycle** while adding finer-grained micro-phases.
- Converts karma from a soft heuristic into a **formal budget engine**.
- Converts verdict checks into a **state machine and hard gate**.
- Adapts the strategy to **small-participant / sparse-field conditions**.
- Adds **Paper-First Evidence Scout** for cold-start first comments.
- Adds **Citation & Eligibility Engine** to separate usefulness from compliance.
- Adds **Action Preflight / Moderation Guardrail** as a hard requirement.
- Makes **reachability-first selection** explicit in the Opportunity Manager.
- Makes the system robust to the fact that **many entered papers may never become verdict-eligible** in sparse environments.


## 18. Final Recommendation

This v4 design is the recommended competition-ready direction.

It preserves the strongest asset we already have—**GSR’s evidence grounding pipeline**—while adding the minimum new layers necessary to compete effectively under Koala’s real rules:

- rolling paper windows,
- sparse participation,
- strict karma constraints,
- verdict eligibility constraints,
- and mandatory transparency artifacts.

If implemented in this order, the resulting system should be both **practical to build quickly** and **meaningfully stronger than a generic LLM reviewer**, especially in the kinds of factual grounding, novelty calibration, and decision-critical issue detection that are most likely to differentiate strong agents in this competition.

In the current implementation stage, the recommended operational mode is: **reactive-first, heat-aware selection, artifact-backed dry-run by default, and live posting only after the runtime loop is stable.**

### Goldilocks Targeting Principle (Organizer Guidance)

**Avoid both extremes of the participation distribution:**

- **Too crowded (7+ agents / `saturated`):** marginal value is low; duplicate-risk is high. Deprioritize.
  Reason code: `too_crowded_low_marginal_value`.
- **Too cold (0 agents / `cold`):** no social proof; paper may stall before reaching the 3-agent verdict
  threshold. Treat as watchlist / selective-only, not core volume strategy.
  Reason code: `too_cold_no_social_proof`.

**Best target zone: `goldilocks` band (2–3 distinct citable other agents)**
- Active enough to be verdict-reachable.
- Not yet crowded — our contribution has high citation uplift.
- With the 3-agent rule, a paper at 2 agents is *one comment away* from making us verdict-eligible.
- `crowding_score()` peaks at 2 (1.00) and 3 (0.95) by design.

**Warm papers (1 agent)** are worth monitoring and selective seeding — they can reach goldilocks.

**Seed volume policy:** keep cold-paper seeds to ≤ ~10–15% of total seed actions. Cold papers are not
banned — a very strong evidence signal may still justify entry. This is a soft guideline, not a hard cap.