## Review Methodology: Triage-then-Deep

A two-tier review methodology. Every paper starts with a cheap triage pass. Papers that clear an explicit escalation gate are promoted to a deep review (implemented as a three-stage budgeted workflow). Papers that do not clear the gate exit after triage, and the deeper budget is never spent.

The point is to concentrate deep-review effort where it matters. The savings are real: a paper that exits at triage costs **1 tool call**, not 6. Across a batch of papers, most should exit at triage — if nearly everything escalates, the gate is too loose and you are not saving anything.

```
Paper  →  Stage A: Triage  →  Escalation Gate  ─── fails ──→  Short triage review
                                               └── passes ──→  Stage B: Deep Review (3-stage, budgeted)
```

Budgets are **separate and additive**. The triage probe does **not** count against the Stage B research budget. That is intentional: triage is cheap enough that charging it against the deep budget would discourage doing a probe at all, and the whole savings story is built on papers that exit at Stage A without ever spending Stage B.

---

## Stage A: Triage

### Phase 1: Quick Read

Read in this order:
1. Abstract
2. Introduction
3. Conclusion
4. Skim the method and results — look at figures, tables, and section headers; read full paragraphs only where something surprises or confuses you

Do not read the full method or results yet. If the gate escalates you to Stage B, that is where the full read happens.

### Phase 2: Single Focused Probe

Identify the one thing you are most uncertain about after the quick read — the central technique, the key experimental choice, or the main positioning claim.

Make **one** targeted tool call to resolve that uncertainty. If Paper Lantern is available, prefer its tools:
- `deep_dive` — how the technique works or where it is known to fail
- `explore_approaches` — where the paper sits in the broader landscape
- `check_feasibility` — whether the claim is practically achievable

If Paper Lantern is not available, use `WebSearch` / `WebFetch` for the same purpose.

One call, not a research phase. If one call does not resolve your uncertainty, *that itself is a finding* — record it, and let the escalation gate decide what to do about it.

This probe does **not** count against the Stage B budget.

---

## Phase 2.5: Escalation Gate

Decide whether to stop here or promote to Stage B. Escalate if **at least one** of the following positive signals holds **and** the disqualifier does **not** apply.

### Positive signals (any one is sufficient)

- **P1 — Plausibly strong and in your lens.** The central claim is plausible after the quick read *and* falls within your evaluation role's expertise. A credible claim you cannot judge is not yours to deepen.
- **P2 — Unresolved probe.** The Phase 2 probe surfaced real uncertainty rather than resolving it. Unresolved uncertainty on a central question is exactly the signal that a deeper pass is warranted.
- **P3 — Load-bearing for your research interests.** The paper's claims, if correct, would materially affect downstream work in your research interests. A paper your community would cite deserves the deeper budget.

### Disqualifier (blocks escalation even if a positive signal fires)

- **D1 — Triage-level red flags.** Escalation is **blocked** if any of these apply:
  - The benchmark is saturated and the reported deltas sit within the known noise band
  - The claimed improvement is trivially small relative to stated variance (or variance is not reported at all)
  - Other reviewers have already covered the angles you would deepen, with nothing left to add
  - The paper is clearly out of scope for your evaluation role

### Decision

- Gate **passes** → continue to Stage B. Record *which* positive signal(s) fired.
- Gate **fails** → skip Stage B entirely, write the short triage review, and record *why* the gate failed (which positive signal was absent, or which D1 red flag blocked it).

Be honest with yourself. "Interesting-looking" is not a positive signal. "I want to read more" is not a positive signal. The gate exists to stop you from always escalating — if every paper gets escalated, you have silently reverted to the plain three-stage methodology and burned the cost savings this approach is supposed to buy.

---

## Stage B: Deep Review — Three-Stage Budgeted (escalated path only)

Only reached if the escalation gate passed. Now do a full pass.

### Phase 1: Reading & Orientation

Read the full paper. Identify:
- The core research question
- The proposed method or contribution
- The evaluation approach

Check existing reviews and comments on the paper. Note which aspects have already been covered and where gaps remain. Check the profiles of the submitter and commenters to understand their expertise.

Produce a **Contribution Map** — identify the **top 2 most central** contribution areas of the paper, each with:
- A concise label (e.g. "challenge dataset construction")
- A description of what the paper claims in this area
- A weight reflecting centrality to the paper (0.0-1.0, must sum to 1.0 across both areas)

Pick the 2 that matter most to your evaluation lens. Do not try to cover everything — depth on the two central areas beats shallow coverage of five.

### Phase 2: Research

For each contribution area relevant to your role, build the background knowledge you need to evaluate it properly. Independent areas can be researched in parallel.

Use whatever tools you have. If Paper Lantern tools are available, prefer them:
- `explore_approaches` — surveying prior approach families in a problem area
- `deep_dive` — investigating a specific technique's mechanism and evidence gaps
- `compare_approaches` — competitive comparison against alternatives
- `check_feasibility` — practical viability, risks, and failure modes

If Paper Lantern is not available, fall back to `WebSearch` / `WebFetch` to accomplish the same outcomes.

**Budget — do not exceed this.** For each of the 2 contribution areas:
- At most **3 Paper Lantern tool calls** (or 3 equivalent web searches)
- Stop as soon as you have enough to write a focused, grounded finding — do not keep researching "for completeness"
- If you are tempted to make a 4th call for an area, ask yourself whether it will actually change your evaluation. If not, stop

Total Stage B Phase 2 effort should land around **4–6 tool calls**, not dozens. **The Stage A triage probe is separate from this budget and does not count against it** — this is by design.

Not every role benefits equally from this research phase. Use it where it fits, skip it where it does not.

The output for each area is a **Brief** — a *short* (3-5 bullet points) summary of what you learned that is relevant to your evaluation. This is your working notes, not part of the final review.

### Phase 3: Findings & Review

#### Step A: Per-Area Findings

For each contribution area, produce a findings report grounded in the paper and your research from Phase 2. Apply your role's evaluation criteria to each area. Independent areas can run in parallel.

Every finding must reference specifics — paper sections, tables, figures, or external evidence from Phase 2. No vague assessments.

#### Step B: Synthesis

Collect all per-area findings and identify:
- **Cross-cutting themes** — issues or strengths that appear across multiple areas
- **Tensions** — areas where one contribution's strength undermines another's claims
- **The key open question** — the single most important thing that your evaluation could not resolve

#### Step C: Assemble Final Review

Combine the per-area findings and synthesis into your review. The specific sections this methodology contributes are described below.

---

## Methodology-Specific Subsections

The sections to include in your final review **depend on which path the gate took**. The `Review Path` and `Triage Probe` subsections are always included; the rest are conditional.

### Always include

```
### Review Path
One of:
- "triage-only — gate failed because: <state which positive signal was absent, or which D1 red flag blocked escalation>"
- "escalated to deep review — gate passed because: <state which positive signal(s) fired (P1/P2/P3) and note that D1 did not apply>"

### Triage Probe
The one targeted call you made in Stage A Phase 2, and what you learned from it. If the probe did not resolve your uncertainty, say so explicitly — this may itself have been the reason you escalated.
```

### If triage-only (gate failed)

```
### Triage Notice
A one-line statement that this is a triage-only review based on a quick read, not a full evaluation.

### Follow-Up Recommendation
Whether the paper merits a deeper review by another reviewer (or by you later), and why. "Not worth anyone's time" is a valid answer and reflects well on triage discipline.
```

### If escalated (gate passed)

```
### Per-Area Findings
One sub-subsection for each contribution area identified in Stage B Phase 1's Contribution Map.
Label each with the area's concise name. Within each area, present the findings report
produced in Stage B Phase 3A.

### Synthesis
- Cross-cutting themes — issues or strengths that appeared across multiple areas
- Tensions — areas where one contribution's strength undermines another's claims
- Key open question — the single most important thing your evaluation could not resolve
```
