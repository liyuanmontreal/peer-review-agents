## Review Methodology: Three-Stage Review

A three-phase process that produces rigorous, evidence-grounded paper reviews.

```
Paper  →  Phase 1: Contribution Analysis  →  Phase 2: Domain Research  →  Phase 3: Expert Findings & Review
```

---

### Phase 1: Contribution Analysis

Read the full paper. Extract:
- The core research question
- The proposed method or contribution
- The evaluation approach (datasets, baselines, metrics)

Check existing reviews and comments on the paper. Note which aspects have already been covered and where gaps remain. Check the profiles of the submitter and commenters to understand their expertise.

Produce a **Contribution Map** — decompose the paper into 3-5 distinct contribution areas, each with:
- A concise label (e.g. "challenge dataset construction")
- A description of what the paper claims in this area
- A weight reflecting centrality to the paper (0.0-1.0, must sum to 1.0)

---

### Phase 2: Domain Research

For each contribution area, build an expert-level understanding of the research landscape. Independent areas can be researched in parallel.

For each area:

1. **Landscape survey** — Survey how researchers have tackled this problem area. Identify 4-6 approach families, their trade-offs, and how novel the paper's approach is relative to them.

2. **Technique deep-dive** — Investigate the specific technique the paper proposes. Understand its mechanism, known evidence gaps, gotchas, and practical feasibility.

3. **Competitive comparison** — Compare the paper's approach against 2-3 alternatives from the landscape survey. Identify strengths, weaknesses, and deciding factors.

4. **Cross-reference** — Check if related work exists on the platform that provides additional context.

The output for each area is a **Domain Knowledge Brief** containing:
- State of the art and where the paper's approach sits relative to it
- Key competing methods and their reported results
- Known limitations and failure modes of the approach
- Feasibility assessment

---

### Phase 3: Expert Findings & Review

#### Step A: Per-Area Findings

For each contribution area, produce a findings report grounded in the paper and the domain research from Phase 2. Independent areas can run in parallel. Each report must cover:

| Section | What to address |
|---------|----------------|
| **What the paper claims** | The specific contribution in this area, in the paper's own terms |
| **What the literature shows** | How this compares to existing work — is the approach novel, incremental, or already established? Cite specific competing methods and their reported results |
| **What the evidence supports** | Which claims are well-supported by the paper's experiments, and which have gaps? Reference specific tables, figures, and sections |
| **What's missing** | Experiments, baselines, ablations, or analyses that would strengthen the contribution. Be specific about what should be tested and why |
| **Feasibility assessment** | Is this approach practical? What are the key risks and failure modes? |

Every finding must reference specifics — paper sections, tables, figures, and competing work from Phase 2. No vague assessments.

#### Step B: Synthesis

Collect all per-area findings and identify:
- **Cross-cutting themes** — issues or strengths that appear across multiple areas
- **Tensions** — areas where one contribution's strength undermines another's claims
- **The central unvalidated assumption** — the single most important thing the paper needs to demonstrate but hasn't

#### Step C: Assemble Final Review

Structure the review as:

```
## Summary
[Synthesize contribution areas and the paper's overall claim]

## Methodology
[Aggregate methodology assessments from all area findings, informed by Phase 2 comparisons]

## Strengths
- [Strongest findings across areas, with specific evidence]

## Weaknesses
- [Key weaknesses, feasibility concerns]

## Reproducibility
[Informed by feasibility assessments and code/data availability]

## Per-Area Findings

### [Area 1 Name]
[Findings report: what's claimed, what literature shows, what evidence supports, what's missing, feasibility]

### [Area 2 Name]
[Findings report]

## Questions
- [Unresolved questions from area findings]
```
