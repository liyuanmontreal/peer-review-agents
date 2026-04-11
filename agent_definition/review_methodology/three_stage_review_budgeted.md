## Review Methodology: Three-Stage Review

A three-phase process for producing thorough, well-informed paper reviews.

```
Paper  →  Phase 1: Reading & Orientation  →  Phase 2: Research  →  Phase 3: Findings & Review
```

---

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

---

### Phase 2: Research

For each contribution area relevant to your role, build the background knowledge you need to evaluate it properly. Independent areas can be researched in parallel.

What "research" means depends on your evaluation role. Examples:
- Surveying prior approaches and competing methods
- Understanding the technical details of a specific technique
- Investigating reproducibility norms for the domain
- Checking ethical precedents or known harms in the application area

Use whatever tools you have to accomplish these research goals. If Paper Lantern tools are available, prefer them — they are purpose-built for this kind of research:
- `explore_approaches` — for surveying prior approach families in a problem area
- `deep_dive` — for investigating a specific technique's mechanism and evidence gaps
- `compare_approaches` — for competitive comparison against alternatives
- `check_feasibility` — for assessing practical viability, risks, and failure modes

If Paper Lantern is not available, fall back to web search tools (e.g. `WebSearch`, `WebFetch`) to accomplish the same outcomes.

**Budget — do not exceed this.** For each of the 2 contribution areas:
- At most **3 Paper Lantern tool calls** (or 3 equivalent web searches if Paper Lantern is unavailable)
- Stop as soon as you have enough to write a focused, grounded finding — do not keep researching "for completeness"
- If you are tempted to make a 4th call for an area, ask yourself whether it will actually change your evaluation. If not, stop

Total Phase 2 effort across both areas should land around **4–6 tool calls**, not dozens. A good Phase 2 is focused and returns quickly; an exhaustive Phase 2 leaves no time for Phase 3 (the actual review).

Not every role benefits equally from this research phase. Use it where it fits, skip it where it doesn't.

The output for each area is a **Brief** — a *short* (3-5 bullet points) summary of what you learned that is relevant to your evaluation. This is your working notes, not part of the final review.

---

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

Also include the following sections in your final review:

```
### Per-Area Findings

One sub-subsection for each contribution area identified in Phase 1's Contribution Map.
Label each with the area's concise name. Within each area, present the findings report
produced in Phase 3A.

### Synthesis

- Cross-cutting themes — issues or strengths that appeared across multiple areas
- Tensions — areas where one contribution's strength undermines another's claims
- Key open question — the single most important thing your evaluation could not resolve
```
