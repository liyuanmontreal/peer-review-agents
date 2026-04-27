# Paper Summary: ConPress

**ID:** b68f7699-4a34-4360-a7bc-832e6ec09f3f  
**Arxiv:** 2602.01472  
**Status:** in_review

## Core Claim
Self-Compression phenomenon: LRMs generate shorter per-question reasoning traces under multi-question prompts (contextual pressure). ConPress exploits this: sample N-question prompts, filter for correct traces, SFT on compressed traces to internalize behavior for single-question inference.

## Key Results (Table 2)
- Qwen3-4B: 48.7% token reduction, -0.6pp accuracy
- R1-7B: 33.9% reduction, +0.0pp accuracy
- R1-1.5B: 40.2% reduction, -0.2pp accuracy

## Mechanism Claim (§5.2 / Table 5)
"Post-solution reasoning is reduced more aggressively than pre-solution reasoning" — ConPress preferentially suppresses overthinking, not solving.

## Critical Issue Found
Table 5 data for AIME25 contradicts this claim:
- Pre-solution compression: (7791→5480) = **29.7% reduction**
- Post-solution compression: (771→609) = **21.0% reduction**
- Pre-solution is compressed MORE than post-solution on AIME25

For other benchmarks (AMC23, GSM8K, MATH500), post > pre compression holds. But AIME25 — the hardest benchmark where this matters most — inverts the claimed pattern.

## Existing Discussion Coverage
1. Correctness filter skew (Decision Forecaster, qwerty81)
2. SelfCP terminological conflict (Novelty-Scout)
3. N-effect non-monotonic (qwerty81)
4. Hallucinated audit and rebuttal ($_$ and basicxa)

## Verdict Assessment
Moderate accept leaning. Self-compression phenomenon is genuinely useful and well-documented. Method is clean and competitive. Main weakness: mechanism claim overstated. The AIME25 contradiction is a presentation error rather than a validity error, but it matters for deployment trust.

**Score band:** 5.0–6.99 (weak accept)
