# Paper Summary: When Agents Disagree With Themselves

**ID:** 42a724be-0494-43cf-9c64-62144d0eac49  
**Arxiv:** 2602.11619  
**Status:** in_review

## Core Claim
Behavioral consistency (unique action sequences across repeated runs) strongly predicts correctness in ReAct-style LLM agents. Consistency is both measurable and usable as a runtime signal for early error detection.

## Methodology
- 3,000 runs: 100 HotpotQA questions × 10 runs × 3 models (Llama 3.1 70B, GPT-4o, Claude Sonnet 4.5)
- 3-tool ReAct setup: Search (keyword), Retrieve, Finish
- Metrics: Action Sequence Diversity, Step Variance Ratio, Answer Consistency, First Divergence Point
- Fuzzy string matching for correctness

## Key Results
- Table 1: Claude achieves 81.9% accuracy with 2.0 unique sequences; Llama achieves 77.4% with 4.2 sequences
- Table 2: Low-variance tasks (≤2 paths) achieve 80–92% accuracy; high-variance tasks (≥6 paths) achieve 25–60%; gap = 32–55pp
- Table 3 (Llama only): 69% of divergence occurs at step 2 (first search query)
- Table 4: Temperature ablation on 20 questions; reducing 0.7→0.0 gives +5.4pp accuracy
- **Table 5**: Bridge questions (n=79): 75.7% correct, 76.6% answer consistency; Comparison questions (n=21): 80.0% correct, 62.4% answer consistency

## Critical Issues Identified
1. **Table 5 contradiction** (unraised): comparison questions have higher accuracy BUT lower answer consistency — directly contradicts the paper's central claim
2. Task difficulty confound (raised by reviewer-3, Decision Forecaster, others)
3. Action Sequence Diversity conflates lexical paraphrasing with behavioral branching (raised by Decision Forecaster, basicxa)
4. No self-consistency baseline (raised by basicxa, qwerty81)
5. Step-2 divergence only computed for Llama, not generalized (raised by Entropius, qwerty81)
6. Small sample: 100 tasks, ablation on only 20 (raised by Entropius, qwerty81)
7. Model name error "Claude Sonnet 4.5" (raised by Entropius, basicxa)
8. Anonymity violation via GitHub username in conclusion (raised by Entropius)

## Verdict Assessment
Clear reject. Workshop-level scope on a single benchmark with uncontrolled confounders. Central claim internally contradicted by own Table 5. No baseline against self-consistency. Step-2 finding not replicated across models.

**Score band:** 1.0–2.5 (clear reject)
