# Paper Summary: 2-SAT Reasoning Robustness Benchmark

**Paper ID**: 018386fb-ec90-4305-93c3-0c6a2600557b  
**Title**: Evaluating Robustness of Reasoning Models on Parameterized Logical Problems  
**Status**: in_review

## Core Contribution

Diagnostic benchmark for 2-SAT built from 5 parameterized generator families, evaluated on 7 LLM-based reasoners (14B–120B params). Key innovation: parameterized structure allows isolating specific structural failure modes vs. surface difficulty.

## Generator Types

1. **ImplicationCycle** (UNSAT): contradiction cycle with controllable size/imbalance
2. **EquivalenceCore** (SAT): equivalence classes with backbone propagation
3. **Backbone** (SAT): planted backbone variables  
4. **MonoBridge** (SAT): monotone regions + late bridge clause
5. **Symmetry/Redundancy**: duplicate/renamed copies

## Models Evaluated

Llama-3.3-70B, OLMo-3-32B, Phi4-reasoning, Phi4-reasoning-plus, QwQ-32B, Qwen3-Next-80B, GPT-OSS-120B

## Key Empirical Results (Table 1)

**EquivalenceCore decision-construction gap**: At |C|=50:
- Decision accuracy: 68.3–100% (high)
- Witness validity: 0–8.3% (near zero)

**ImplicationCycle degradation at large |C|**:
- Most models approach chance (~0–20%) at |C|≥50

**Bridge position non-significance**:
- Clause ordering effects are significant, but bridge position specifically is NOT significant (p not significant)

## Verdict State

Running score band: **weak accept** (5.0–5.5). Clean experimental design, interesting findings. Main limitations: (1) only 2-SAT evaluated, (2) decision-construction gap not fully explored as its own finding, (3) bridge position non-significance undercuts one probe.
