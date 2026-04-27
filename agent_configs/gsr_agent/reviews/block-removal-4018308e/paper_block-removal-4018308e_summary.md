# Paper Summary: Block Removal for LLMs through Constrained Binary Optimization

**Paper ID:** 4018308e-0bdc-4dd8-9421-b517562caff4
**arXiv:** 2602.00161
**Institution:** Multiverse Computing

## Core Contribution

Formulates transformer block pruning as a constrained binary optimization (CBO) problem via second-order Taylor expansion of the loss, mapped to an Ising model with fixed magnetization. Key insight: model block interactions explicitly rather than using greedy/local selection criteria.

## Method (§3)

- Introduces coupling variable αᵢ per block (multiplicative modulation of ATTN and FFN outputs)
- Taylor-expands loss L(α) around α⁰ = 1 (original model)
- **Neglects first-order term** (∇L(α⁰) ≈ 0 for well-trained model) — paper explicitly notes Rosenberg et al. (2025) found including this term "reported to improve results"
- Approximates Hessian as H⁰ ≈ (1/m) AᵀA (per-sample gradients outer product)
- Results in: min xᵀH⁰x subject to Σxᵢ = M, xᵢ ∈ {0,1}
- Solvable by brute force for small N,M; convertable to QUBO for larger cases

## Experiments (§4)

- Models: Llama-3.1-8B-Instruct (32 blocks, remove 8/16), Qwen3-14B (40 blocks, remove 8/12)
- Calibration: 2048 samples from OpenHermes-2.5
- Retraining: 1 epoch (Llama) / 0.5 epoch (Qwen) knowledge distillation on OpenHermes-2.5
- Baselines: Block Influence (BI, iterative), Sliding Window Merging (SWM), Norm Ratio
- Section 4.3 confirms same calibration data and retraining procedure for all baselines

## Key Findings

- CBO outperforms baselines especially on MMLU at high compression rates (>40%)
- 17th excited state (non-ground-state) outperforms ground state on Qwen3-14B 16-block removal
- Excited states provide useful diversity of removal configurations
- Generalizes to MoE architectures (Nemotron-3-Nano-30B)

## Critical Issues Identified

### Issue 1: Duplicate block indices in BI baseline (Table 2)
- Qwen3-14B, 8 blocks removed, BI: [35,34,33,32,22,**2,2**,28] — block 2 listed twice
- Qwen3-14B, 12 blocks removed, BI: [35,34,33,32,22,**2,2**,28,**17,17**,8,24] — 2 and 17 each listed twice
- If actual (not just reporting error): BI removes only 7/10 unique blocks vs CBO's 8/12
- This creates comparison asymmetry: fewer-block model retains more accuracy naturally

### Issue 2: First-order term ablation missing
- Paper cites Rosenberg et al. (2025) finding first-order term improves results
- No ablation shows the performance cost of neglecting this term
