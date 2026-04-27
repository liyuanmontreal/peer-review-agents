# Paper Summary: VLANeXt — Recipes for Building Strong VLA Models

**Paper ID:** f4e7471a-af54-48fd-872f-1befe808ead5  
**Status:** in_review  
**Created:** 2026-04-27T16:00:01

## Overview

Systematic ablation study of VLA design choices across three dimensions:
1. **Foundational components** — policy module design, action chunking, learning objective, VLM backbone, VLM-policy connection
2. **Perception essentials** — temporal history, camera views, proprioception conditioning
3. **Action modelling** — world modelling, time-series/frequency-domain loss

The final VLANeXt (2.5B) outperforms π₀ (4B) and OpenVLA-OFT (7B) on LIBERO and LIBERO-plus benchmarks.

## Key Findings from Table 1 (Sequential Ablation on LIBERO / LIBERO-plus)

| Design Choice | LIBERO | LIBERO-plus Total |
|---|---|---|
| RT-2-like baseline | 19.8 | <5.0 |
| + Separate policy head | 30.2 | 16.6 |
| + Large policy module (MetaQuery, 12L) | 64.4 | 34.0 |
| + Action chunk 8 | 74.6 | 43.4 |
| + Flow matching loss | 80.0 | 45.0 |
| + Qwen3-VL-2B backbone | 90.0 | 53.7 |
| + Soft VLM-policy connection | 91.8 | 56.2 |
| - Temporal history removed | 91.8 | 56.2 (improves −remove history) |
| + Multi-view (3rd person + wrist) | 97.6 | 80.5 |
| + Proprioception to VLM | 98.0 | 87.7 |
| + Frequency-domain loss | 99.0 | 92.8 |

## Surprising Finding
Temporal observation history **hurts** performance on LIBERO (91.8→85.0%) and LIBERO-plus (56.2→50.2%). The paper attributes this to "redundant temporal inputs introducing noise." This contradicts findings in some other VLA work.

## Real-World Evaluation
- 4 tasks, 20 trials each (single-arm: Franka, bimanual: Aloha)
- VLANeXt: 14/20, 11/20, 11/20, 15/20
- π₀: 10/20, 8/20, 10/20, 13/20
- OpenVLA-OFT: 7/20, 7/20, 5/20, 9/20

## Critical Methodological Concern
**Sequential ablation methodology**: All design choices in Table 1 are evaluated sequentially (each on top of all previous). This means:
- Later results include compound interactions from all prior choices
- Individual contributions cannot be disentangled (e.g., proprioception-to-VLM evaluated after multi-view added)
- Design choice ordering may affect conclusions

## Backbone Confound in Table 2
Final VLANeXt uses Qwen3-VL-2B; baselines use different backbones (π₀: PaliGemma-3B, OpenVLA-OFT: LLaMA). Table 2 doesn't show Qwen3-VL backbone + competitor architecture, making it hard to separate recipe value from backbone advantage.

## Score Assessment (Running)
- **Strengths:** Clear, actionable findings; strong benchmark performance; real-world validation; codebase release; valuable community resource
- **Weaknesses:** Sequential ablation methodology limits causal interpretation of individual findings; LIBERO-only derivation; backbone confound in main comparison
- **Running band:** Weak accept (5.0–6.99) — valuable community resource but causal claims about individual design choices are overstated
