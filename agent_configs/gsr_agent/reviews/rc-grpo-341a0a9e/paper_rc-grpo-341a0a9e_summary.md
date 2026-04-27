# Paper Summary: RC-GRPO

**Paper ID:** 341a0a9e-a52b-4581-8150-7e9c548d6abe
**Title:** RC-GRPO: Reward-Conditioned Group Relative Policy Optimization for Multi-Turn Tool Calling Agents
**ArXiv:** 2602.03025
**Status:** in_review
**Domains:** NLP, Reinforcement-Learning

## Core Claim

Multi-turn tool-calling RL stalls when within-group reward variance is low (all-0 or all-1 reward groups). RC-GRPO addresses this by (1) pretraining a Reward-Conditioned Trajectory Policy (RCTP) on mixed-quality trajectories with special reward tokens (`<|high_reward|>`, `<|low_reward|>`), then (2) sampling diverse reward tokens during GRPO groups to manufacture within-group variance.

## Method

Two stages:
1. **RCTP pretraining**: Fine-tune on mixed-quality (success + failure) trajectories with reward-goal tokens injected into the prompt. Teaches the model to generate high- or low-quality outputs on demand.
2. **RC-GRPO**: During RL, sample diverse reward tokens within each GRPO group. Rollouts are conditioned on the sampled token. Inference uses `<|high_reward|>` token.

## Evaluation

- Benchmark: Berkeley Function Calling Leaderboard v4 (BFCLv4) multi-turn
- Test set: 80 samples across 4 categories (Base=18, MissFunc=17, MissParam=22, LongContext=23)
- Models: Qwen-2.5-7B-Instruct and LLaMA-3.1-8B

## Key Ablation Results (Table 1)

| Config | Qwen Overall | LLaMA Overall |
|--------|-------------|---------------|
| Base   | 11.25%      | 0.00%         |
| SFT+GRPO | 48.75%   | 35.00%        |
| SFT+RC-GRPO | 46.25% | 35.00%      |
| RCTP-FT+GRPO | 73.75% | 46.25%    |
| RCTP-FT+RC-GRPO (Ours) | 85.00% | 48.75% |

## Critical Observation

SFT+RC-GRPO underperforms SFT+GRPO for Qwen (−2.5pp) and ties for LLaMA. The core RL sampling mechanism provides no independent empirical benefit. All substantive gains trace to RCTP pretraining (+25pp for Qwen alone).

## Known Discussion Issues

- Table arithmetic error in LLaMA headline row (MissParam/LongContext appear swapped)
- Single benchmark, 80 test samples
- Inference-time reward-token policy unspecified
- Mechanism isolation (RCTP vs RC-GRPO contributions conflated)
