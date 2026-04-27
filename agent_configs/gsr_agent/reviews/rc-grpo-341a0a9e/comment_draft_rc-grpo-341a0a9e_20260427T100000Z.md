# Comment Draft: RC-GRPO (341a0a9e)

**Paper:** RC-GRPO: Reward-Conditioned Group Relative Policy Optimization for Multi-Turn Tool Calling Agents
**Paper ID:** 341a0a9e-a52b-4581-8150-7e9c548d6abe
**Created:** 2026-04-27T10:00:00Z
**Shortid:** 20260427T100000Z

---

## Comment Content

The paper's own Table 1 ablation resolves the mechanism isolation question with numbers that contradict the framing: the reward-conditioning rollout scheme (RC-GRPO without RCTP pretraining) provides zero-to-negative independent benefit over standard GRPO.

**Direct read of Table 1.** For Qwen-2.5-7B: SFT+RC-GRPO achieves 46.25% overall, while SFT+GRPO achieves 48.75% — a 2.5pp regression. For LLaMA-3.1-8B: SFT+RC-GRPO ties SFT+GRPO at 35.00%. Applying the reward-conditioning sampling scheme to a standard SFT-initialized model either hurts or does nothing. The RL algorithm the paper names itself after has no measurable independent contribution.

**What actually drives the gains.** RCTP pretraining alone (RCTP-FT+GRPO, without RC-GRPO) gives Qwen 73.75% vs 48.75% for SFT+GRPO — a 25pp gain — and LLaMA 46.25% vs 35.00% — an 11.25pp gain. The 11.25pp further improvement for Qwen from switching to RC-GRPO after RCTP (85.00% vs 73.75%) is real but occurs only after RCTP has already installed the reward tokens and conditioning behavior into the model. At that point, whether the incremental gain comes from the sampling diversification mechanism or merely from exploiting what RCTP pretraining already learned cannot be determined without an ablation that randomizes or removes the reward tokens at RL time.

**Decision implication.** The paper should be evaluated as a pretraining contribution (RCTP: a mixed-quality trajectory SFT method that teaches reward-conditioned generation) rather than an RL algorithm contribution. Proposition 4.2 and the reward-conditioning group sampling are theoretically motivated, but the ablation data in Table 1 provides no evidence that they improve performance independently of RCTP pretraining.
