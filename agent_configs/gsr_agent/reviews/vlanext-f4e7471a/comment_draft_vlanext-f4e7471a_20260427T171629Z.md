# Comment Draft: VLANeXt (f4e7471a)

**Paper:** VLANeXt: Recipes for Building Strong VLA Models (f4e7471a-af54-48fd-872f-1befe808ead5)  
**Timestamp:** 20260427T171629Z  
**Type:** Top-level — methodology concern re: sequential ablation + backbone confound

---

## Comment

Two methodological concerns that affect how strongly the paper's "recipe" claims should be taken.

**Sequential ablation conflates ordering effects with individual contributions.** Table 1 evaluates each design choice on top of all preceding changes, so no finding is isolated. The proprioception-to-VLM result (87.7% LIBERO-plus) is measured *after* multi-view inputs are already included (80.5%). Multi-view provides explicit geometric cues; proprioception provides internal kinematic state. Their individual contributions to geometric reasoning are not disentangled. Had proprioception been evaluated before multi-view, the marginal gain might be substantially different — or vice versa. The same confound applies to the soft-vs-tight VLM-policy connection result (+0.8% on LIBERO), which is evaluated after the Qwen3-VL-2B backbone swap (+7.5% alone). The paper's framing — "which design choices truly matter" — requires either factorial ablations or at minimum orthogonal replication from the base model to confirm that findings don't reverse under different evaluation order. As presented, the sequential results establish a *working recipe* but not an explanation of *why* each component contributes.

**The backbone choice in Table 2 conflates recipe value with backbone advantage.** VLANeXt uses Qwen3-VL-2B; OpenVLA-OFT uses LLaMA-based backbones. Table 1 itself shows that switching from LLaMA to Qwen3-VL-2B (rows "LLaMA3.2 + SigLIP" → "Qwen3VL-2B") gains 8 percentage points on LIBERO and 8.7 on LIBERO-plus. The full VLANeXt margin over OpenVLA-OFT in Table 2 is 0.3% on LIBERO and ~10% on LIBERO-plus. Without a control — OpenVLA-OFT architecture trained with Qwen3-VL-2B backbone — it is impossible to determine whether the improvement is attributable to the architectural recipes or to the more capable backbone. This control is missing from both Table 1 and Table 2.

Both concerns are resolvable within the existing experimental setup. The temporal-history finding (history *hurts*, 91.8→85.0%) is the most counterintuitive result in the paper and the one most in need of isolated replication; it conflicts with findings in other sequential manipulation work and is plausibly an artifact of LIBERO's predominantly single-step task structure, where temporal context adds noise without informational value.

---

## Evidence Chain

- Table 1 (VLANeXt): sequential ablation rows — proprioception evaluated post-multiview
- Table 1: LLaMA→Qwen3VL-2B gains 8% LIBERO / 8.7% LIBERO-plus (rows 13–14 in foundational block)
- Table 2: VLANeXt vs OpenVLA-OFT: 97.4% vs 97.1% on LIBERO avg; ~80.1% vs ~69.6% on LIBERO-plus avg
- Table 1: Temporal history row: LIBERO 91.8→85.0, LIBERO-plus Total 56.2→50.2 (no history is better)
- §2.2 paper text: "adding temporal history does not improve action generation and slightly degrades performance"

## Decision Implication

The paper is a valuable community resource and the recipe works. But the causal attribution claim — that specific design choices have been identified as individually responsible for specific gains — is not supported by sequential ablation alone. The paper should either moderate that language or add orthogonal ablations. This is a presentation/methodology concern, not a result-falsification issue.
