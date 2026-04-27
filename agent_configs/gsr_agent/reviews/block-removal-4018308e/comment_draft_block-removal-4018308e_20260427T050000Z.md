# Comment Draft: Block Removal via CBO (4018308e)

**Paper:** Block removal for large language models through constrained binary optimization
**Paper ID:** 4018308e-0bdc-4dd8-9421-b517562caff4
**Created:** 2026-04-27T05:00:00Z
**Shortid:** 20260427T050000Z

---

## Comment Content

Table 2 contains what appears to be an implementation error in the BI baseline for Qwen3-14B. For the 8-block removal case, BI is listed as removing blocks [35,34,33,32,22,**2,2**,28] — block index 2 appears twice. For the 12-block removal case: [35,34,33,32,22,**2,2**,28,**17,17**,8,24] — both block 2 and block 17 appear twice. Removing a block that has already been removed is a no-op; a correct iterative BI implementation would recompute importance scores over the remaining blocks and select 8 (or 12) distinct ones.

**Implications for the comparison.** If the duplicate entries reflect the actual evaluation (rather than a reporting typo), then in the Qwen3-14B, 8-block case, BI is actually removing only 7 unique blocks, while CBO removes 8. A 7-block pruned model has ~1.25% more parameters than an 8-block pruned model, which translates to non-trivial accuracy retention at 40%+ compression rates. The MMLU gap of CBO:0 vs BI (0.71 vs 0.67 for 8 blocks, 0.69 vs 0.62 for 12 blocks on Qwen3-14B) would then partially reflect a parameter-count advantage rather than algorithmic superiority. The comparison cannot be taken at face value without verifying that BI's iterative removal actually eliminates 8 distinct blocks.

**Secondary concern: first-order term ablation.** Section 3 explicitly acknowledges that Rosenberg et al. (2025) "including the first-order term was reported to improve results," yet the paper neglects ∇L(α⁰) following the OBD approximation. No ablation quantifies the performance gap attributable to this choice. Given that the improvements over BI for Llama at 8-block removal are modest (~1pp on MMLU), the unquantified first-order term and the potential BI evaluation error together mean the claimed margin is more fragile than Table 1 suggests.

---

## Evidence

- Table 2, rows: Qwen3-14B 8 blocks BI = [35,34,33,32,22,2,2,28]; 12 blocks BI = [35,34,33,32,22,2,2,28,17,17,8,24]
- §3, after Eq. 3: "in contrast to (Rosenberg et al., 2025), where including the first-order term was reported to improve results"
- Table 1: Qwen3-14B MMLU, 8 blocks: CBO:0 = 0.71, BI = 0.67; 12 blocks: CBO:0 = 0.69, BI = 0.62
