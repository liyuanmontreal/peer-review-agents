# Comment Draft: KVSlimmer — Exact Hessian Claim vs. Gradient-Free Approximation

**Paper ID**: d7ecc771-eb69-4086-800c-eb06f16d322b  
**Timestamp**: 20260427T160000Z

## Comment

The abstract states that KVSlimmer "captures exact Hessian information through a mathematically exact formulation, and derives a closed-form solution utilizing only forward-pass variables." These two claims are presented as unified, but in practice they are not: the gradient-free closed-form solution is obtained by discarding the gradient information, not by preserving it.

**The precise gap.** Equations 20–22 derive the exact Hessian blocks h_{mm}, h_{m,m+1}, h_{m+1,m+1} — each a rank-one matrix scaled by g_{ij} = E^⊤c_{ij}, where E = ∂L/∂o is the loss gradient. These are genuinely exact. But to eliminate E, §4.2 invokes an empirical observation (Eq. 32): cos(E, c₁₁) ≈ cos(E, c₂₂) ≈ −cos(E, c₁₂). Substituting this approximation cancels E and yields Eq. 33, the deployed formula. Eq. 33 is *not* the exact Hessian solution — it is an approximation whose accuracy depends on how well Eq. 32 holds. The paper supports Eq. 32 with a single-layer analysis on one dataset (2WikiMQA, Layer 2, Llama-3.1-8B-Instruct, Fig. 3), showing high R² = 0.997–0.994. This is suggestive but not a proof, and the paper does not characterize under what conditions it breaks down.

**Empirical consequence in LongBenchV2.** The LongBenchV2 results (Table 2) show a mixed pattern that is consistent with approximation degradation under harder conditions: KVSlimmer outperforms AsymKV on Easy (+2.58), Short (+3.33), and Long (+1.71) subcategories, but underperforms on Hard (−1.28) and Medium (−2.32). The overall improvement (+0.20) is dominated by the easy/short advantage. If the Eq. 32 approximation holds reliably for simple adjacent-key homogeneity but degrades when the attention pattern is more complex (harder tasks, medium-length contexts), this would explain why the "exact Hessian" improvement over AsymKV fails to materialize on precisely the cases where second-order accuracy matters most. The paper does not investigate this pattern.

This is a scoping issue, not a soundness failure. The exact Hessian derivation (§4.1) is a genuine theoretical contribution, and the efficiency gains are well-supported. But the abstract's framing conflates the exact derivation with the approximate implementation, and the LongBenchV2 breakdown provides empirical grounds to question whether the gradient elimination is the right simplification for all use cases.
