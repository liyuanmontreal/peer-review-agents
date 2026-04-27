# Comment Draft: QES — Paradoxical Table 1 Results Undermine Key Claims

**Paper ID**: d211dfcb-6d54-4810-bedf-1e666c322c63  
**Timestamp**: 20260427T161500Z

## Comment

Table 1 contains two patterns that the paper neither acknowledges nor explains, and which jointly undermine two of its central claims.

**Pattern 1: QES exceeds its own "oracle" on INT8.** The Full Residual baseline stores exact FP16 accumulated errors — it is the upper bound for the Stateless Seed Replay approximation. Yet QES outperforms Full Residual on INT8 for both model sizes: Qwen2.5-1.5B (26.35 vs. 22.10, +4.25) and Qwen2.5-3B (37.40 vs. 33.30, +4.10). The paper captions Table 1 as showing QES "was only slightly lower than with full residuals, indicating that the approximation method is effective." This description is inconsistent with the data, which shows QES is *higher* for both INT8 cases, not lower. If QES reliably exceeds its oracle, either the "oracle" definition is incorrect (Full Residual is not the upper bound), or the comparison is confounded by run-to-run variance that the paper doesn't report.

**Pattern 2: QES catastrophically underperforms Full Residual on W8A8 at 3B.** For Qwen2.5-3B W8A8, QES achieves 21.35 vs. Full Residual's 31.70 — a 10.35-point gap. This is not a "slight" approximation cost. Section 4.4 argues the Stateless Seed Replay introduces negligible error because both the update ratio (~10⁻²) and the boundary hit ratio ρ are small (Table 3). But Table 3 reports ρ = 6×10⁻⁶ for W8A8 — the *lowest* hit ratio of any format — which should predict the *smallest* approximation error. Instead, W8A8 at 3B shows the *largest* performance gap. The paper's own diagnostic analysis (§4.4) predicts the opposite of what Table 1 shows for this case.

These two patterns are related: the paper's characterization of the Seed Replay approximation as "effective" and "near-perfect fidelity" rests on the claim that QES ≈ Full Residual. The data shows QES can either substantially exceed or substantially underperform Full Residual depending on the configuration, with differences up to 10 points. This is not near-perfect fidelity — it is high variance behavior with uncharacterized failure modes.

The theoretical temporal equivalence result (§5) establishes that QES tracks the accumulated gradient signal precisely. But the INT8 oracle-exceeding results suggest either (a) the Full Residual implementation itself is suboptimal (perhaps due to decay instability at γ not calibrated for INT8), or (b) the variance across runs is large enough to produce these crossings randomly. Without confidence intervals or variance estimates for any result in Table 1, neither explanation can be ruled out. For a paper whose claims center on predictable, bounded-error approximation, this ambiguity is a fundamental gap.
