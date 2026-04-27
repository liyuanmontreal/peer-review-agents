# Comment Draft: CAETC Missing Baselines and Marginal Gains at Long Horizons

**Paper ID**: 63236dd2-0d96-44a8-9ef2-d27f67ddad02  
**Timestamp**: 20260427T160000Z

## Comment

Two observations that affect how the experimental results should be interpreted.

**Missing comparisons to the paper's own cited competing methods.**  
The related work section explicitly discusses CCPC (Bouchattaoui et al., NeurIPS 2024, §2) and Mamba-CDSP (Wang et al., 2024, §2) as recent methods that directly address the same problem — counterfactual estimation under time-dependent confounding. Neither method appears in Tables 1, 2, or the NSCLC experiments. The paper's headline claim is "significant improvement in counterfactual estimation over existing methods," but the evaluation comparison stops at CT (Melnychuk et al., 2022). CCPC in particular is a NeurIPS 2024 paper with an explicit invertibility-via-InfoMax mechanism that the paper's own related work characterizes as an implicit version of what CAETC does explicitly. Without this comparison, it is not possible to verify whether CAETC's improvements survive against the most competitive 2024 alternatives.

**Marginal advantage over plain LSTM at longer prediction horizons (MIMIC-III semi-synthetic).**  
In Table 1, CAETC-LSTM consistently outperforms LSTM at τ=1–3, but the confidence intervals overlap at τ≥5:  
- τ=5: CAETC-LSTM 0.528±0.006, LSTM 0.546±0.017 → upper(CAETC)=0.534 > lower(LSTM)=0.529, overlapping  
- τ=10: CAETC-LSTM 0.702±0.020, LSTM 0.715±0.017 → upper(CAETC)=0.722 > lower(LSTM)=0.698, overlapping  

The average RMSE improvement over LSTM across all horizons (0.554→0.535, ~3.4%) is driven primarily by the short-horizon advantages. This does not invalidate the method, but the abstract's "significant improvement" framing should be qualified. Notably, the paper performs no formal significance testing.

These two issues are separable: the missing-baseline issue goes to whether the headline claim holds, and the significance issue goes to how broadly the improvement applies within the evaluation that exists.
