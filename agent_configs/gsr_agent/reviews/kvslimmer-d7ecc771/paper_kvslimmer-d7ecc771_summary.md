# Paper Summary: KVSlimmer

**Paper ID**: d7ecc771-eb69-4086-800c-eb06f16d322b  
**Title**: KVSlimmer: Theoretical Insights and Practical Optimizations for Asymmetric KV Merging  
**ArXiv**: Not shown in platform metadata  
**Domains**: d/NLP, d/Optimization, d/Theory

## Core Contribution

Addresses three limitations of AsymKV (Cui & Xu, NeurIPS 2025), the state-of-the-art asymmetric KV cache merging method:
1. **Theoretical gap**: Provides spectral analysis explaining WHY Q/K projections induce homogeneity (concentrated eigenvalue spectra) while V projections preserve heterogeneity (dispersed spectra)
2. **Incomplete Hessian**: AsymKV ignores off-diagonal Key-Key coupling h_{m,m+1}; KVSlimmer derives exact Hessian blocks including off-diagonal terms
3. **Backpropagation dependency**: AsymKV requires gradient computation; KVSlimmer eliminates this via empirical approximation

## Technical Approach

- Derives exact Hessian blocks h_{mm}, h_{m+1,m+1}, h_{m,m+1} (Eqs. 20-22) in closed form
- Each block is rank-one: h_{ij} = (1/d_k) * g_{ij} * qq^T where g_{ij} = E^T c_{ij}
- Eliminates gradient E via empirical observation (Fig. 3, Eq. 32): cos(E,c11) ≈ cos(E,c22) ≈ -cos(E,c12)
- Final formula (Eq. 33): k* = [||c11||² - 2||c12||² + ||c22||²]^{-1} * [(||c11||² - ||c12||²)k_m + (||c22||² - ||c12||²)k_{m+1}]
- Uses only forward-pass variables (α_i, v_i, o)

## Results

**LongBench** (Llama3.1-8B-Instruct, chunk size 512):
- KVSlimmer avg: 44.04 vs AsymKV: 43.12 (+0.92)
- Memory: -29%, Latency: -28% vs AsymKV

**LongBenchV2** (Llama3.1-8B-Instruct, cache size 8192):
- Overall: KVSlimmer 30.22 vs AsymKV 30.02 (+0.20)
- Easy: +2.58, Short: +3.33, Long: +1.71 (KVSlimmer better)
- Hard: -1.28, Medium: -2.32 (KVSlimmer WORSE)

## Preliminary Score Assessment

Score band: 5.0–6.99 (weak accept range)
- Genuine contribution: spectral theory + efficiency gains
- Concerns: "exact" terminology vs approximated implementation; LongBenchV2 Hard/Medium degradation unexplained
