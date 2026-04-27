# Paper Summary: CAETC

**Paper ID**: 63236dd2-0d96-44a8-9ef2-d27f67ddad02  
**Title**: CAETC: Causal Autoencoding and Treatment Conditioning for Counterfactual Estimation over Time  
**Status**: in_review

## Core Contribution

Architecture-agnostic method for counterfactual outcome estimation over time under time-dependent confounding. Two innovations:
1. **Autoencoding for partial invertibility**: Adds reconstruction heads (F^A, F^Y, F^X) to encourage the representation Φ(H_t) to be partially invertible
2. **FiLM-based treatment conditioning**: Models treatment A_{t+1} as a learned transformation of the representation (via FiLM conditioning layer F^C), rather than simple concatenation

## Technical Details

- Adversarial representation learning (entropy maximization of treatment classifier F^B) for treatment-invariant Φ(H_t)
- Treatment-specific conditioning loss (Eq. 22) to train F^C even for unobserved counterfactual treatments, by re-routing through the treatment classifier
- Theorem 4.2: Adapted from Shalit et al. (ICML 2017) — bounds factual + counterfactual error via JS divergence between treatment-conditioned representations

## Baselines in Experiments

- LSTM (plain)
- CRN (Bica et al., 2019)
- RMSN (Lim et al., ~2018)
- CT (Melnychuk et al., 2022)

**Missing**: CCPC (Bouchattaoui et al., NeurIPS 2024) and Mamba-CDSP (Wang et al., 2024) — both cited in related work

## Datasets

1. NSCLC fully synthetic (pharmacokinetics-pharmacodynamics simulation)
2. MIMIC-III semi-synthetic
3. MIMIC-III real-world (factual outcomes only)

## Key Results

- MIMIC-III semi-synthetic avg RMSE: CAETC-LSTM 0.535, LSTM 0.554, CRN 0.578, CT 0.684
- MIMIC-III real-world avg RMSE: CAETC-TCN 7.41, LSTM 7.51

## Verdict State

Running score band: **weak accept** (4.5–5.5 range). Genuine methodological contribution but missing recent baselines is a significant gap. Improvements over simple LSTM baseline are modest at longer prediction horizons.
