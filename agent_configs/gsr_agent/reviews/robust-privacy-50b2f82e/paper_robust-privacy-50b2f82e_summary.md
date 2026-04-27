# Paper Summary: Robust Privacy

**Paper ID:** 50b2f82e-0c3b-4ff7-b966-6339a234f65c  
**Title:** Robust Privacy: Inference-Time Privacy through Certified Robustness  
**arXiv:** 2601.17360  
**Domains:** Trustworthy-ML, Theory

## Core Contribution

Introduces Robust Privacy (RP): if model f is prediction-invariant in an L_p ball of radius R around input x, then x has R-RP. Implements via randomized smoothing (RS). Develops Attribute Privacy Enhancement (APE) for attribute-level privacy analysis. Demonstrates empirical MIA mitigation on FaceNet64/CelebA.

## Key Results

- MIA ASR: 73% (base) → 44% (σ=0.03, no accuracy loss) → 4% (σ=0.1, accuracy 100%→59.4%)
- APE: Positive recommendation set expands by ~R BMI units in controlled task
- Certified radii at σ∈{1,2,3}: R ≈ {0.56, 0.63, 0.65}

## Critical Analysis

### Definition-Theorem Gap
Output invariance within [z-R, z+R] does NOT prevent inference that x₁ ∈ [z-R, z+R]. No formal theorem bounds adversarial advantage or mutual information I(x₁; f(x)). RP is definitional, not a formal privacy guarantee.

### Loose Conceptual Analogy
Certified robustness addresses perturbation attack on model; privacy addresses information leakage to inference adversary. Output invariance in one regime does not formally imply privacy in the other. Counterexample: invariant prediction on [z-R, z+R] still leaks that x₁ is in that interval.

### Narrow MIA Evaluation
Only Kahla et al. (2022) label-only attack evaluated. No adaptive adversary. No comparison to gradient-based or GAN-based MIA. Label-only is weakest MIA class.

### Severe Utility Trade-off
4% ASR achieved at 59.4% accuracy — equivalent to ~40% abstention rate. No comparison to simpler baselines (DP-SGD, output noise injection, random abstention).

### Experimental Setup Artifacts
- Controlled recommendation uses ℓ1-penalized model to artificially concentrate dependence on BMI — not representative of real systems
- APE holds all non-sensitive features fixed (1D subspace analysis)
- No code released, ASR threshold undefined
