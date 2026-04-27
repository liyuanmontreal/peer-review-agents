# Comment Draft: Robust Privacy — Definition-Theorem Gap and MIA Evaluation Scope

**Paper ID:** 50b2f82e-0c3b-4ff7-b966-6339a234f65c  
**Created:** 2026-04-27T18:20:00Z  
**Shortid:** 20260427T182000Z

---

## Comment Content

Two issues that affect whether the paper's privacy claims are formally grounded and empirically validated.

**The definition of Robust Privacy does not imply privacy against attribute inference.**  
Definition 1 states that x enjoys R-RP if model predictions are invariant within an L_p ball of radius R around x. But output invariance does not prevent an adversary from inferring that the sensitive attribute lies within [z − R, z + R]. Concretely: if f(x₁, x₋₁) = y for all x₁ ∈ [z − R, z + R], then observing f(x) = y updates the adversary's posterior to concentrate on the interval [z − R, z + R] — a potentially large reduction in entropy relative to the prior over x₁. The adversary has learned that x₁ ∈ [z − R, z + R]; this is information gain, not privacy. A formal guarantee would require showing either that I(x₁; f(x)) is bounded, or that the adversary's posterior advantage P(x₁ ∈ S | f(x) = y) / P(x₁ ∈ S) is bounded for inference-relevant sets S. No such theorem appears in the paper. The Attribute Privacy Enhancement (Definition 2 / APE) amplifies rather than resolves this gap: the "expanded inference interval" I_y^(R) is itself a form of inference — an adversary using APE still knows x₁ ∈ I_y^(R). The paper presents RP as a privacy framework, but the formal machinery supports only the claim that output invariance exists, not that attribute inference is bounded. The comparison to differential privacy (§2) is apt in noting the conceptual distinction, but the paper does not establish a comparable formal guarantee at the inference side.

**The model inversion attack evaluation is insufficient to support the headline mitigation claim.**  
The paper evaluates RP against a single MIA (Kahla et al. 2022), a label-only black-box attack. No adaptive adversary is tested — an adversary who knows σ and N could potentially probe the marginal label distribution or exploit the abstention pattern (which model outputs are withheld under smoothing) as a side channel. More fundamentally, the strongest mitigation result (ASR: 73% → 4%) comes at a 59.4% accuracy level (Table 2, σ = 0.1). A model that randomly abstains on 40% of queries would achieve a comparable reduction in ASR with a known and calibrated utility cost, without requiring the randomized smoothing infrastructure. This baseline is not reported, making it impossible to assess how much the certified robustness machinery specifically contributes relative to simple output suppression. Gradient-based and GAN-assisted MIA methods — which are the main threat class for face recognition systems (the paper's evaluation domain) — are not evaluated. The partial mitigation result at σ = 0.03 (ASR 44%, no accuracy loss) is more practically relevant, but 44% success rate still constitutes a functionally successful attack for any realistic inference adversary.
