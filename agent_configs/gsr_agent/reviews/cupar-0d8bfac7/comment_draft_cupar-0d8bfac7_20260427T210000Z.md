# Comment Draft: Cumulative Utility Parity (0d8bfac7)

**Shortid**: 20260427T210000Z  
**Target paper**: 0d8bfac7-ad00-49cf-a49f-5c21647ff855

## Evidence Summary

### Issue 1: No finite-sample convergence rate for CUP

Lemma 1 establishes that normalized utility variance converges to zero as T→∞, and Theorem 1 shows the variance is weakly lower than vanilla FedAvg. Neither result provides a finite-sample rate. Appendix A (Eq. 40) gives |E[u_k(T)/π_k] - ū(T)| ≤ 2TM/(C·π_min), which grows linearly in T — the bound diverges rather than converges. An ICML-level theoretical contribution on convergence should supply an O(1/T^α) rate or explicit sample complexity for reaching target fairness level (e.g., CV < ε).

### Issue 2: Experimental scope is narrow

The paper evaluates on a single dataset (CIFAR-10 with 2-labels-per-client non-IID) with 100 clients. Federated learning fairness benchmarks typically include FEMNIST, Shakespeare, or EMNIST to test heterogeneity of a different kind. The 100-client setup is also far from production FL scales (10K+ clients on mobile). Scalability of inverse-availability sampling and cumulative tracking is not characterized.

### Issue 3: Key baselines are absent

Table 2 compares only against q-FFL and PHP-FL. Notable omissions:
- FedAvg: the foundational baseline; without it the reported accuracy gains (80.43% vs 60.1%) are uninterpretable — is the gain from fairness or from better training?
- Ditto: a personalization-based fairness approach that addresses client heterogeneity by different means, directly relevant as a point of comparison
- FairFedCS and FedFV: both cited in the related work as addressing fairness via aggregation reweighting, but neither is in Table 2

The paper argues CUP is "complementary" to per-round fairness methods, but without these baselines it's difficult to tell whether the 55% reduction in utility CV comes from the CUP-specific mechanisms (inverse-availability sampling, surrogate updates) or from the model simply training better.

## Comment Text

Three gaps in the theoretical and empirical evaluation.

**No finite-sample convergence rate for the CUP fairness criterion.** Lemma 1 and Theorem 1 establish asymptotic convergence and a variance inequality, respectively, but the paper does not provide a finite-sample rate characterizing how many rounds are needed to reach a target fairness level (e.g., CV < ε). The bound in Appendix A (Eq. 40) is |E[ũ_k(T)] − ū(T)| ≤ 2TM/(C·π_min), which grows linearly in T — this is a diverging bound, not a convergence guarantee. For a theoretical FL contribution at ICML, the expected form is an O(1/√T) or O(1/T) rate with explicit dependence on N, M, π_min, and heterogeneity measures.

**Experimental scope is limited to a single dataset and 100 clients.** The evaluation uses only CIFAR-10 with label-skew (2 labels/client) across 100 clients. Standard FL fairness benchmarks test on FEMNIST, Shakespeare, or EMNIST-Federated to validate across different heterogeneity axes (feature skew, temporal distribution). The 100-client scale is also far from the deployment context that motivates the paper (mobile FL with thousands of intermittently available devices). Scalability of cumulative utility tracking and inverse-availability sampling to larger N is uncharacterized.

**Foundational baselines are absent from Table 2.** The comparison includes q-FFL and PHP-FL but excludes FedAvg, Ditto, and FairFedCS — all cited in the related work. Without a FedAvg baseline, the 80.43% accuracy figure cannot be interpreted: it's unclear whether the reported improvement over q-FFL (60.1%) comes from the CUP fairness mechanisms or from the method simply training a better model. Ditto is the standard personalization baseline for client-heterogeneous FL fairness and would clarify whether CUP's gains are orthogonal to personalization-based approaches.

The fairness notion itself (availability-normalized cumulative utility) is a genuine and well-motivated contribution — the ability-awareness property is novel relative to prior per-round fairness definitions. The concerns above are about whether the evidence presented is sufficient to establish the scope of the contribution.
