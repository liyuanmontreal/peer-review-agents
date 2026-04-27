# Paper Summary: Cumulative Utility Parity for Fair Federated Learning

**Paper ID**: 0d8bfac7-ad00-49cf-a49f-5c21647ff855  
**Title**: Cumulative Utility Parity for Fair Federated Learning under Intermittent Client Participation  
**Domains**: Trustworthy-ML, Theory

## Summary

Proposes "cumulative utility parity" (CUP): a new FL fairness notion that evaluates whether clients receive comparable long-term utility per participation opportunity (u_k(T)/π_k), rather than per-round accuracy. Addresses a gap left by per-round fairness methods (q-FFL, PHP-FL) that implicitly assume comparable participation frequency.

## Core Technical Contributions

1. Availability-normalized cumulative utility ũ_k(T) = u_k(T)/π_k
2. Inverse-availability sampling: selects clients ∝ 1/π̂_k to equalize selection rates
3. Reactive missed-round reweighting: boosts weights of clients with cumulative missed rounds
4. Surrogate updates for unavailable clients: uses cached checkpoints with exponential staleness decay

## Experimental Results (Table 2)

- Utility CV (lower is better): 0.19 vs 0.42 (PHP-FL) vs 0.64 (q-FFL)
- Jain's accuracy index: 0.975 (proposed) vs 0.80 (PHP-FL) vs 0.72 (q-FFL)
- Setup: 100 clients, CIFAR-10, ResNet-18/34, real device availability traces

## Key Concerns

1. **Missing convergence rates**: Only asymptotic (T→∞) convergence shown; no finite-sample rate
2. **Single-dataset, small-scale**: CIFAR-10 only, 100 clients — limits generalizability
3. **Missing baselines**: FedAvg, Ditto, FairFedCS, FedFV not included in Table 2
4. **Surrogate mechanism not ablated**: Impact of surrogate updates on fairness not isolated
