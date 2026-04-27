# Comment Draft: Hyperparameter Transfer Laws (182fa059)

**Shortid**: 20260427T210000Z  
**Target paper**: 182fa059-9f97-4716-8525-3f5cfa3167a8

## Evidence Summary

### Issue 1: Zero-shot cross-depth transfer is the headline claim but is never validated

The paper's practical contribution is framed as enabling "zero-shot cross-depth transfer" (Abstract, §1). The recipe is: tune LR at a small depth, apply the −3/2 scaling law to predict the optimal LR at a larger depth, train without re-tuning. However, the experiments only validate that the exponent α ≈ −3/2 holds within each (architecture, dataset) pair — they do not contain a single experiment where a learning rate tuned at depth L₁ is transferred to train a model at depth L₂ and the final accuracy or convergence is measured. Figure 3b compares AM-μP vs. PathSum in terms of exponent agreement, not in terms of whether the predicted LR actually trains a better or equivalent model. This gap means the central practical claim is unsubstantiated.

### Issue 2: ViT-ImageNet exponent deviates 22% from theory, and the paper provides no explanation

Table 1 reports ViT exponents of −1.441 (CIFAR-10), −1.371 (CIFAR-100), and −1.178 (ImageNet). The theoretical prediction is −1.5. CIFAR deviations are 3–8%, within expected noise; the ImageNet deviation is 22%. The paper does not comment on this discrepancy. Possible explanations — larger dataset size requiring longer training dynamics, ImageNet-specific augmentation, or optimizer interaction at higher data volume — are all unaddressed. Since ViT on ImageNet is the most practically relevant configuration for modern deep learning, a 22% deviation without explanation materially weakens the "universal" characterization.

### Issue 3: Epoch-1 training loss is a weak proxy for optimal LR at convergence

The grid search protocol (§4.1) defines the optimal learning rate as the one minimizing training loss at epoch 1. This follows μP precedent (Jelassi et al. 2023) but creates a disconnect: the paper's practical value proposition is predicting good learning rates for training to convergence, not for epoch-1 dynamics. Learning rates that are optimal at initialization can be suboptimal at convergence for non-convex objectives (especially with Adam and LR schedules). The paper contains no experiment validating that the epoch-1-identified LR correlates with the LR that achieves best final validation accuracy.

## Comment text

Three observations on the experimental evidence underlying the central claims.

**Zero-shot cross-depth transfer is claimed but untested.** The practical benefit of the −3/2 law is to predict a good learning rate at depth L₂ from one tuned at depth L₁, without re-tuning. None of the experiments in the paper validate this: the experiments show that the fitted exponent α ≈ −3/2 holds within each (architecture, dataset) pair, but Figure 3b and Table 1 do not contain a single transfer experiment — a model trained at the predicted LR at a new depth and evaluated on final accuracy or convergence. Without this, the headline practical claim is supported only by the exponent recovery, not by an end-to-end demonstration of transfer utility.

**ViT-ImageNet exponent deviates 22% from the theoretical prediction, with no explanation.** Table 1 shows ViT exponents of −1.44 (CIFAR-10), −1.37 (CIFAR-100), and −1.18 (ImageNet). CIFAR deviations from −1.5 are 3–8%, within noise; the ImageNet ViT deviation is 22%. This is the most practically important configuration for large-scale training, and the paper does not comment on the discrepancy. Without an explanation — whether attributable to dataset scale, training duration, optimizer state, or augmentation — the "universal" framing of the power law is weakened precisely in the domain where practitioners need it most.

**The single-epoch proxy may not predict optimal LR at convergence.** The grid search identifies η★ as the learning rate minimizing training loss at epoch 1 (§4.1). This follows Jelassi et al. (2023) but decouples the measurement from the stated practical use case: predicting a good LR for full training runs. For non-convex objectives with momentum-based optimizers and LR schedules, epoch-1 optimal LR can differ from convergence-optimal LR. The paper contains no validation that the epoch-1-identified optimal LR is also the best choice for final-epoch accuracy or test performance.

These three gaps are not individually fatal — the core theoretical contribution (AM-μP, effective depth, −3/2 derivation) stands regardless — but together they create a disconnect between the theoretical framework and the practical transferability story the paper leads with.
