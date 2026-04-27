# Paper Summary: Hyperparameter Transfer Laws for Non-Recurrent Multi-Path Neural Networks

**Paper ID**: 182fa059-9f97-4716-8525-3f5cfa3167a8  
**Title**: Hyperparameter Transfer Laws for Non-Recurrent Multi-Path Neural Networks  
**Domains**: Deep-Learning, Theory, Optimization

## Summary

The paper extends μP (Yang et al.) to heterogeneous multi-path architectures (CNNs, ResNets, Transformers) and proves a universal −3/2 power law for optimal learning-rate scaling with effective depth. The central contribution is AM-μP (Arithmetic-Mean μP), which replaces per-layer update constraints with a network-wide arithmetic mean budget, enabling depth-transfer of hyperparameters.

## Core Claims

1. Effective depth (minimal input-to-output path length counting layers and residual additions) unifies CNNs, ResNets, and Transformers
2. AM-μP generalizes μP to heterogeneous architectures via merge-consistency axioms
3. η★(L) ∝ L^(-3/2) universally across architectures — enabling zero-shot cross-depth LR transfer
4. Experiments on CIFAR-10/100 and ImageNet confirm ~3-8% agreement with theory

## Empirical Results (Table 1)

- CNN: exponents -1.339 to -1.392
- ResNet: exponents -1.355 to -1.567
- ViT: exponents -1.178 (ImageNet) to -1.441 (CIFAR-10)
- Mean: -1.38

## Key Concerns

1. **Zero-shot cross-depth transfer never validated**: Central practical benefit untested — paper only shows exponent holds within-dataset, never across depths
2. **ViT-ImageNet 22% deviation** (-1.178 vs -1.5 theoretical): Root cause unexplained, undermines universality
3. **Single-epoch training loss proxy**: Grid search minimizes epoch-1 training loss, not final accuracy or validation performance; decoupled from generalization
4. **Theory-practice gap on normalization**: BatchNorm/LayerNorm modify the exponent significantly (BatchNorm ResNet: -1.701 vs -1.435), yet theory doesn't cover normalized networks
