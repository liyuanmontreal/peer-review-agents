# Paper Summary: Uncovering Context Reliance in Unstructured Knowledge Editing

**Paper ID:** a2225f35-7e0a-4051-9e62-72dd01763783  
**ArXiv:** 2602.19043  
**Domains:** NLP  
**Status:** in_review (released 2026-04-26T20:00)

## Summary

The paper identifies "Context Reliance" (CR) as a failure mode in NTP-based unstructured knowledge editing: models learn to recall edited knowledge only when the training context is present, failing at inference without it. The paper provides (1) theoretical analysis of why gradient-based optimization induces CR, (2) empirical validation that prepending context during inference recovers recall, and (3) COIN (COntext-INdependent editing) framework that reduces CR by 45.2%.

## Key Claims

- CR is an inherent consequence of gradient-based optimization (Theorem 3.2)
- COIN reduces CR by 45.2% (ROUGE decline metric)
- COIN outperforms best FT baseline by 23.6% on AKEW

## Setup

**Models:** Llama3-8B, Qwen2.5-7B (pretrained + instruction-tuned)  
**Benchmarks:** AKEW (975 instances), UnKEBench, MQuAKE (multi-hop)  
**NOT used:** CounterFact, zsRE (standard benchmarks)

**Baselines (main benchmarks):** FT, LoRA, AdaLoRA, UnKE, AnyEdit, AlphaEdit-D  
**Baselines (MQuAKE only):** ROME, MEMIT, AlphaEdit (structured methods)

## COIN Method

Loss: ℒ_COIN = ℒ_NLL + α·ℒ_align + β·ℒ_cons

- **ℒ_align**: KL divergence between predictions under local-context (k-token window) vs. global-context
- **ℒ_cons**: Constrains weight changes along frequently-activated directions (prevents forgetting)

## Theory

Theorem 3.2: Under Assumption 3.1 (single gradient step, simplified transformer, one relevant + one context token), gradient optimization binds target token prediction to aggregated context representation.

**Major caveat:** Single gradient step only, multi-step dynamics unaddressed.

## Internal Assessment

**Score band:** 5.0–6.99 (weak accept, possibly 3.0–4.99 depending on baseline omission)

**Key issues:**
1. Theory-method gap: Theorem proves representation-level binding, COIN addresses prediction-level alignment
2. ROME/MEMIT excluded from main benchmarks — weakens claim that CR is NTP-specific
3. Custom benchmarks only (AKEW, UnKEBench), no standard benchmarks
4. Theory valid for 1 gradient step; practical COIN uses many steps
