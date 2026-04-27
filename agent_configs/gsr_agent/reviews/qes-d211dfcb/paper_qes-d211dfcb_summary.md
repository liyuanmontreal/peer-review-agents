# Paper Summary: Quantized Evolution Strategies (QES)

**Paper ID**: d211dfcb-6d54-4810-bedf-1e666c322c63  
**Title**: Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs at Low-precision Cost  
**Domains**: d/NLP, d/Optimization

## Core Contribution

QES enables fine-tuning of post-training quantized LLMs directly in discrete parameter space, solving two failure modes of existing zeroth-order methods (QuZO):
1. **Gradient stagnation**: Updates too small to cross quantization threshold → ΔW=0
2. **Discretization error accumulation**: Random walk-scale noise drowns signal

**Mechanism:**
- Accumulated error feedback (Delta-Sigma modulation): maintain FP16 error vector et, carry forward fractional updates until they cross quantization threshold
- Stateless seed replay: eliminate O(d) error storage by replaying last K steps using stored seeds and scalar rewards

**Theoretical claim**: "Temporal equivalence" - virtual parameters Θt = Wt + et follow exact high-precision gradient ascent trajectory, with deviation bounded by ||eT||∞ ≤ Δ/2

## Evaluation

Single task: Countdown arithmetic reasoning (3-number arithmetic) with Qwen2.5-1.5B and 3B models in INT4/INT8/W8A8 formats.

**Table 1 key results:**
- 1.5B INT4: QES 16.00 vs Full Res. 18.05 (-2.05) — expected
- 1.5B INT8: QES 26.35 vs Full Res. 22.10 (+4.25) — PARADOX: QES > oracle
- 3B INT4: QES 31.85 vs Full Res. 33.50 (-1.65) — expected  
- 3B INT8: QES 37.40 vs Full Res. 33.30 (+4.10) — PARADOX: QES > oracle
- 3B W8A8: QES 21.35 vs Full Res. 31.70 (-10.35) — LARGE UNEXPLAINED GAP

## Score Assessment

Score band: 4.0–5.99 (weak reject/accept boundary)
- Novel application of Delta-Sigma modulation to quantized LLM fine-tuning
- Concerns: single-task evaluation, paradoxical Table 1 results unexplained, temporal equivalence doesn't directly prove solution quality
