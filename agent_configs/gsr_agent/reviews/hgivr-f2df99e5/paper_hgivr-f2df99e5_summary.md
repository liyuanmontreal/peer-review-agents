# Paper Summary: H-GIVR

**Platform ID:** f2df99e5-b955-4a1d-9f7e-3294ccd55951  
**ArXiv:** 2602.04413  
**Title:** History-Guided Iterative Visual Reasoning with Self-Correction

## What the Paper Does

Proposes H-GIVR, an inference-time framework for multimodal LLMs that:
1. Generates an image description before answering (Visual Description module)
2. Feeds previous answers as historical context into each iteration (Consistency-Iterative Reasoning)
3. Re-generates the image description every even iteration (Image Re-observation Mechanism)
4. Stops when two identical answers appear or after 10 iterations (Answer Confirmation Mechanism)

The framework requires no training — it is prompt-engineering only, implemented via Ollama.

## Claimed Contributions

- Moves beyond "repeated sampling + majority voting" self-consistency
- Achieves 107% improvement on ScienceQA with Llama3.2-vision:11b
- Average 2.57–4.04 model calls per question (computationally efficient)

## Models and Datasets

- **Models**: Llama3.2-vision:11b, Qwen2.5vl:7b, Gemma3:12b (via Ollama)
- **Datasets**: ScienceQA, A-OKVQA, OK-VQA, VQAv2, TextVQA
- **Baselines**: Standard, FS-CoT, Auto-CoT, Active-Pro, Self-Consistency

## Critical Observations

### 1. Anomalously Low Baselines
The "Standard" baseline for Llama3.2-vision:11b on ScienceQA is 38.08%, far below published benchmarks for this model class (typically 60-75%+ without fine-tuning). The baseline for Qwen2.5vl:7b is 40.16% and Gemma3:12b is 40.16% — all suspiciously similar and low. This suggests the Ollama inference setup is misconfigured or not reproducing standard evaluation conditions.

### 2. Table 1 Paradox: False Answers Outperform H-GIVR
Table 1 shows:
- Standard: 38.08%
- False (incorrect historical answers): 83.33%
- H-GIVR: 78.90%

Providing explicitly *wrong* historical answers outperforms H-GIVR. The paper says this "indicates that incorrect historical answers cannot mislead the model", but this doesn't explain why False > H-GIVR. This undermines the claim that correct historical context is essential.

### 3. Complex CoT Underperforms Standard
The paper notes H-GIVR with Complex CoT prompts achieves *lower* accuracy than H-GIVR without prompts. This suggests the framework is fragile to prompt design.

### 4. Missing Modern Baselines
No comparison with multimodal self-correction methods: SCoRe (Kumar et al., 2024), RISE (Qu et al., 2024), or S2R (Ma et al., 2025).

## Preliminary Score Band
Weak reject (3.0–4.99): The core idea is reasonable but the anomalous baselines and Table 1 paradox raise serious validity concerns.
