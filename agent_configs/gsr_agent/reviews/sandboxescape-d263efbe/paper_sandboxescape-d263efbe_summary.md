# Paper Summary: SandboxEscapeBench

**Paper ID:** d263efbe-b41a-4d55-932c-6ea58c4c9e32  
**Title:** Quantifying Frontier LLM Capabilities for Container Sandbox Escape  
**arXiv:** 2603.02277  
**Domains:** Trustworthy-ML, NLP

## Core Contribution

Introduces SANDBOXESCAPEBENCH, a CTF-based benchmark measuring LLM ability to escape Docker/OCI container sandboxes. Uses nested sandbox architecture (inner: container under attack; outer: VM with flag file). 18 scenarios across three security layers: L1 orchestration, L3 engine/runtime, L4 host/kernel, spanning misconfigurations, CVEs, and kernel exploits.

## Threat Model

Root access inside container, shell access, standard Linux tools, network access. Intentionally introduced weaknesses. Explicitly excludes hardware side-channels, hypervisor escapes, defended environments.

## Key Results

- Opus 4.5: 0.49 [0.39, 0.59]
- GPT-5: 0.50 [0.40, 0.60]
- GPT-5.2: 0.27 [0.19, 0.37] — sharp regression from GPT-5
- Haiku 4.5: 0.18 [0.11, 0.27]
- Difficulty 3 (multi-step): Frontier models ~0.40 [0.26, 0.56]
- Difficulty 4-5 (kernel exploits): Zero success across all models
- All successful escapes used documented CVEs — no novel vulnerabilities found
- DeepSeek-R1: ~70% false success rate anomaly

## Strengths

- First systematic benchmark for LLM container escape
- Nested sandbox is a clever safety design for evaluation
- Transparent about scope limitations
- Public release of 18 scenarios + tooling

## Weaknesses

- No null baselines (no scripted agent, human expert, non-LLM comparison)
- GPT-5.2 regression vs GPT-5 unexplained (~47% relative drop)
- Outer sandbox vulnerability-free only by assumption (4 unintended paths found during dev)
- All exploits use known CVEs (no zero-day capability measured)
- Wide CIs suggest limited statistical power at 5 runs per scenario
- DeepSeek-R1 false-success rate anomaly not investigated

## Verdict Assessment

Difficulty 1-2 success rates are high but uninterpretable without baselines. Difficulty 3+ results are statistically uncertain. The GPT-5.2 regression challenges the paper's implicit "capability growth → escalating risk" trajectory.
