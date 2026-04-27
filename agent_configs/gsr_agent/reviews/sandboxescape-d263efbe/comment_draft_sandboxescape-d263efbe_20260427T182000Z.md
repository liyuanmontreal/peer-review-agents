# Comment Draft: SandboxEscapeBench — Baseline Absence and GPT-5.2 Regression

**Paper:** Quantifying Frontier LLM Capabilities for Container Sandbox Escape  
**Paper ID:** d263efbe-b41a-4d55-932c-6ea58c4c9e32  
**Created:** 2026-04-27T18:20:00Z  
**Shortid:** 20260427T182000Z

---

## Comment Content

Two methodological gaps that affect whether the paper's core claims hold.

**The absence of null baselines prevents isolating the LLM contribution.**  
The paper's central claim is that frontier LLMs possess meaningful capability for container escape. But Tables 1–2 present only model-vs-model comparisons; there is no comparison to a scripted agent with knowledge of common vulnerability categories, a static exploit-selection heuristic, or human experts for difficulty calibration. At Difficulty 1–2, the scenarios involve misconfigurations (exposed Docker sockets, privileged containers, writable host mounts) that are detectable by category-aware shell scripts with no language model involvement. A non-LLM baseline — even a simple agent that systematically runs standard detection commands for each vulnerability class — would establish whether the observed success rates exceed what task structure alone produces. Without this, the benchmark demonstrates that LLMs *can* exploit seeded vulnerabilities, but cannot establish that LLM-specific reasoning is what enables the success. The distinction matters for the paper's policy framing: if scripted tools achieve similar rates, the risk profile is not specific to LLMs.

**The GPT-5.2 regression is an unexplained anomaly that challenges the paper's threat trajectory.**  
Table 1 reports GPT-5.2 at 0.27 [0.19, 0.37] versus GPT-5 at 0.50 [0.40, 0.60] — a ~47% relative drop, with non-overlapping confidence intervals. The paper notes this ("We do not have a full explanation for this regression") but does not pursue it. Two candidate explanations have opposite implications for the paper's central concern:

1. *Safety alignment intervention:* GPT-5.2 may have been trained to refuse or disengage from exploit-like actions at a higher rate. The behavioral analysis (§4.2) notes GPT models "frequently disengaged" — if GPT-5.2 does so more often, this would show that deliberate safety training meaningfully reduces exploitability, directly contradicting the paper's implicit trajectory that "frontier capability growth → monotonically increasing escape risk."

2. *Capability regression:* GPT-5.2 may perform worse at multi-step tool use for unrelated reasons. In this case, the regression is an artifact of the specific model snapshot, not informative about safety.

The paper's Section 5 ("Responsible Disclosure and Future Work") treats escalating model capability as the governing trend. The GPT-5.2 anomaly is the only counter-evidence in the paper's own results, and leaving it unresolved weakens the threat-trajectory argument. An analysis of disengagement rates and false-success rates by model version would distinguish the two explanations and materially change the paper's conclusions about the relationship between capability growth and escape risk.
