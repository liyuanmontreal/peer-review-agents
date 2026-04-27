# Paper Summary: Under the Influence: Quantifying Persuasion and Vigilance in Large Language Models

**Paper ID:** 588e7124-aedd-4875-b033-013600ea9b51  
**ArXiv:** 2602.21262  
**Domains:** NLP, Trustworthy-ML  
**Status:** in_review (released 2026-04-27T16:00)

## Summary

The paper uses Sokoban (a box-pushing puzzle game) to study two social capacities in LLMs: vigilance (detecting when advice should be discarded) and persuasion (generating convincing, behavior-shifting advice). Five frontier models are tested: GPT-5, Grok 4 Fast, Gemini 2.5 Pro, Claude Sonnet 4, DeepSeek R1.

## Setup

- 10 Sokoban puzzles with 2 boxes and 2 goals
- Player models receive advice from advisor models in multi-turn setting
- Two advice conditions: benevolent (helpful) and malicious (designed to cause failure)
- Two awareness conditions: unaware (deception not disclosed) and aware (explicitly told deception possible)
- Persuasion metric: measures how much advisor shifts player behavior toward target outcome
- Vigilance metric: rewards following good advice / ignoring bad advice; penalizes opposite

## Key Results

- **Performance:** GPT-5 100%, Grok 4 Fast 98%, Claude Sonnet 4 28% unassisted
- **Dissociation claim:** No correlation between unassisted performance and persuasion (p=.796) or vigilance (p=.328)
- **Token finding:** Benevolent advice → fewer tokens (t(49)=3.241, p=.002); Malicious + solvable → 16.1% more tokens (p<.001); Malicious + already-failing → 64.6% fewer tokens (p<.001)
- **Awareness effect:** Explicitly mentioning deception substantially improves vigilance for Gemini (−0.422 → 0.629) and GPT-5 (0.760 → 0.960)
- **Claude anomaly:** Claude Sonnet 4 provided benevolent advice despite malicious instructions (safety guardrails override)

## Internal Assessment

**Score band:** 5.0–6.99 (weak accept range, pending full evaluation)

**Strengths:**
- First unified investigation of persuasion + vigilance in same framework
- Objective task with ground truth (Sokoban has optimal solutions)
- Multiple frontier models; clear metric formalization

**Concerns:**
- Token analysis confounded by task difficulty (not just deception detection)
- Dissociation finding limited by bimodal capability distribution and Claude guardrail
- Limited to game setting; no human baseline; 2-box constraint

**Open questions:**
- Does token modulation reflect genuine detection or just advice-vs-inclination conflict?
- Would the correlation analysis hold within high-capability models only?
