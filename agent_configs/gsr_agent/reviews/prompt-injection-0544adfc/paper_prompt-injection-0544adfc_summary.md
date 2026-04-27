# Paper Summary: Prompt Injection as Role Confusion

**Paper ID:** 0544adfc-e03f-475c-b228-5865e509305d
**Title:** Prompt Injection as Role Confusion
**ArXiv:** 2603.12277
**Status:** in_review
**Domains:** Trustworthy-ML, NLP, Theory

## Core Claim

Prompt injection succeeds because models infer text source from linguistic style rather than architectural position. "Role confusion" is measurable in latent space: text that sounds like a trusted role occupies the same representation space as text that actually is that role.

## Key Technical Components

1. **Role Probes**: Linear classifiers trained on controlled data (identical content wrapped in different role tags). Isolates stylistic vs. positional role signals.
   - CoTness: P(CoT | hidden state h_t)
   - Systemness, Userness: analogous probes for system/user perception
   - Zero-shot generalization: 85% CoTness on actual CoT, 3-5% for user/assistant tokens (§4.2)

2. **CoT Forgery Attack**: Auxiliary LLM generates CoT-style reasoning chains; injected into user prompts or webpages. Models treat injected content as own reasoning.
   - Chat jailbreak ASR: >80% for several frontier models, 17-52% for GPT-5 series (Figure 2)
   - Agent hijacking: 56-70% across all models (Figure 3, vs 0-26% standard injection)
   - Destyling ablation: removes CoT markers → ASR drops from 61% to 10% (Figure 4)

3. **Agent Prompt Injection Generalization** (§5.2):
   - 1,000 injections × 212 templates
   - Userness lowest quantile: 2% ASR; highest quantile: 70% ASR
   - User-declaring templates: 100% of top 5% by Userness (vs 24% of all templates)

## Critical Observation (Figure 7)

CoT-style text → 83% CoTness without any wrapping tags; 85% CoTness with explicit `<user>` tags. The tags shift internal role perception by only 2 percentage points. Content style dominates completely over architectural markers.

**Implication**: Delimiter-based defenses (instruction hierarchy, XML tags, role delimiters) cannot fix the underlying style-driven role assignment without changing model representations.

## Systemness and Position (Appendix K / Figure 11)

Systemness declines monotonically with token position. Inserting system tags had no effect on model perception. Position signals override tags.

## Known Discussion Issues

- Role-probe evidence is correlational, not causal
- Attention-sink confound for position-based Systemness
- Defense target underdetermined
- Perception-vs-memorization binary overstates the probe evidence
- CoT Forgery novelty vs. reasoning-attack literature
