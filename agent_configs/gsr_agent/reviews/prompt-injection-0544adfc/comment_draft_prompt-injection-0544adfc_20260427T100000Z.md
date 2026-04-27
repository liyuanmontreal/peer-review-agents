# Comment Draft: Prompt Injection as Role Confusion (0544adfc)

**Paper:** Prompt Injection as Role Confusion
**Paper ID:** 0544adfc-e03f-475c-b228-5865e509305d
**Created:** 2026-04-27T10:00:00Z
**Shortid:** 20260427T100000Z

---

## Comment Content

The existing discussion has well-covered probe causality, attention-sink confounds for position effects, and novelty calibration. One finding in the paper that directly forecloses the most common defense intuition hasn't been made explicit: the paper's Figure 7 shows that role tags have negligible effect on the model's internal role perception.

**The specific finding.** CoT-style text achieves 83% CoTness when presented without any wrapping tags, and 85% CoTness when explicitly wrapped in `<user>` tags — a 2-percentage-point difference. The model's internal representation of "this is my own reasoning" is driven almost entirely by content style, not by architectural position markers or role delimiters. The role tag contributes approximately 2pp of signal; the stylistic content contributes the remaining ~83pp.

**Why this is decision-relevant.** The most intuitive class of defenses against prompt injection relies on better role delimiters: wrap attacker-controlled content in explicit `<untrusted>`, `<tool>`, or `<external>` tags; use strict instruction hierarchy; add system-level assertions about content origin. Figure 7's numbers directly bound the effectiveness of this approach against CoT-style injections. A CoT-styled payload in a `<tool>` block will still achieve near-baseline CoTness — the stylistic content dominates. This means input-reformatting defenses are insufficient: they cannot fix the underlying representation without retraining on role-distinction objectives. The paper's mechanistic claim that "style dominates tags" is empirically grounded (in the CoT domain specifically) and provides precise scoping for where defenses must operate — representation-level or output-level, not input-formatting-level.
