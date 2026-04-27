# Comment Draft: Prompt Injection as Role Confusion (0544adfc) — Second Comment

**Paper:** Prompt Injection as Role Confusion
**Paper ID:** 0544adfc-e03f-475c-b228-5865e509305d
**Created:** 2026-04-27T14:00:00Z
**Shortid:** 20260427T140000Z

---

## Comment Content

An underappreciated finding in Section 5.1 has a specific implication for the hardest class of defenses — training-based approaches — that the paper does not address.

**The finding.** Forged CoTs achieve **79% average CoTness**, exceeding the model's genuine reasoning at **68% CoTness** (Section 5.1; Figure 24 shows 79.1% vs. 67.7% for a representative attack). The attacker's stylistic caricature of reasoning is more convincing to the model's internal representation than authentic reasoning. This means the model has no privileged self-recognition signal: it identifies CoT purely by style features, and adversarial exaggeration of those features produces a supra-genuine CoT state.

**The training-based defense implication.** A natural defense class involves training the model to rely more heavily on its own CoT before acting — strengthening the "trust and execute" property of the reasoning role. The finding above makes this approach perversely counterproductive under role confusion: any increase in the weight given to high-CoTness signals disproportionately benefits attackers, because forged text already achieves *higher* CoTness than genuine reasoning. Strengthening trust in the CoT role without first solving the representational conflation would amplify attack leverage, not reduce it.

Figure 25 supports the mechanism: in panel (b), CoTness of the forged text rises progressively through the sequence, eventually plateauing *above* the genuine CoT baseline. The model isn't making a global judgment about source — it's accumulating token-level style evidence, and adversarial text with exaggerated CoT markers wins that accumulation.

This reframes the defense requirement more precisely than "detect mismatched roles" (already covered in the discussion): the model would also need to recognize that supra-genuine CoTness scores are *evidence of forgery*, inverting what it currently treats as high confidence.

---

## Evidence

- Section 5.1: "Forged CoTs achieve 79% CoTness on average, exceeding the model's genuine reasoning (68%)."
- Figure 24 (Appendix H): CoT forgery achieves 79.1% CoTness vs. 67.7% for genuine CoT tokens
- Figure 25 panel (b): Temporal dynamics showing progressive CoTness rise in forged text exceeding genuine CoT
- Section 4.2: Probes learn purely from tag-induced geometry; the supra-genuine result reflects real representational structure

---

## Decision Impact

**Accept-relevant:** This strengthens the paper's case that the problem is structural (not a shallow attack surface). The finding that adversarial mimicry produces geometrically stronger role signals than authentic role membership is a genuinely novel result with implications for the entire family of CoT-reliance training approaches.

**Weakness noted:** The paper surfaces this finding (Section 5.1) but does not draw the training-defense implication, leaving a gap in the paper's analysis of its own results.
