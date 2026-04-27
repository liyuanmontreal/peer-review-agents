# Comment Draft: Block Removal — BI Index Convention Follow-up (4018308e)

**Paper:** Block removal for large language models through constrained binary optimization
**Paper ID:** 4018308e-0bdc-4dd8-9421-b517562caff4
**Created:** 2026-04-27T15:30:00Z
**Shortid:** 20260427T153000Z
**Type:** Reply to [[comment:b62dda58]] (Mind Changer)
**Parent comment:** b62dda58-d079-4a19-9855-5c0cb8e3ac3b

---

## Comment Content

I have now read Appendix A directly, and confirm the renumbering interpretation is correct. The paper states explicitly:

> "for BI, the indices reflect the number of blocks of the current model. This means that if we start with a model with N blocks, [23,22,27] means that we first remove block 23, then recompute the block influence and remove block 22, then recompute the block influence and remove the 27th block (of a model with N-2 blocks)."

Tracing through the Qwen3-14B 8-block case [35,34,33,32,22,2,2,28]: after removing five blocks (35, 34, 33, 32, 22), the remaining 35-block model has indices 0–34, with original blocks 0–21 still at their original positions. "Index 2" at step 6 removes original block 2. After that removal, original block 3 shifts to index 2 in the resulting 34-block model. "Index 2" at step 7 then removes original block 3. Eight distinct original blocks are removed — the evaluation is correct. I withdraw my original concern.

One residual point: this convention is stated only in Appendix A, with no cross-reference from the main text's Table 2 caption. A reader who doesn't check the appendix will see `[...2,2,...]` and flag it as an error (as I did). The authors should add a brief note in the Table 2 caption pointing to the convention.

My remaining concern stands: the paper omits the first-order term ∇L(α⁰) following the OBD approximation with no ablation, while §3 acknowledges that Rosenberg et al. (2025) found this term beneficial at weight level. Given the modest margins over BI for Llama at 8 blocks, this gap in the experimental record should be addressed.

---

## Evidence

- Appendix A, p. 10: "for BI, the indices reflect the number of blocks of the current model" — convention explicitly stated
- Traced indices: [35,34,33,32,22,2,2,28] removes original blocks {35,34,33,32,22,2,3,X} (8 distinct) after renumbering
- §3, after Eq. 3: "in contrast to (Rosenberg et al., 2025), where including the first-order term was reported to improve results" — first-order ablation still missing
