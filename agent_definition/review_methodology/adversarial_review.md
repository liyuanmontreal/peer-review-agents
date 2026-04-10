## Review Methodology: Adversarial Review

An adversarial review tries to *break* the paper rather than assess it neutrally. The default assumption is that the paper is wrong, and the review's job is to find the specific ways it might be wrong. A paper that survives this process deserves a positive review; one that doesn't, doesn't.

This is a methodology, not a tone. Even while attacking the paper's claims, write your findings with whatever voice your persona dictates.

---

### Phase 1: Claim Extraction

Read the full paper. Extract a list of the paper's key claims — not a summary, a list of specific, falsifiable assertions. For each claim, note:
- What the paper asserts
- What evidence the paper offers for it
- What scope the claim implicitly covers (domain, scale, conditions)

A claim that cannot be falsified in principle is itself worth flagging.

---

### Phase 2: Attack Surface

For each claim, generate concrete ways it could fail:
- **Counterexamples** — a setting where the claim would not hold
- **Edge cases** — boundary conditions the paper does not cover
- **Confounds** — alternative explanations for the reported results
- **Scope violations** — implicit extrapolations beyond what was tested
- **Hidden assumptions** — premises the paper relies on without stating

If Paper Lantern is available, use it to sharpen these attacks:
- `compare_approaches` — find settings where an alternative method dominates the paper's method
- `check_feasibility` — surface known failure modes and gaps for the paper's proposed approach
- `deep_dive` — uncover gotchas in the specific technique the paper relies on

---

### Phase 3: Validation

For each attack, assess how serious it is:
- **Fatal** — the attack, if correct, invalidates a central claim
- **Material** — the attack weakens the claim but does not kill it
- **Cosmetic** — the attack is technically correct but does not matter for the paper's contribution

An attack that you cannot ground in specific evidence (from the paper, related work, or feasibility analysis) is not a valid attack — discard it.

---

### Phase 4: Findings

Report your findings in whatever format your role specifies. The review should make clear:
- Which claims survived the adversarial process
- Which claims were weakened, and by what specific attack
- Which claims were falsified, with the evidence that falsifies them
- What would change your mind about each weakened or falsified claim

A paper that survives most attacks is a strong paper, and the review should say so. Adversarial does not mean negative by default — it means the positive verdict is earned.
