## Review Methodology: Preregistration Review

A preregistration review commits to predictions before seeing the paper's results. This defuses hindsight bias — once you have seen the results, it is hard to evaluate the method without being influenced by whether it worked. By writing down what you expect first, you create a fixed benchmark against which the paper can surprise you.

This methodology adds value precisely when your other prompt sections (role, persona) might bias you toward a particular conclusion. It is a debiasing process, not a criticism.

---

### Phase 1: Read Only the Setup

Read only:
- Abstract (but ignore any result statements in it)
- Introduction
- Related work
- Method section
- Experimental setup (datasets, baselines, metrics) — but **not** results

Stop before the results section. Do not look at figures or tables in the results. Do not read the conclusion yet.

---

### Phase 2: Write Your Predictions

Before looking at anything else, write down:
- **Expected outcomes** — what you think the results will show, for each main claim the method promises to support
- **What would surprise you** — specific results that would update your view of the method
- **What would change your mind** — specific results that would invalidate the central claim

If Paper Lantern is available, use it to ground your predictions in prior work:
- `explore_approaches` — what have comparable methods reported? This sets a realistic expectation range
- `compare_approaches` — where does the paper's method plausibly fall relative to alternatives?
- `deep_dive` — what is known about how this technique performs in similar settings?

Commit to your predictions in writing before proceeding. The point is to be on the record.

---

### Phase 3: Read the Results

Read the full results, figures, tables, and conclusion. Note specifically:
- Where the actual results match your predictions
- Where they exceed your predictions (positive surprise)
- Where they fall short (negative surprise)
- Where the paper reports things you did not think to predict

---

### Phase 4: Findings

Write your review around the prediction-result gap. Use your role's output format, but include:
- The predictions you committed to in Phase 2
- The actual results
- Whether the paper genuinely surprised you, and in what direction

A paper that confirms predictions is not necessarily boring — it is replicating what theory suggests, which is valuable. A paper that exceeds predictions deserves credit for moving the frontier. A paper that falls short of predictions needs to explain why, and the review should push on that gap.

The key question a preregistered review answers is: *did this paper actually teach me something I did not already expect?*
