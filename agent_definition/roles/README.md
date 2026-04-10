# Evaluation Roles

*This README was seeded from the project discussion summary. The roles subteam should update this with their own approach.*

---

Each agent is assigned one evaluation role that defines its primary reviewing lens. Current roles:

| File | Role |
|------|------|
| `01_novelty_and_originality.md` | Is the contribution genuinely new? |
| `02_technical_soundness.md` | Are the methods and theory correct? |
| `03_experimental_rigor.md` | Are the experiments well-designed and results reliable? |
| `04_reproducibility_and_transparency.md` | Can the results be reproduced? |
| `05_clarity_and_presentation.md` | Is the paper clear and well-written? |
| `06_significance_and_impact.md` | Does the work matter to the field? |
| `07_ethics_and_responsible_research.md` | Are there ethical concerns? |
| `08_completeness_and_limitations.md` | Does the paper acknowledge its limitations? |

The role prompt is passed into `prompt_builder.build_prompt()` as the `role_prompt` argument.
