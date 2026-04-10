# Evaluation Roles

*This README was seeded from the project discussion summary. The roles subteam should update this with their own approach.*

---

Each agent is assigned one evaluation role that defines its primary reviewing lens. From the project discussion, the proposed roles are:

- **Novelty** — Is the contribution genuinely new? Requires search over related work.
- **Rigor** — Are the claims well-supported by experiments, baselines, and statistical evidence?
- **Reproducibility** — Can the results be reproduced? May involve GPU access to run code.
- **Ethics** — Are there harms, biases, or ethical gaps the paper doesn't address?

The role prompt for a given agent will be passed into `global_prompt.build_agent_prompt()` as the `role_prompt` argument.
