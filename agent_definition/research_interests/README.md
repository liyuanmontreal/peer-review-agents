# Research Interests

*This README was seeded from the project discussion summary. The research interests subteam should update this with their own approach.*

---

Each agent is given a set of research interests that bias which papers it selects and how it weighs its own expertise when reviewing. From the project discussion:

- Research interests are one dimension of the Cartesian product used to instantiate agents.
- Their primary purpose is to reduce topical bias across the agent population — by varying interests, the aggregate leaderboard is less skewed toward any one research area.
- Agents should be aware of when they are reviewing outside their area of expertise.

The research interests prompt for a given agent will be passed into `prompt_builder.build_prompt()` as the `research_interests_prompt` argument.

Current contents: `ml_taxonomy.json` (unified ML topic taxonomy) and tooling to generate interest prompts. Prompt files are pending from the subteam.
