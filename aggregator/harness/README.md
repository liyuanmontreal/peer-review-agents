# Harness / Scaffolding

*This README was seeded from the project discussion summary. The harness subteam should update this with their own approach.*

---

The harness owns the agent execution loop and any scaffolding that goes beyond the prompt. From the project discussion:

- The proposed agent scaffold is [OpenHands](https://github.com/OpenHands/OpenHands).
- Scaffolding dimensions include: memory (context compression, interaction history), tool access (literature search, GPU/code execution), and action dispatch (post, comment, vote, view).
- Unlike the other subteams, this is not just a prompt — it is the code that runs the agent.

The scaffolding prompt (describing available tools and capabilities to the agent) will be passed into `global_prompt.build_agent_prompt()` as the `scaffolding_prompt` argument. The harness team owns both the prompt and the actual tool implementations.
