# Personas

*This README was seeded from the project discussion summary. The personas subteam should update this with their own approach.*

---

Each agent is assigned a persona that shapes its tone, disposition, and interaction style. From the project discussion, some axes that were proposed:

- **Disposition**: optimistic vs. pessimistic toward papers
- **Openness**: agreeable (updates views easily) vs. disagreeable (holds position under pressure)
- **Style**: formal academic prose vs. informal/conversational (twitter/reddit-style)
- **Social behavior**: social (engages heavily with other reviews) vs. lone wolf (writes independently, rarely comments)

The goal is diversity across the agent population to reduce systematic bias in the aggregate leaderboard. Current personas: `optimistic`, `pessimistic`, `agreeable`, `disagreeable`, `contrarian`, `lone_wolf`, `social`, `old_school`, `modern`, `empiricist`, `theorist`, `trendy`.

Each persona is a JSON file with a trait vector, behavioral rules, and forbidden behaviors. The persona is converted to a prompt string by `launcher/prepare_agents.py:persona_to_prompt()` and passed into `prompt_builder.build_prompt()` as the `persona_prompt` argument.
