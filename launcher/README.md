# Launcher

*This README should be updated with the launcher subteam's approach.*

---

The launcher is responsible for instantiating and running agents on the platform. From the project discussion:

- Agents are instantiated via a **Cartesian product** of: evaluation role × research interests × persona × scaffolding.
- The simulation initializes a pool of papers, assigns agents to papers aligned with their interests, then runs an interaction loop for a fixed horizon.
- The system needs to support 50–200 concurrent agents for the retreat simulation.

The launcher consumes the fully-assembled system prompts from `aggregator/global_prompt.py` and hands them off to the harness for execution.
