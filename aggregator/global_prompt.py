"""
global_prompt.py

Assembles the full system prompt for an agent from pre-written section strings.
The content of each section (role, persona, research interests, scaffolding) is
owned by the respective subteams and passed in here — this module only formats.
"""

from pathlib import Path

GLOBAL_RULES_PATH = Path(__file__).parent / "GLOBAL_RULES.md"

SECTION_SEPARATOR = "\n\n---\n\n"


def load_global_rules() -> str:
    return GLOBAL_RULES_PATH.read_text(encoding="utf-8")


def build_agent_prompt(
    role_prompt: str,
    research_interests_prompt: str,
    persona_prompt: str,
    scaffolding_prompt: str,
) -> str:
    """
    Assemble the full system prompt for an agent.

    Args:
        role_prompt: Evaluation role instructions (novelty, rigor, reproducibility, ethics).
                     Defined by the Evaluation Roles subteam.
        research_interests_prompt: Research domain context for the agent.
                                   Defined by the Research Interests subteam.
        persona_prompt: Behavioral style and disposition of the agent.
                        Defined by the Personas subteam.
        scaffolding_prompt: Available tools and harness instructions (memory, search, GPU).
                            Defined by the Scaffolding/Harness subteam.

    Returns:
        The full system prompt string to pass to the agent.
    """
    sections = [
        load_global_rules(),
        role_prompt,
        research_interests_prompt,
        persona_prompt,
        scaffolding_prompt,
    ]
    return SECTION_SEPARATOR.join(s.strip() for s in sections if s and s.strip())
