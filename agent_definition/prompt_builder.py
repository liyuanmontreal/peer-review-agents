"""
global_prompt.py

Assembles the full system prompt for an agent from pre-written section strings.
This module only handles formatting — the content of each section is defined elsewhere.
"""

from pathlib import Path

GLOBAL_RULES_PATH = Path(__file__).parent / "GLOBAL_RULES.md"

SECTION_SEPARATOR = "\n\n---\n\n"


def load_global_rules() -> str:
    return GLOBAL_RULES_PATH.read_text(encoding="utf-8")


def build_prompt(
    role_prompt: str,
    research_interests_prompt: str,
    persona_prompt: str,
    scaffolding_prompt: str,
) -> str:
    """
    Assemble the full system prompt for an agent from its component sections.
    Global rules are prepended automatically from GLOBAL_RULES.md.
    """
    sections = [
        load_global_rules(),
        role_prompt,
        research_interests_prompt,
        persona_prompt,
        scaffolding_prompt,
    ]
    return SECTION_SEPARATOR.join(s.strip() for s in sections if s and s.strip())
