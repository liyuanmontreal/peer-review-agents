"""
prepare_agents.py

Generates agent directories from a Cartesian product of role × research interests
× persona × scaffolding. Each output directory contains a CLAUDE.md (assembled
system prompt) and a copy of the MCP config, ready for run_agents.py to consume.

Usage:
    python launcher/prepare_agents.py \
        --roles agent_definition/roles/01_novelty_and_originality.md agent_definition/roles/02_technical_soundness.md \
        --interests agent_definition/research_interests/nlp.md \
        --personas agent_definition/personas/optimistic.json agent_definition/personas/pessimistic.json \
        --scaffolding agent_definition/harness/scaffolding.md \
        --coalescence-api-key cs_... \
        --output-dir agent_configs/
"""

import argparse
import json
from itertools import product
from pathlib import Path

from agent_definition.prompt_builder import build_prompt


def load(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def persona_to_prompt(path: str) -> str:
    """Convert a persona JSON or .md file to a prompt string."""
    p = Path(path)
    if p.suffix != ".json":
        return p.read_text(encoding="utf-8")

    d = json.loads(p.read_text(encoding="utf-8"))
    lines = [f"## Persona: {d['name']}", f"\n{d['description']}"]

    traits = {k: v for k, v in d["trait_vector"].items() if v != 0}
    if traits:
        lines.append("\n### Traits")
        for trait, value in traits.items():
            definition = d["trait_definitions"].get(trait, "")
            direction = "High" if value == 1 else "Low"
            lines.append(f"- **{trait}** ({direction}): {definition}")

    if d.get("behavioral_rules"):
        lines.append("\n### Behavioral rules")
        lines.extend(f"- {r}" for r in d["behavioral_rules"])

    if d.get("forbidden_behaviors"):
        lines.append("\n### Do not")
        lines.extend(f"- {r}" for r in d["forbidden_behaviors"])

    return "\n".join(lines)


def prepare_agents(
    roles: list[str],
    interests: list[str],
    personas: list[str],
    scaffolding: str,
    coalescence_api_key: str,
    output_dir: Path,
) -> list[Path]:
    """
    Generate one agent directory per combination of role × interests × persona.
    Returns list of created agent directories.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    scaffolding_prompt = load(scaffolding)
    agent_dirs = []

    for i, (role, interests_path, persona) in enumerate(product(roles, interests, personas)):
        role_name = Path(role).stem
        interests_name = Path(interests_path).stem
        persona_name = Path(persona).stem
        agent_name = f"agent_{i:03d}__{role_name}__{interests_name}__{persona_name}"

        agent_dir = output_dir / agent_name
        agent_dir.mkdir(exist_ok=True)

        system_prompt = build_prompt(
            role_prompt=load(role),
            research_interests_prompt=load(interests_path),
            persona_prompt=persona_to_prompt(persona),
            scaffolding_prompt=scaffolding_prompt,
        )
        (agent_dir / "CLAUDE.md").write_text(system_prompt, encoding="utf-8")

        settings = {
            "mcpServers": {
                "coalescence": {
                    "type": "url",
                    "url": "https://coale.science/mcp",
                    "headers": {"Authorization": f"Bearer {coalescence_api_key}"},
                }
            }
        }
        claude_dir = agent_dir / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

        agent_dirs.append(agent_dir)
        print(f"  created: {agent_dir.name}")

    print(f"\n{len(agent_dirs)} agent configs written to {output_dir}")
    return agent_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", nargs="+", required=True)
    parser.add_argument("--interests", nargs="+", required=True)
    parser.add_argument("--personas", nargs="+", required=True)
    parser.add_argument("--scaffolding", required=True)
    parser.add_argument("--coalescence-api-key", required=True)
    parser.add_argument("--output-dir", default="agent_configs/")
    args = parser.parse_args()

    prepare_agents(
        roles=args.roles,
        interests=args.interests,
        personas=args.personas,
        scaffolding=args.scaffolding,
        coalescence_api_key=args.coalescence_api_key,
        output_dir=Path(args.output_dir),
    )
