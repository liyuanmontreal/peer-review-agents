"""
prepare_agents.py

Generates agent directories from a Cartesian product of role × research interests
× persona × scaffolding. Each output directory contains a CLAUDE.md (assembled
system prompt). Agents self-register on the platform at runtime.

Usage:
    python launcher/prepare_agents.py \
        --roles agent_definition/roles/*.md \
        --interests agent_definition/research_interests/*.md \
        --personas agent_definition/personas/*.json \
        --scaffolding agent_definition/harness/scaffolding.md \
        --n 50 \
        --output-dir agent_configs/
"""

import argparse
import json
from pathlib import Path

from agent_definition.prompt_builder import build_prompt
from launcher.sampler import sample


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
    output_dir: Path,
    n: int,
    strategy: str = "stratified",
    seed: int = 42,
) -> list[Path]:
    """
    Generate agent directories from a sampled subset of role × interests × persona.
    Returns list of created agent directories.
    """
    samples = sample(roles, interests, personas, n=n, strategy=strategy, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    scaffolding_prompt = load(scaffolding)
    agent_dirs = []

    for i, agent in enumerate(samples):
        agent_dir = output_dir / f"agent_{i:03d}__{agent.name}"
        agent_dir.mkdir(exist_ok=True)

        system_prompt = build_prompt(
            role_prompt=load(agent.role),
            research_interests_prompt=load(agent.interests),
            persona_prompt=persona_to_prompt(agent.persona),
            scaffolding_prompt=scaffolding_prompt,
        )
        (agent_dir / "CLAUDE.md").write_text(system_prompt, encoding="utf-8")

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
    parser.add_argument("--n", type=int, required=True, help="Number of agents to sample")
    parser.add_argument("--strategy", default="stratified", choices=["random", "stratified"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="agent_configs/")
    args = parser.parse_args()

    prepare_agents(
        roles=args.roles,
        interests=args.interests,
        personas=args.personas,
        scaffolding=args.scaffolding,
        n=args.n,
        strategy=args.strategy,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )
