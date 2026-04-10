#!/usr/bin/env python3
"""
Run a single agent on the Coalescence platform.
The agent will self-register on the platform at runtime.

Usage:
    python run_agent.py \
        --role agent_definition/roles/01_novelty_and_originality.md \
        --interests agent_definition/research_interests/nlp.md \
        --persona agent_definition/personas/optimistic.json \
        --scaffolding agent_definition/harness/scaffolding.md \
        [--duration 60]
"""

import argparse
import sys
from pathlib import Path

from agent_definition.prompt_builder import build_prompt
from launcher.prepare_agents import persona_to_prompt


def load(path: str) -> str:
    p = Path(path)
    if not p.exists():
        sys.exit(f"File not found: {path}")
    return p.read_text(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True)
    parser.add_argument("--interests", required=True)
    parser.add_argument("--persona", required=True)
    parser.add_argument("--scaffolding", required=True)
    parser.add_argument("--review-methodology", default="",
                        help="Path to review methodology .md file")
    parser.add_argument("--review-format", default="",
                        help="Path to review format .md file")
    parser.add_argument("--duration", type=float, default=None,
                        help="How long to run in minutes (omit to run indefinitely)")
    parser.add_argument("--backend", default="claude_code", choices=["claude_code"])
    args = parser.parse_args()

    review_methodology = load(args.review_methodology) if args.review_methodology else ""
    review_format = load(args.review_format) if args.review_format else ""

    system_prompt = build_prompt(
        role_prompt=load(args.role),
        research_interests_prompt=load(args.interests),
        persona_prompt=persona_to_prompt(args.persona),
        scaffolding_prompt=load(args.scaffolding),
        review_methodology_prompt=review_methodology,
        review_format_prompt=review_format,
    )

    if args.backend == "claude_code":
        from launcher.backends.claude_code import run
        run(system_prompt, duration=args.duration)


if __name__ == "__main__":
    main()
