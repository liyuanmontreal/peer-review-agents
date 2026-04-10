"""Sampling logic for batch agent creation.

Samples from the cartesian product of roles x interests x personas.
"""

import itertools
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgentSample:
    role: str
    interests: str
    persona: str

    @property
    def name(self) -> str:
        r = Path(self.role).stem
        i = Path(self.interests).stem
        p = Path(self.persona).stem
        return f"{r}__{i}__{p}"


def sample(
    roles: list[str],
    interests: list[str],
    personas: list[str],
    n: int,
    strategy: str = "stratified",
    seed: int = 42,
) -> list[AgentSample]:
    """Sample n agent configurations from the cartesian product."""
    if strategy == "random":
        return _random_sample(roles, interests, personas, n, seed)
    elif strategy == "stratified":
        return _stratified_sample(roles, interests, personas, n, seed)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")


def _random_sample(
    roles: list[str],
    interests: list[str],
    personas: list[str],
    n: int,
    seed: int,
) -> list[AgentSample]:
    rng = random.Random(seed)
    pool = list(itertools.product(roles, interests, personas))
    if n >= len(pool):
        return [AgentSample(*combo) for combo in pool]
    chosen = rng.sample(pool, n)
    return [AgentSample(*combo) for combo in chosen]


def _stratified_sample(
    roles: list[str],
    interests: list[str],
    personas: list[str],
    n: int,
    seed: int,
) -> list[AgentSample]:
    """Ensure each role and persona appears at least once, fill remainder randomly."""
    rng = random.Random(seed)
    pool = list(itertools.product(roles, interests, personas))
    rng.shuffle(pool)

    selected: list[tuple[str, str, str]] = []
    seen_roles: set[str] = set()
    seen_personas: set[str] = set()
    remainder: list[tuple[str, str, str]] = []

    # first pass: pick one combo per role and per persona
    for combo in pool:
        role, _, persona = combo
        needed = role not in seen_roles or persona not in seen_personas
        if needed and len(selected) < n:
            selected.append(combo)
            seen_roles.add(role)
            seen_personas.add(persona)
        else:
            remainder.append(combo)

    # fill up to n from remainder
    rng.shuffle(remainder)
    for combo in remainder:
        if len(selected) >= n:
            break
        selected.append(combo)

    return [AgentSample(*combo) for combo in selected[:n]]
