"""Sampling logic for batch agent creation.

Samples from the cartesian product of roles x interests x personas x
methodologies x formats. All five axes are required; callers must pass a
non-empty list for each. The CLI layer enforces this by populating each axis
from its corresponding config directory when no explicit glob is provided.
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
    methodology: str
    review_format: str

    @property
    def name(self) -> str:
        return "__".join(
            [
                Path(self.role).stem,
                Path(self.interests).stem,
                Path(self.persona).stem,
                Path(self.methodology).stem,
                Path(self.review_format).stem,
            ]
        )


def sample(
    roles: list[str],
    interests: list[str],
    personas: list[str],
    methodologies: list[str],
    formats: list[str],
    n: int,
    strategy: str = "stratified",
    seed: int = 42,
) -> list[AgentSample]:
    """Sample n agent configurations from the cartesian product of axes.

    Stratified strategy guarantees coverage of every role, persona,
    methodology, and format at least once. Interests are not stratified —
    their cardinality is usually too high to cover without exhausting n,
    so they are left to the random fill phase.
    """
    if strategy == "random":
        return _random_sample(roles, interests, personas, methodologies, formats, n, seed)
    elif strategy == "stratified":
        return _stratified_sample(roles, interests, personas, methodologies, formats, n, seed)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")


def _random_sample(
    roles: list[str],
    interests: list[str],
    personas: list[str],
    methodologies: list[str],
    formats: list[str],
    n: int,
    seed: int,
) -> list[AgentSample]:
    rng = random.Random(seed)
    pool = list(itertools.product(roles, interests, personas, methodologies, formats))
    if n >= len(pool):
        return [AgentSample(*combo) for combo in pool]
    chosen = rng.sample(pool, n)
    return [AgentSample(*combo) for combo in chosen]


def _stratified_sample(
    roles: list[str],
    interests: list[str],
    personas: list[str],
    methodologies: list[str],
    formats: list[str],
    n: int,
    seed: int,
) -> list[AgentSample]:
    """Ensure each role, persona, methodology, and format appears at least once.

    Interests are not stratified — their cardinality is usually too high to
    cover without exhausting n, so their coverage is left to the random fill
    phase.
    """
    rng = random.Random(seed)
    pool = list(itertools.product(roles, interests, personas, methodologies, formats))
    rng.shuffle(pool)

    selected: list[tuple] = []
    seen_roles: set[str] = set()
    seen_personas: set[str] = set()
    seen_methodologies: set[str] = set()
    seen_formats: set[str] = set()
    remainder: list[tuple] = []

    # first pass: pick combos that bring in a new stratified-axis value
    for combo in pool:
        role, _, persona, methodology, review_format = combo
        needed = (
            role not in seen_roles
            or persona not in seen_personas
            or methodology not in seen_methodologies
            or review_format not in seen_formats
        )
        if needed and len(selected) < n:
            selected.append(combo)
            seen_roles.add(role)
            seen_personas.add(persona)
            seen_methodologies.add(methodology)
            seen_formats.add(review_format)
        else:
            remainder.append(combo)

    # fill up to n from remainder
    rng.shuffle(remainder)
    for combo in remainder:
        if len(selected) >= n:
            break
        selected.append(combo)

    return [AgentSample(*combo) for combo in selected[:n]]
