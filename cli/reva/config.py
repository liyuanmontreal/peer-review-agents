"""Config resolution for reva projects.

Resolution order (first match wins):
  1. --config flag (passed via click context)
  2. REVA_CONFIG env var
  3. Walk up from cwd looking for config.toml
  4. ~/.reva/config.toml (global default)
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

CONFIG_FILENAME = "config.toml"

DEFAULT_CONFIG = {
    "agents_dir": "./agents/",
    "personas_dir": "./personas/",
    "roles_dir": "./roles/",
    "interests_dir": "./interests/",
}

DEFAULT_INITIAL_PROMPT = (
    "You are starting a new session on the Coalescence scientific paper "
    "evaluation platform. Your role, research interests, and persona are "
    "described in your instructions. Start by reading "
    "https://coale.science/skill.md and following the instructions to "
    "register yourself and get started."
)


@dataclass
class RevaConfig:
    """Resolved project configuration."""

    project_root: Path
    agents_dir: Path
    personas_dir: Path
    roles_dir: Path
    interests_dir: Path

    @property
    def global_rules_path(self) -> Path:
        return self.project_root / "GLOBAL_RULES.md"

    @property
    def platform_skills_path(self) -> Path:
        return self.project_root / "platform_skills.md"


def _walk_up(start: Path) -> Path | None:
    """Walk up from *start* looking for config.toml."""
    current = start.resolve()
    while True:
        candidate = current / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            return None
        current = parent


def find_config(explicit: str | None = None) -> Path | None:
    """Find config.toml using the resolution order."""
    # 1. explicit --config flag
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        return None

    # 2. REVA_CONFIG env var
    env = os.environ.get("REVA_CONFIG")
    if env:
        p = Path(env)
        if p.is_file():
            return p

    # 3. walk up from cwd
    found = _walk_up(Path.cwd())
    if found:
        return found

    # 4. global default
    global_default = Path.home() / ".reva" / CONFIG_FILENAME
    if global_default.is_file():
        return global_default

    return None


def load_config(explicit: str | None = None) -> RevaConfig:
    """Load and resolve config, falling back to defaults."""
    config_path = find_config(explicit)

    if config_path is not None:
        with open(config_path, "rb") as f:
            raw = tomllib.load(f)
        project_root = config_path.parent
    else:
        raw = {}
        project_root = Path.cwd()

    merged = {**DEFAULT_CONFIG, **raw}

    return RevaConfig(
        project_root=project_root,
        agents_dir=(project_root / merged["agents_dir"]).resolve(),
        personas_dir=(project_root / merged["personas_dir"]).resolve(),
        roles_dir=(project_root / merged["roles_dir"]).resolve(),
        interests_dir=(project_root / merged["interests_dir"]).resolve(),
    )


def write_default_config(path: Path) -> Path:
    """Write a default config.toml to *path* and return it."""
    path.mkdir(parents=True, exist_ok=True)
    config_file = path / CONFIG_FILENAME
    lines = [f'{k:<14s} = "{v}"' for k, v in DEFAULT_CONFIG.items()]
    config_file.write_text("\n".join(lines) + "\n")
    return config_file
