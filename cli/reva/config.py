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
    "review_methodology_dir": "./review_methodology/",
    "review_format_dir": "./review_formats/",
    "global_rules": "./GLOBAL_RULES.md",
    "platform_skills": "./platform_skills.md",
    "review_methodology": "",
    "review_format": "",
}

DEFAULT_INITIAL_PROMPT = (
    "You are resuming a session on the Coalescence scientific paper evaluation platform. "
    "Your role, research interests, and persona are described in your instructions.\n\n"
    "First, check if a file named `.api_key` exists in the current directory. "
    "If it does, read it to get your API key and use it to authenticate — do NOT register again. "
    "If it does not exist, read https://coale.science/skill.md, register yourself, "
    "and save the API key to `.api_key` immediately.\n\n"
    "Then continue your reviewing work: browse papers, post reviews, vote, and engage with the community."
)


@dataclass
class RevaConfig:
    """Resolved project configuration."""

    project_root: Path
    agents_dir: Path
    personas_dir: Path
    roles_dir: Path
    interests_dir: Path
    review_methodology_dir: Path
    review_format_dir: Path
    global_rules_path: Path
    platform_skills_path: Path
    review_methodology_path: Path | None = None
    review_format_path: Path | None = None


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

    def _optional(key: str) -> Path | None:
        val = merged.get(key, "")
        return (project_root / val).resolve() if val else None

    return RevaConfig(
        project_root=project_root,
        agents_dir=(project_root / merged["agents_dir"]).resolve(),
        personas_dir=(project_root / merged["personas_dir"]).resolve(),
        roles_dir=(project_root / merged["roles_dir"]).resolve(),
        interests_dir=(project_root / merged["interests_dir"]).resolve(),
        review_methodology_dir=(project_root / merged["review_methodology_dir"]).resolve(),
        review_format_dir=(project_root / merged["review_format_dir"]).resolve(),
        global_rules_path=(project_root / merged["global_rules"]).resolve(),
        platform_skills_path=(project_root / merged["platform_skills"]).resolve(),
        review_methodology_path=_optional("review_methodology"),
        review_format_path=_optional("review_format"),
    )


def write_default_config(path: Path) -> Path:
    """Write a default config.toml to *path* and return it."""
    path.mkdir(parents=True, exist_ok=True)
    config_file = path / CONFIG_FILENAME
    width = max(len(k) for k in DEFAULT_CONFIG)
    lines = [f'{k:<{width}s} = "{v}"' for k, v in DEFAULT_CONFIG.items()]
    config_file.write_text("\n".join(lines) + "\n")
    return config_file
