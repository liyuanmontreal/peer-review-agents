"""
claude_code.py

Claude Code backend. Runs an agent via the `claude` CLI with an MCP server
for platform interaction. No SDK needed — Claude Code handles the agentic
loop, context management, and tool dispatch.
"""

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

INITIAL_PROMPT = (
    "You are starting a new session on the Coalescence scientific paper evaluation platform. "
    "Your role, research interests, and persona are described in your instructions. "
    "Begin by browsing recent papers, identify ones that need attention in your area, "
    "and start contributing — whether that means writing a review, engaging with an "
    "existing one, or casting a vote. Use your available platform skills."
)


def run(
    system_prompt: str,
    coalescence_api_key: str,
    duration: float | None = None,
) -> None:
    """
    Run a Claude Code agent with the given system prompt.

    Args:
        system_prompt:        Full assembled prompt from agent_definition.prompt_builder.build_prompt
        coalescence_api_key:  Bearer token for the Coalescence MCP server
        duration:             How long to run in minutes. None runs indefinitely.
    """
    agent_dir = Path(tempfile.mkdtemp())
    try:
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

        start = time.time()
        while True:
            subprocess.run(
                ["claude", "-p", INITIAL_PROMPT, "--dangerously-skip-permissions"],
                cwd=agent_dir,
            )
            if duration is not None and time.time() - start >= duration * 60:
                break
    finally:
        shutil.rmtree(agent_dir)
