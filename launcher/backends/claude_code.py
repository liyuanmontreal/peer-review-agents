"""
claude_code.py

Claude Code backend. Runs an agent via the `claude` CLI.
Agents self-register on the Coalescence platform at runtime by following
the instructions at https://coale.science/skill.md.

Paper Lantern (research intelligence MCP server) is attached at startup
if PAPER_LANTERN_API_KEY is set in the environment.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

INITIAL_PROMPT = (
    "You are starting a new session on the Coalescence scientific paper evaluation platform. "
    "Your role, research interests, and persona are described in your instructions. "
    "Start by reading https://coale.science/skill.md and following the instructions to "
    "register yourself and get started."
)


def run(
    system_prompt: str,
    duration: float | None = None,
) -> None:
    """
    Run a Claude Code agent with the given system prompt.

    Args:
        system_prompt: Full assembled prompt from agent_definition.prompt_builder.build_prompt
        duration:      How long to run in minutes. None runs indefinitely.
    """
    agent_dir = Path(tempfile.mkdtemp())
    try:
        (agent_dir / "CLAUDE.md").write_text(system_prompt, encoding="utf-8")

        pl_key = os.environ.get("PAPER_LANTERN_API_KEY")
        if pl_key:
            claude_dir = agent_dir / ".claude"
            claude_dir.mkdir()
            settings = {
                "mcpServers": {
                    "paperlantern": {
                        "type": "url",
                        "url": f"https://mcp.paperlantern.ai/chat/mcp?key={pl_key}",
                    }
                }
            }
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
