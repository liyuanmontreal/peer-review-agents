"""
run_agents.py

Launches multiple Claude Code agents in parallel, each running against the
Moltbook platform. Each agent has its own directory containing a CLAUDE.md
(assembled by agent_definition/prompt_builder.py) and an .mcp.json config
pointing at the Moltbook MCP server.

Usage:
    python run_agents.py --agent-dirs agents/agent_0 agents/agent_1 ...
    python run_agents.py --agent-dirs agents/agent_0 agents/agent_1 --duration 3600
"""

import argparse
import threading
from pathlib import Path

from launcher.backends.claude_code import run as run_agent_backend


def run_agent(agent_dir: Path, duration: float | None) -> None:
    """Run a single agent via the Claude Code backend."""
    run_agent_backend(
        system_prompt=(agent_dir / "CLAUDE.md").read_text(encoding="utf-8"),
        mcp_config=str(agent_dir / ".mcp.json"),
        duration=duration,
    )


def launch_agents(agent_dirs: list[Path], duration: float | None) -> None:
    """Launch all agents in parallel and wait for all to finish."""
    threads = [
        threading.Thread(target=run_agent, args=(agent_dir, duration))
        for agent_dir in agent_dirs
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent-dirs", nargs="+", required=True,
        help="Paths to agent directories (each must contain a CLAUDE.md and .mcp.json)",
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="How long to run each agent in minutes (omit to run indefinitely)",
    )
    args = parser.parse_args()

    agent_dirs = [Path(d) for d in args.agent_dirs]
    launch_agents(agent_dirs, args.duration)
