"""
foreground.py

Foreground (in-terminal) agent runner — Windows-compatible alternative to
tmux. Runs the generated launch script directly in the current terminal
process instead of creating a detached tmux session.

Use via `reva launch --foreground`.
"""
import shutil
import subprocess
from pathlib import Path

from reva.launch_script import write_launch_files


def run_foreground(agent_name: str, working_dir: str, launch_script: str) -> None:
    """Run the agent launch script in the current terminal.

    Writes the same .reva_launch.sh that create_session() would write, then
    executes it directly with bash. Blocks until the script exits or the user
    presses Ctrl-C.

    Args:
        agent_name: Agent name (unused at runtime; kept for API parity with
            create_session so callers can swap them without changes).
        working_dir: Absolute or relative path to the agent directory.
            The script runs with this as its working directory.
        launch_script: Bash script content (produced by build_launch_script()).

    Raises:
        RuntimeError: If bash cannot be found on the current PATH.
    """
    bash = shutil.which("bash")
    if bash is None:
        raise RuntimeError(
            "bash not found on PATH. "
            "On Windows, install Git for Windows (https://git-scm.com) — "
            "it bundles bash and is required for the --foreground launcher. "
            "Alternatively, enable WSL and run from a WSL terminal."
        )

    working_dir = str(Path(working_dir).resolve())
    script_path = write_launch_files(working_dir, launch_script)

    subprocess.run([bash, str(script_path)], cwd=working_dir)
