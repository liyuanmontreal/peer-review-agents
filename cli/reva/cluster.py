"""SLURM cluster submission for reva agents.

Submits agent launches as sbatch jobs with a self-chaining trap: when a job
hits wall time, SLURM sends SIGTERM, the EXIT trap fires, and a successor
job is queued (unless a .reva_stop sentinel is present or the max-chain cap
has been reached). See .claude/specs/cluster-slurm-support.md for the
design rationale.
"""

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from reva.launch_script import LAUNCH_FILENAME, write_launch_files
from reva.tmux import SESSION_PREFIX

SBATCH_FILENAME = ".reva_cluster.sbatch"
STOP_SENTINEL = ".reva_stop"

JOB_NAME_PREFIX = SESSION_PREFIX  # "reva_" — shared with tmux sessions for symmetry

_TIME_RE = re.compile(r"^(\d+-)?\d{1,2}:\d{2}(:\d{2})?$")

_SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --output={agent_dir}/cluster.%j.out
#SBATCH --error={agent_dir}/cluster.%j.err

cd {agent_dir}
# Only the first link in the chain wipes a stale sentinel from a prior run.
# Successor jobs (REVA_CHAIN_N set via --export) must preserve it so that
# cancel_chain's "write sentinel, then scancel" sequence still aborts an
# already-pending successor whose dependency may fire before scancel lands.
if [ -z "$REVA_CHAIN_N" ]; then
    rm -f {stop_sentinel}
fi

_chain_next() {{
    if [ -f {stop_sentinel} ]; then
        echo "[reva cluster] stop sentinel present, not chaining."
        return
    fi
    if [ "$REVA_CHAIN_N" -ge "$REVA_MAX_CHAIN" ]; then
        echo "[reva cluster] chain limit reached (${{REVA_CHAIN_N}}/${{REVA_MAX_CHAIN}}), not chaining."
        return
    fi
    NEXT=$((REVA_CHAIN_N + 1))
    sbatch --dependency=afterany:$SLURM_JOB_ID \\
        --export=ALL,REVA_CHAIN_N=$NEXT,REVA_MAX_CHAIN=$REVA_MAX_CHAIN \\
        {agent_dir}/{sbatch_filename}
}}
trap _chain_next EXIT

REVA_CHAIN_N=${{REVA_CHAIN_N:-1}}
REVA_MAX_CHAIN=${{REVA_MAX_CHAIN:-{max_chain}}}
export REVA_CHAIN_N REVA_MAX_CHAIN

bash {agent_dir}/{launch_filename}
"""


@dataclass(frozen=True)
class ClusterJob:
    agent_name: str
    job_id: int
    state: str
    time_left: str


def _validate_time(time: str) -> None:
    if not _TIME_RE.match(time):
        raise ValueError(
            f"invalid --time {time!r}: expected SLURM format like "
            "'5-00:00:00', '12:30:00', or '02:30'"
        )


def job_name(agent_name: str) -> str:
    return f"{JOB_NAME_PREFIX}{agent_name}"


def submit_agent(
    agent_dir: str,
    *,
    agent_name: str,
    partition: str,
    time: str,
    cpus: int,
    mem: str,
    max_chain: int,
) -> int:
    """Submit *agent_name* as a SLURM job. Returns the sbatch job ID."""
    if shutil.which("sbatch") is None:
        raise RuntimeError("SLURM not available on this host (sbatch not on $PATH)")

    _validate_time(time)

    working_dir = Path(agent_dir).resolve()
    launch_path = working_dir / LAUNCH_FILENAME
    if not launch_path.exists():
        raise FileNotFoundError(
            f".reva_launch.sh not found at {launch_path}; submit_agent requires "
            "write_launch_files to have been called first"
        )

    sbatch_script = _SBATCH_TEMPLATE.format(
        job_name=job_name(agent_name),
        partition=partition,
        time=time,
        cpus=cpus,
        mem=mem,
        agent_dir=str(working_dir),
        launch_filename=LAUNCH_FILENAME,
        sbatch_filename=SBATCH_FILENAME,
        stop_sentinel=STOP_SENTINEL,
        max_chain=max_chain,
    )
    sbatch_path = working_dir / SBATCH_FILENAME
    sbatch_path.write_text(sbatch_script, encoding="utf-8")
    sbatch_path.chmod(0o755)

    result = subprocess.run(
        ["sbatch", str(sbatch_path)],
        cwd=str(working_dir),
        capture_output=True,
        text=True,
        check=True,
    )
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        raise RuntimeError(
            f"sbatch did not return a job ID. stdout={result.stdout!r} stderr={result.stderr!r}"
        )
    return int(match.group(1))


def list_cluster_jobs() -> list[ClusterJob]:
    """Return all the user's reva_* SLURM jobs (running and pending)."""
    if shutil.which("squeue") is None:
        return []

    # squeue's --name takes exact names (comma-separated), not globs;
    # passing --name=reva_* silently returns zero rows. List all of the
    # user's jobs and filter by prefix client-side (same pattern as
    # tmux.list_sessions).
    result = subprocess.run(
        ["squeue", "-u", _whoami(), "-h", "-o", "%A|%j|%T|%L"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []

    jobs: list[ClusterJob] = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) != 4:
            continue
        job_id_s, name, state, time_left = parts
        if not name.startswith(JOB_NAME_PREFIX):
            continue
        agent_name = name[len(JOB_NAME_PREFIX):]
        jobs.append(ClusterJob(
            agent_name=agent_name,
            job_id=int(job_id_s),
            state=state,
            time_left=time_left,
        ))
    return jobs


def cancel_chain(*, agent_name: str, agent_dir: str) -> int:
    """Cancel every SLURM job in *agent_name*'s chain and block future successors.

    Writes .reva_stop into the agent directory before invoking scancel so that
    a pending successor job whose dependency fires before scancel reaches it
    will still see the sentinel and abort chaining.

    Returns the number of jobs cancelled.
    """
    working_dir = Path(agent_dir).resolve()
    working_dir.mkdir(parents=True, exist_ok=True)
    sentinel = working_dir / STOP_SENTINEL
    sentinel.write_text("", encoding="utf-8")

    if shutil.which("squeue") is None:
        return 0

    result = subprocess.run(
        ["squeue", "-u", _whoami(), "--name=" + job_name(agent_name),
         "-h", "-o", "%A|%j|%T|%L"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return 0

    job_ids: list[str] = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("|")
        if parts and parts[0].strip():
            job_ids.append(parts[0].strip())

    if not job_ids:
        return 0

    subprocess.run(
        ["scancel"] + job_ids,
        capture_output=True,
        text=True,
        check=False,
    )
    return len(job_ids)


def _whoami() -> str:
    import os
    return os.environ.get("USER") or os.environ.get("LOGNAME") or ""
