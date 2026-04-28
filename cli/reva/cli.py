"""reva CLI — reviewer agent command-line tool."""

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import click
from dotenv import load_dotenv

from reva.backends import BACKEND_CHOICES, get_backend
from reva.cluster import cancel_chain, list_cluster_jobs, submit_agent
from reva.config import DEFAULT_INITIAL_PROMPT, find_config, load_config, write_default_config
from reva.foreground import run_foreground
from reva.launch_script import write_launch_files
from reva.prompt import assemble_prompt
from reva.tmux import (
    build_launch_script,
    create_session,
    has_session,
    kill_all_sessions,
    kill_session,
    list_sessions,
)

def _load_project_env(config_path: str | None) -> None:
    """Load the project's `.env` so env-driven settings reach every subcommand."""
    found = find_config(config_path)
    project_root = found.parent if found is not None else Path.cwd()
    load_dotenv(project_root / ".env", override=False)


@click.group()
@click.option("--config", "config_path", default=None, help="Path to config.toml.")
@click.pass_context
def main(ctx, config_path):
    """reva — reviewer agent CLI."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path
    _load_project_env(config_path)


def _get_config(ctx):
    return load_config(ctx.obj.get("config_path"))


# --------------------------------------------------------------------------- #
# reva init
# --------------------------------------------------------------------------- #


@main.command()
@click.argument("path", default=".", type=click.Path())
@click.pass_context
def init(ctx, path):
    """Initialize a reva project (creates config.toml)."""
    target = Path(path).resolve()
    config_file = write_default_config(target)
    cfg = load_config(str(config_file))
    cfg.agents_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"Initialized reva project at {target}")
    click.echo(f"  config: {config_file}")


# --------------------------------------------------------------------------- #
# reva create
# --------------------------------------------------------------------------- #


@main.command()
@click.option("--name", required=True, help="Agent name (slug).")
@click.option(
    "--backend",
    type=click.Choice(BACKEND_CHOICES),
    default="claude-code",
    show_default=True,
    help="Agent backend.",
)
@click.pass_context
def create(ctx, name, backend):
    """Create a new agent directory with a starter system prompt."""
    cfg = _get_config(ctx)

    agent_dir = cfg.agents_dir / name
    if agent_dir.exists():
        raise click.ClickException(f"Agent directory already exists: {agent_dir}")
    agent_dir.mkdir(parents=True)

    starter_template = cfg.default_system_prompt_path.read_text(encoding="utf-8")
    (agent_dir / "system_prompt.md").write_text(
        starter_template.replace("{name}", name), encoding="utf-8"
    )

    config_data = {
        "name": name,
        "backend": backend,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (agent_dir / "config.json").write_text(
        json.dumps(config_data, indent=2), encoding="utf-8"
    )
    (agent_dir / ".agent_name").write_text(name, encoding="utf-8")

    click.echo(f"Created agent: {name}")
    click.echo(f"  directory: {agent_dir}")
    click.echo(f"  backend:   {backend}")
    click.echo(
        f"  next steps: edit {agent_dir / 'system_prompt.md'}, drop a key at "
        f"{agent_dir / '.api_key'}, then `reva launch --name {name}`"
    )


# --------------------------------------------------------------------------- #
# reva launch
# --------------------------------------------------------------------------- #


@main.command()
@click.option("--name", required=True, help="Agent name to launch.")
@click.option("--duration", type=float, default=None, help="Hours to run (omit for indefinite; ignored with --cluster).")
@click.option("--backend", type=click.Choice(BACKEND_CHOICES), default=None, help="Override backend.")
@click.option(
    "--session-timeout",
    type=int,
    default=600,
    help="Max seconds per invocation before restart (default: 600).",
)
@click.option(
    "--foreground",
    is_flag=True,
    help="Run directly in the current terminal instead of a tmux session. Windows-compatible.",
)
@click.option(
    "--fresh",
    is_flag=True,
    help=(
        "Clear saved session state (last_session_id) and force a new session. "
        "Use when a previous run left a stale session that cannot be resumed, "
        "e.g. 'No deferred tool marker found'. "
        "To clear manually: rm agent_configs/<name>/last_session_id"
    ),
)
@click.option(
    "--cluster",
    is_flag=True,
    help="Submit as a SLURM sbatch job instead of running in tmux.",
)
@click.option(
    "--partition",
    default="main-cpu",
    show_default=True,
    help="SLURM partition (--cluster only).",
)
@click.option(
    "--time",
    "time",
    default="5-00:00:00",
    show_default=True,
    help="SLURM wall time (--cluster only), e.g. 5-00:00:00.",
)
@click.option(
    "--cpus",
    type=int,
    default=4,
    show_default=True,
    help="SLURM --cpus-per-task (--cluster only).",
)
@click.option(
    "--mem",
    default="16G",
    show_default=True,
    help="SLURM --mem (--cluster only).",
)
@click.option(
    "--max-chain",
    type=int,
    default=3,
    show_default=True,
    help="Max sbatch jobs chained via the wall-time trap (--cluster only).",
)
@click.pass_context
def launch(ctx, name, duration, backend, session_timeout, foreground, fresh, cluster, partition, time, cpus, mem, max_chain):
    """Launch an agent in a tmux session (default), current terminal (--foreground), or SLURM job (--cluster)."""
    if foreground and cluster:
        raise click.ClickException("--foreground and --cluster are mutually exclusive.")

    cfg = _get_config(ctx)
    agent_dir = cfg.agents_dir / name
    if not agent_dir.exists():
        raise click.ClickException(f"Agent not found: {agent_dir}")

    api_key_path = agent_dir / ".api_key"
    if not api_key_path.exists() or not api_key_path.read_text(encoding="utf-8").strip():
        raise click.ClickException(
            f".api_key missing — ask the owner to provision it at "
            f"{cfg.koala_base_url}/owners and drop the key at {api_key_path}"
        )

    if fresh:
        session_id_file = agent_dir / "last_session_id"
        session_id_file.unlink(missing_ok=True)

    agent_config = json.loads((agent_dir / "config.json").read_text())
    backend_name = backend or agent_config["backend"]
    backend_obj = get_backend(backend_name)

    prompt = assemble_prompt(
        global_rules_path=cfg.global_rules_path,
        platform_skills_path=cfg.platform_skills_path,
        agent_prompt_path=agent_dir / "system_prompt.md",
    )
    (agent_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    (agent_dir / backend_obj.prompt_filename).write_text(prompt, encoding="utf-8")

    _override_path = agent_dir / "initial_prompt_override.txt"
    if _override_path.exists():
        initial_prompt = _override_path.read_text(encoding="utf-8").format(koala_base_url=cfg.koala_base_url)
    else:
        initial_prompt = DEFAULT_INITIAL_PROMPT.format(koala_base_url=cfg.koala_base_url)
    (agent_dir / "initial_prompt.txt").write_text(initial_prompt, encoding="utf-8")

    if name == "gsr_agent":
        _smoke_dir = agent_dir / "reviews" / "smoke_test"
        _smoke_dir.mkdir(parents=True, exist_ok=True)
        (_smoke_dir / "paper_smoke_test_summary.md").write_text(
            "\n".join([
                "# Smoke Artifact: startup verification",
                "",
                f"- **timestamp**: {datetime.now(timezone.utc).isoformat()}",
                f"- **agent_name**: {name}",
                "- **purpose**: local smoke artifact — verifies artifact creation path only",
                "- **koala_post**: no Koala post attempted",
            ]),
            encoding="utf-8",
        )
        _local_smoke_dir = agent_dir / "reviews" / "local_artifact_smoke"
        _local_smoke_dir.mkdir(parents=True, exist_ok=True)
        _now = datetime.now(timezone.utc)
        _sid = _now.strftime("%Y%m%dT%H%M%SZ")
        _ts = _now.isoformat()
        _placeholder = "Local artifact smoke test only; no Koala comment was attempted."
        (_local_smoke_dir / "paper_local_artifact_smoke_summary.md").write_text(
            "\n".join([
                "# GSR Artifact: paper_summary",
                "",
                "- **artifact_kind**: paper_summary",
                "- **paper_id**: local_artifact_smoke",
                f"- **created_at**: {_ts}",
                "- **source_phase**: review",
                "",
                "**Summary:** Local-only smoke artifact — no Koala comment was attempted.",
                "",
                "## Content",
                "",
                "Local-only smoke artifact. Verifies the realistic comment artifact format.",
                "",
                "- **local_only**: true",
                "- **karma_consumed**: none",
                f"- **agent_name**: {name}",
                "- **note**: no Koala post_comment or post_verdict was called",
            ]),
            encoding="utf-8",
        )
        (_local_smoke_dir / f"comment_draft_local_artifact_smoke_{_sid}.md").write_text(
            "\n".join([
                "# GSR Artifact: comment_draft",
                "",
                "- **artifact_kind**: comment_draft",
                "- **paper_id**: local_artifact_smoke",
                f"- **created_at**: {_ts}",
                "- **source_phase**: review",
                "",
                f"**Summary:** {_placeholder}",
                "",
                "## Content",
                "",
                _placeholder,
            ]),
            encoding="utf-8",
        )
        (_local_smoke_dir / f"comment_trace_local_artifact_smoke_{_sid}.json").write_text(
            json.dumps({
                "artifact_kind": "comment_trace",
                "paper_id": "local_artifact_smoke",
                "created_at": _ts,
                "source_phase": "review",
                "summary": _placeholder,
                "payload": {
                    "action_type": "local_smoke",
                    "paper_id": "local_artifact_smoke",
                    "safe_mode": True,
                    "local_only": True,
                    "note": "no post_comment called; no karma consumed",
                    "agent_name": name,
                },
            }, indent=2),
            encoding="utf-8",
        )

    escaped_prompt = initial_prompt.replace('"', '\\"')
    cmd = backend_obj.command_template.format(prompt=escaped_prompt)

    resume_cmd = (
        backend_obj.resume_command_template.format(prompt=escaped_prompt)
        if backend_obj.resume_command_template is not None
        else None
    )

    if name == "gsr_agent":
        _repo_root = str(cfg.project_root)
        cmd = (
            f'env PYTHONPATH="{_repo_root}:{_repo_root}/src"'
            " KOALA_RUN_MODE=live"
            ' KOALA_API_TOKEN="$KOALA_API_KEY"'
            " KOALA_ARTIFACT_MODE=github"
            ' KOALA_GITHUB_REPO="$KOALA_GITHUB_REPO"'
            " python -m gsr_agent.orchestration.operational_loop"
            f' --db "{_repo_root}/workspace/gsr_agent.db"'
            f' --out "{_repo_root}/workspace/reports"'
            " --live-reactive"
            " --live-verdict-auto"
            " 2>&1 | tee -a agent.log"
        )
        resume_cmd = None

    if cluster:
        script = build_launch_script(
            cmd,
            duration_hours=None,
            session_timeout=session_timeout,
            resume_command=resume_cmd,
            session_id_extractor=backend_obj.session_id_extractor,
        )
        write_launch_files(str(agent_dir), script)
        try:
            job_id = submit_agent(
                str(agent_dir),
                agent_name=name,
                partition=partition,
                time=time,
                cpus=cpus,
                mem=mem,
                max_chain=max_chain,
            )
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            raise click.ClickException(str(exc))
        click.echo(f"Submitted job {job_id} for agent {name} (chain 1/{max_chain})")
        click.echo(f"  SLURM job-name: reva_{name}")
        click.echo(f"  logs: {agent_dir / 'agent.log'}")
        return

    script = build_launch_script(
        cmd,
        duration_hours=duration,
        session_timeout=session_timeout,
        resume_command=resume_cmd,
        session_id_extractor=backend_obj.session_id_extractor,
    )

    if foreground:
        dur_str = f"{duration}h" if duration else "indefinite"
        click.echo(f"Launching: {name} ({backend_name}, {dur_str}, foreground)")
        click.echo("  Press Ctrl-C to stop.")
        try:
            run_foreground(name, str(agent_dir), script)
        except RuntimeError as exc:
            raise click.ClickException(str(exc))
        except KeyboardInterrupt:
            click.echo("\nStopped.")
        return

    create_session(name, str(agent_dir), script)

    dur_str = f"{duration}h" if duration else "indefinite"
    click.echo(f"Launched: {name} ({backend_name}, {dur_str})")
    click.echo(f"  tmux session: reva_{name}")
    click.echo(f"  attach: tmux attach -t reva_{name}")


# --------------------------------------------------------------------------- #
# reva stop
# --------------------------------------------------------------------------- #


@main.command()
@click.option("--name", default=None, help="Agent name to stop.")
@click.option("--all", "kill_all", is_flag=True, help="Stop all running agents.")
@click.option("--cluster", is_flag=True, help="Cancel SLURM chain(s) instead of tmux session(s).")
@click.pass_context
def stop(ctx, name, kill_all, cluster):
    """Stop a running agent (kill its tmux session or cancel its SLURM chain)."""
    if cluster:
        cfg = _get_config(ctx)
        if kill_all:
            jobs = list_cluster_jobs()
            agents = sorted({j.agent_name for j in jobs})
            total = 0
            for agent_name in agents:
                total += cancel_chain(agent_name=agent_name, agent_dir=str(cfg.agents_dir / agent_name))
            click.echo(f"Cancelled {total} SLURM job(s) across {len(agents)} agent(s).")
        elif name:
            count = cancel_chain(agent_name=name, agent_dir=str(cfg.agents_dir / name))
            click.echo(f"Cancelled {count} SLURM job(s) for: {name}")
        else:
            raise click.ClickException("Provide --name or --all.")
        return

    if kill_all:
        count = kill_all_sessions()
        click.echo(f"Stopped {count} agent(s).")
    elif name:
        if kill_session(name):
            click.echo(f"Stopped: {name}")
        else:
            click.echo(f"No running session for: {name}")
    else:
        raise click.ClickException("Provide --name or --all.")


# hidden alias so `reva kill` still works
_kill = click.Command(name="kill", callback=stop.callback, params=stop.params, help=stop.help, hidden=True)
main.add_command(_kill)


# --------------------------------------------------------------------------- #
# reva delete
# --------------------------------------------------------------------------- #


@main.command()
@click.argument("names", nargs=-1, required=True)
@click.option("--force", is_flag=True, help="Skip confirmation.")
@click.pass_context
def delete(ctx, names, force):
    """Remove agent directories (kills running sessions first)."""
    cfg = _get_config(ctx)
    for name in names:
        agent_dir = cfg.agents_dir / name
        if not agent_dir.exists():
            click.echo(f"Not found: {name}")
            continue
        if not force:
            click.confirm(f"Delete {agent_dir}?", abort=True)
        if has_session(name):
            kill_session(name)
        shutil.rmtree(agent_dir)
        click.echo(f"Deleted: {name}")


# --------------------------------------------------------------------------- #
# reva status
# --------------------------------------------------------------------------- #


@main.command()
@click.pass_context
def status(ctx):
    """List running agents (tmux sessions and SLURM cluster jobs)."""
    sessions = list_sessions()
    jobs = list_cluster_jobs()

    if not sessions and not jobs:
        click.echo("No running agents.")
        return

    cfg = _get_config(ctx)

    def _backend_for(agent_name: str) -> str:
        agent_config_path = cfg.agents_dir / agent_name / "config.json"
        if agent_config_path.exists():
            return json.loads(agent_config_path.read_text()).get("backend", "?")
        return "?"

    click.echo(f"{'NAME':<24s} {'BACKEND':<12s} {'MODE':<6s} {'JOB/SESSION':<20s} {'STATE'}")
    click.echo("-" * 78)
    for s in sessions:
        click.echo(
            f"{s.agent_name:<24s} {_backend_for(s.agent_name):<12s} "
            f"{'tmux':<6s} {s.session:<20s} RUNNING"
        )
    for j in jobs:
        click.echo(
            f"{j.agent_name:<24s} {_backend_for(j.agent_name):<12s} "
            f"{'slurm':<6s} {str(j.job_id):<20s} {j.state}"
        )


# --------------------------------------------------------------------------- #
# reva log
# --------------------------------------------------------------------------- #


@main.command(name="log")
@click.argument("name", required=False)
@click.option("--all", "watch_all", is_flag=True, help="Interleave all running agents.")
@click.pass_context
def log(ctx, name, watch_all):
    """Stream a readable live view of agent activity (ATIF-backed)."""
    from reva.render import render_step_terminal
    from reva.session import SessionContext

    cfg = _get_config(ctx)

    if watch_all:
        agents = sorted(d for d in cfg.agents_dir.iterdir() if d.is_dir() and (d / "agent.log").exists())
        if not agents:
            raise click.ClickException("No agent logs found.")
        log_files = [(a.name, a / "agent.log") for a in agents]
    elif name:
        agent_dir = cfg.agents_dir / name
        log_file = agent_dir / "agent.log"
        if not log_file.exists():
            raise click.ClickException(f"No agent.log found for: {name}")
        log_files = [(name, log_file)]
    else:
        agents = sorted(
            (d for d in cfg.agents_dir.iterdir() if d.is_dir() and (d / "agent.log").exists()),
            key=lambda d: (d / "agent.log").stat().st_mtime,
            reverse=True,
        )
        if not agents:
            raise click.ClickException("No agent logs found.")
        log_files = [(agents[0].name, agents[0] / "agent.log")]
        click.echo(f"Watching: {agents[0].name}\n")

    handles = {n: open(p, "r") for n, p in log_files}
    contexts = {n: SessionContext.for_agent(cfg.agents_dir / n) for n, _ in log_files}
    prefix = len(log_files) > 1

    try:
        while True:
            activity = False
            for agent_name, fh in handles.items():
                line = fh.readline()
                if not line:
                    continue
                activity = True
                sess = contexts[agent_name]
                for step in sess.consume_lines([line]):
                    for rendered in render_step_terminal(step, agent_name if prefix else None):
                        click.echo(rendered)
            if not activity:
                for agent_name, sess in contexts.items():
                    for step in sess.flush_pending():
                        for rendered in render_step_terminal(step, agent_name if prefix else None):
                            click.echo(rendered)
                    try:
                        sess.flush()
                    except Exception:
                        pass
                time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        for fh in handles.values():
            fh.close()
        for sess in contexts.values():
            try:
                sess.flush()
            except Exception:
                pass


# hidden alias so `reva watch` still works
_watch = click.Command(name="watch", callback=log.callback, params=log.params, help=log.help, hidden=True)
main.add_command(_watch)


# --------------------------------------------------------------------------- #
# reva view
# --------------------------------------------------------------------------- #


@main.command()
@click.option("--web", is_flag=True, help="Serve an interactive web UI instead of the TUI.")
@click.option("--host", default="127.0.0.1", show_default=True, help="Web host (with --web).")
@click.option("--port", default=8765, show_default=True, type=int, help="Web port (with --web).")
@click.pass_context
def view(ctx, web, host, port):
    """Launch the interactive ATIF viewer (TUI, or web with --web)."""
    cfg = _get_config(ctx)
    if web:
        from reva.web import serve

        serve(cfg, host=host, port=port)
        return
    from reva.viewer import RevaViewer

    app = RevaViewer(cfg=cfg)
    app.run()


# --------------------------------------------------------------------------- #
# reva archive / unarchive
# --------------------------------------------------------------------------- #


@main.command()
@click.option("--name", default=None, help="Agent name to archive.")
@click.option("--list", "list_archived", is_flag=True, help="List archived agents.")
@click.pass_context
def archive(ctx, name, list_archived):
    """Archive (retire) an agent by moving it to .archived/."""
    cfg = _get_config(ctx)
    archived_dir = cfg.agents_dir / ".archived"

    if list_archived:
        if not archived_dir.exists():
            click.echo("No archived agents.")
            return
        agents = sorted(
            d for d in archived_dir.iterdir()
            if d.is_dir() and (d / "config.json").exists()
        )
        if not agents:
            click.echo("No archived agents.")
            return
        for a in agents:
            click.echo(f"  {a.name}")
        click.echo(f"\n{len(agents)} archived agent(s)")
        return

    if not name:
        raise click.ClickException("Provide --name or --list.")

    agent_dir = cfg.agents_dir / name
    if not agent_dir.exists():
        raise click.ClickException(f"Agent not found: {name}")

    if has_session(name):
        kill_session(name)
        click.echo(f"Killed running session for: {name}")

    archived_dir.mkdir(parents=True, exist_ok=True)
    dest = archived_dir / name
    if dest.exists():
        raise click.ClickException(f"Already archived: {name}")
    shutil.move(str(agent_dir), str(dest))
    click.echo(f"Archived agent: {name}")


@main.command()
@click.option("--name", required=True, help="Agent name to unarchive.")
@click.pass_context
def unarchive(ctx, name):
    """Unarchive (restore) an agent from .archived/ back to agents_dir."""
    cfg = _get_config(ctx)
    archived_dir = cfg.agents_dir / ".archived"
    src = archived_dir / name
    if not src.exists():
        raise click.ClickException(f"Agent '{name}' is not archived.")

    dest = cfg.agents_dir / name
    if dest.exists():
        raise click.ClickException(f"Agent already exists at: {dest}")

    shutil.move(str(src), str(dest))
    click.echo(f"Unarchived agent: {name}")
