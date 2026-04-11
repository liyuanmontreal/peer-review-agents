"""reva CLI — reviewer agent command-line tool."""

import glob
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click

from reva.backends import BACKEND_CHOICES, get_backend
from reva.compiler import compile_agent_prompt, persona_to_markdown
from reva.config import DEFAULT_INITIAL_PROMPT, load_config, write_default_config
from reva.sampler import sample
from reva.tmux import (
    build_launch_script,
    create_session,
    has_session,
    kill_all_sessions,
    kill_session,
    list_sessions,
)


pass_config_path = click.make_pass_decorator(str, ensure=True)


@click.group()
@click.option("--config", "config_path", default=None, help="Path to config.toml.")
@click.pass_context
def main(ctx, config_path):
    """reva — reviewer agent CLI."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path


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
    # create default subdirectories
    cfg = load_config(str(config_file))
    for d in [cfg.agents_dir, cfg.personas_dir, cfg.roles_dir, cfg.interests_dir]:
        d.mkdir(parents=True, exist_ok=True)
    click.echo(f"Initialized reva project at {target}")
    click.echo(f"  config: {config_file}")


# --------------------------------------------------------------------------- #
# reva create
# --------------------------------------------------------------------------- #


@main.command()
@click.option("--name", required=True, help="Agent name (slug).")
@click.option("--backend", required=True, type=click.Choice(BACKEND_CHOICES), help="Agent backend.")
@click.option("--role", required=True, type=click.Path(exists=True), help="Path to role .md file.")
@click.option("--persona", required=True, type=click.Path(exists=True), help="Path to persona .json file.")
@click.option("--interest", required=True, type=click.Path(exists=True), help="Path to research interest .md file.")
@click.pass_context
def create(ctx, name, backend, role, persona, interest):
    """Create a new agent directory with compiled prompt."""
    cfg = _get_config(ctx)
    backend_obj = get_backend(backend)

    agent_dir = cfg.agents_dir / name
    if agent_dir.exists():
        raise click.ClickException(f"Agent directory already exists: {agent_dir}")
    agent_dir.mkdir(parents=True)

    # compile prompt
    prompt = compile_agent_prompt(
        role_path=Path(role),
        persona_path=Path(persona),
        interest_path=Path(interest),
        global_rules_path=cfg.global_rules_path,
        platform_skills_path=cfg.platform_skills_path,
    )

    # write files
    (agent_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    (agent_dir / backend_obj.prompt_filename).write_text(prompt, encoding="utf-8")
    (agent_dir / "initial_prompt.txt").write_text(DEFAULT_INITIAL_PROMPT, encoding="utf-8")

    config_data = {
        "name": name,
        "backend": backend,
        "role": str(Path(role).resolve()),
        "persona": str(Path(persona).resolve()),
        "interest": str(Path(interest).resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (agent_dir / "config.json").write_text(json.dumps(config_data, indent=2), encoding="utf-8")

    click.echo(f"Created agent: {name}")
    click.echo(f"  directory: {agent_dir}")
    click.echo(f"  backend:   {backend}")
    click.echo(f"  prompt:    {len(prompt)} chars")


# --------------------------------------------------------------------------- #
# reva launch
# --------------------------------------------------------------------------- #


@main.command()
@click.option("--name", required=True, help="Agent name to launch.")
@click.option("--duration", type=float, default=None, help="Hours to run (omit for indefinite).")
@click.option("--backend", type=click.Choice(BACKEND_CHOICES), default=None, help="Override backend.")
@click.option("--session-timeout", type=int, default=600, help="Max seconds per invocation before restart (default: 600).")
@click.pass_context
def launch(ctx, name, duration, backend, session_timeout):
    """Launch an agent in a tmux session."""
    cfg = _get_config(ctx)
    agent_dir = cfg.agents_dir / name
    if not agent_dir.exists():
        raise click.ClickException(f"Agent not found: {agent_dir}")

    agent_config = json.loads((agent_dir / "config.json").read_text())
    backend_name = backend or agent_config["backend"]
    backend_obj = get_backend(backend_name)

    # ensure backend-specific prompt file exists
    prompt_file = agent_dir / backend_obj.prompt_filename
    if not prompt_file.exists():
        prompt = (agent_dir / "prompt.md").read_text(encoding="utf-8")
        prompt_file.write_text(prompt, encoding="utf-8")

    initial_prompt = (agent_dir / "initial_prompt.txt").read_text(encoding="utf-8").strip()
    # escape double quotes for shell embedding
    escaped_prompt = initial_prompt.replace('"', '\\"')
    cmd = backend_obj.command_template.format(prompt=escaped_prompt)

    script = build_launch_script(cmd, duration_hours=duration, session_timeout=session_timeout)
    create_session(name, str(agent_dir), script)

    dur_str = f"{duration}h" if duration else "indefinite"
    click.echo(f"Launched: {name} ({backend_name}, {dur_str})")
    click.echo(f"  tmux session: reva_{name}")
    click.echo(f"  attach: tmux attach -t reva_{name}")


# --------------------------------------------------------------------------- #
# reva kill
# --------------------------------------------------------------------------- #


@main.command()
@click.option("--name", default=None, help="Agent name to stop.")
@click.option("--all", "kill_all", is_flag=True, help="Stop all running agents.")
@click.pass_context
def kill(ctx, name, kill_all):
    """Stop a running agent (kill its tmux session)."""
    if kill_all:
        count = kill_all_sessions()
        click.echo(f"Killed {count} agent(s).")
    elif name:
        if kill_session(name):
            click.echo(f"Killed: {name}")
        else:
            click.echo(f"No running session for: {name}")
    else:
        raise click.ClickException("Provide --name or --all.")


# --------------------------------------------------------------------------- #
# reva status
# --------------------------------------------------------------------------- #


@main.command()
@click.pass_context
def status(ctx):
    """List running agents."""
    sessions = list_sessions()
    if not sessions:
        click.echo("No running agents.")
        return

    # load backend from config.json where available
    cfg = _get_config(ctx)
    click.echo(f"{'NAME':<30s} {'BACKEND':<15s} {'SESSION'}")
    click.echo("-" * 70)
    for s in sessions:
        backend_name = "?"
        agent_config_path = cfg.agents_dir / s.agent_name / "config.json"
        if agent_config_path.exists():
            agent_config = json.loads(agent_config_path.read_text())
            backend_name = agent_config.get("backend", "?")
        click.echo(f"{s.agent_name:<30s} {backend_name:<15s} {s.session}")


# --------------------------------------------------------------------------- #
# reva list
# --------------------------------------------------------------------------- #


@main.command("list")
@click.argument("component", type=click.Choice(["roles", "personas", "interests", "agents"]))
@click.pass_context
def list_components(ctx, component):
    """List available components (roles, personas, interests, agents)."""
    cfg = _get_config(ctx)

    if component == "roles":
        _list_files(cfg.roles_dir, "*.md")
    elif component == "personas":
        _list_files(cfg.personas_dir, "*.json")
    elif component == "interests":
        _list_files(cfg.interests_dir, "**/*.md")
    elif component == "agents":
        _list_agents(cfg.agents_dir)


def _list_files(directory: Path, pattern: str):
    if not directory.exists():
        click.echo(f"Directory not found: {directory}")
        return
    files = sorted(directory.glob(pattern))
    files = [f for f in files if f.name != "README.md"]
    if not files:
        click.echo("  (none)")
        return
    for f in files:
        rel = f.relative_to(directory)
        click.echo(f"  {rel}")
    click.echo(f"\n{len(files)} file(s)")


def _list_agents(directory: Path):
    if not directory.exists():
        click.echo(f"Directory not found: {directory}")
        return
    agents = sorted(d for d in directory.iterdir() if d.is_dir() and (d / "config.json").exists())
    if not agents:
        click.echo("  (none)")
        return
    for a in agents:
        config = json.loads((a / "config.json").read_text())
        running = "running" if has_session(a.name) else "stopped"
        click.echo(f"  {a.name:<30s} {config.get('backend', '?'):<15s} {running}")
    click.echo(f"\n{len(agents)} agent(s)")


# --------------------------------------------------------------------------- #
# reva persona
# --------------------------------------------------------------------------- #


@main.group()
def persona():
    """Manage persona profiles."""
    pass


main.add_command(persona)


@persona.command("list")
@click.pass_context
def persona_list(ctx):
    """List available personas."""
    cfg = _get_config(ctx)
    _list_files(cfg.personas_dir, "*.json")


@persona.command("show")
@click.argument("name")
@click.pass_context
def persona_show(ctx, name):
    """Show a persona's details."""
    cfg = _get_config(ctx)
    path = cfg.personas_dir / f"{name}.json"
    if not path.exists():
        raise click.ClickException(f"Persona not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    click.echo(f"Name: {data['name']}")
    click.echo(f"Description: {data.get('description', '')}\n")

    click.echo("Trait vector:")
    for trait, value in data.get("trait_vector", {}).items():
        label = {-1: "Low", 0: "Neutral", 1: "High"}.get(value, str(value))
        click.echo(f"  {trait:<20s} {label}")

    rules = data.get("behavioral_rules", [])
    if rules:
        click.echo("\nBehavioral rules:")
        for r in rules:
            click.echo(f"  - {r}")

    forbidden = data.get("forbidden_behaviors", [])
    if forbidden:
        click.echo("\nForbidden:")
        for r in forbidden:
            click.echo(f"  - {r}")


# --------------------------------------------------------------------------- #
# reva interests
# --------------------------------------------------------------------------- #


@main.group()
def interests():
    """Manage research interest profiles."""
    pass


@interests.command("list-topics")
@click.option("--depth", type=int, multiple=True, default=None, help="Tree depths to show.")
@click.pass_context
def interests_list_topics(ctx, depth):
    """List taxonomy topics."""
    cfg = _get_config(ctx)
    taxonomy_path = cfg.interests_dir / "ml_taxonomy.json"
    if not taxonomy_path.exists():
        raise click.ClickException(f"Taxonomy not found: {taxonomy_path}")

    data = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    nodes = _walk_taxonomy(data)

    if depth:
        nodes = [n for n in nodes if n["depth"] in depth]
    # exclude root
    nodes = [n for n in nodes if n["depth"] > 0]

    for i, node in enumerate(nodes, 1):
        tag = "leaf" if node["is_leaf"] else f"parent, {len(node['children'])} children"
        click.echo(f"  {i:3d}. [depth={node['depth']}] {node['path']}  ({tag})")

    click.echo(f"\n{len(nodes)} topic(s)")


def _walk_taxonomy(node, parent_path="", depth=0):
    path = f"{parent_path} > {node['name']}" if parent_path else node["name"]
    children = node.get("children", [])
    entry = {
        "name": node["name"],
        "path": path,
        "depth": depth,
        "is_leaf": len(children) == 0,
        "children": [c["name"] for c in children],
    }
    results = [entry]
    for child in children:
        results.extend(_walk_taxonomy(child, path, depth + 1))
    return results


@interests.command("generate")
@click.option("--depth", type=int, multiple=True, help="Tree depths to generate for.")
@click.option("--levels", multiple=True, help="Expertise levels (senior, mid, junior, adjacent).")
@click.option("--dry-run", is_flag=True, help="Preview without calling API.")
@click.pass_context
def interests_generate(ctx, depth, levels, dry_run):
    """Generate interest profiles via LLM (placeholder)."""
    click.echo("Interest generation is not yet implemented in reva.")
    click.echo("Use the standalone generate_personas.py script for now.")


@interests.command("validate")
@click.pass_context
def interests_validate(ctx):
    """Validate generated interest profiles (placeholder)."""
    click.echo("Interest validation is not yet implemented in reva.")


# --------------------------------------------------------------------------- #
# reva batch
# --------------------------------------------------------------------------- #


@main.group()
def batch():
    """Batch operations on multiple agents."""
    pass


@batch.command("create")
@click.option("--roles", multiple=True, help="Role .md file globs (default: roles_dir/*.md).")
@click.option("--interests", "interest_globs", multiple=True, help="Interest .md file globs (default: interests_dir/**/*.md).")
@click.option("--personas", multiple=True, help="Persona .json file globs (default: personas_dir/*.json).")
@click.option("--n", "count", type=int, default=1, help="Number of agents to sample (default: 1).")
@click.option("--strategy", type=click.Choice(["stratified", "random"]), default="random")
@click.option("--seed", type=int, default=42)
@click.option("--backend", type=click.Choice(BACKEND_CHOICES), default="claude-code")
@click.option("--output-dir", type=click.Path(), default=None, help="Output directory (default: agents_dir).")
@click.option("--clean", is_flag=True, default=False, help="Kill running agents and wipe output dir before creating.")
@click.pass_context
def batch_create(ctx, roles, interest_globs, personas, count, strategy, seed, backend, output_dir, clean):
    """Create a batch of agents by sampling from role x interest x persona."""
    cfg = _get_config(ctx)
    out = Path(output_dir) if output_dir else cfg.agents_dir

    if clean and out.exists():
        killed = kill_all_sessions()
        if killed:
            click.echo(f"Killed {killed} running agent(s).")
        shutil.rmtree(out)
        click.echo(f"Cleared {out}")

    # fall back to config dirs when no globs provided
    role_files = _expand_globs(roles) if roles else sorted(
        str(f) for f in cfg.roles_dir.glob("*.md") if f.name != "README.md"
    )
    interest_files = _expand_globs(interest_globs) if interest_globs else sorted(
        str(f) for f in cfg.interests_dir.glob("**/*.md") if f.name != "README.md"
    )
    persona_files = _expand_globs(personas) if personas else sorted(
        str(f) for f in cfg.personas_dir.glob("*.json") if not f.name.startswith("all_")
    )

    if not role_files:
        raise click.ClickException("No role files matched.")
    if not interest_files:
        raise click.ClickException("No interest files matched.")
    if not persona_files:
        raise click.ClickException("No persona files matched.")

    click.echo(f"Roles: {len(role_files)}, Interests: {len(interest_files)}, Personas: {len(persona_files)}")
    click.echo(f"Sampling {count} agents ({strategy}, seed={seed})\n")

    samples = sample(role_files, interest_files, persona_files, n=count, strategy=strategy, seed=seed)

    out.mkdir(parents=True, exist_ok=True)

    backend_obj = get_backend(backend)

    for i, s in enumerate(samples):
        agent_name = f"agent_{i:03d}__{s.name}"
        agent_dir = out / agent_name
        agent_dir.mkdir(exist_ok=True)

        prompt = compile_agent_prompt(
            role_path=Path(s.role),
            persona_path=Path(s.persona),
            interest_path=Path(s.interests),
            global_rules_path=cfg.global_rules_path,
            platform_skills_path=cfg.platform_skills_path,
        )

        (agent_dir / "prompt.md").write_text(prompt, encoding="utf-8")
        (agent_dir / backend_obj.prompt_filename).write_text(prompt, encoding="utf-8")
        (agent_dir / "initial_prompt.txt").write_text(DEFAULT_INITIAL_PROMPT, encoding="utf-8")

        config_data = {
            "name": agent_name,
            "backend": backend,
            "role": str(Path(s.role).resolve()),
            "persona": str(Path(s.persona).resolve()),
            "interest": str(Path(s.interests).resolve()),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (agent_dir / "config.json").write_text(json.dumps(config_data, indent=2), encoding="utf-8")
        click.echo(f"  created: {agent_name}")

    click.echo(f"\n{len(samples)} agent(s) written to {out}")


@batch.command("launch")
@click.option("--agent-dirs", multiple=True, help="Agent directory globs (default: agents_dir/*).")
@click.option("--duration", type=float, default=None, help="Hours to run (default: indefinite).")
@click.option("--session-timeout", type=int, default=600, help="Max seconds per invocation before restart (default: 600).")
@click.pass_context
def batch_launch(ctx, agent_dirs, duration, session_timeout):
    """Launch all agents in parallel (one tmux session each)."""
    cfg = _get_config(ctx)
    if agent_dirs:
        dirs = _expand_globs(agent_dirs)
    else:
        dirs = sorted(str(d) for d in cfg.agents_dir.iterdir() if d.is_dir())
    dirs = [d for d in dirs if Path(d).is_dir() and (Path(d) / "config.json").exists()]

    if not dirs:
        raise click.ClickException("No valid agent directories found.")

    launched = 0
    for d in dirs:
        d = Path(d)
        agent_config = json.loads((d / "config.json").read_text())
        name = agent_config["name"]
        backend_obj = get_backend(agent_config["backend"])

        if has_session(name):
            click.echo(f"  skipped (already running): {name}")
            continue

        # ensure backend-specific prompt file
        prompt_file = d / backend_obj.prompt_filename
        if not prompt_file.exists():
            prompt = (d / "prompt.md").read_text(encoding="utf-8")
            prompt_file.write_text(prompt, encoding="utf-8")

        initial_prompt = (d / "initial_prompt.txt").read_text(encoding="utf-8").strip()
        escaped_prompt = initial_prompt.replace('"', '\\"')
        cmd = backend_obj.command_template.format(prompt=escaped_prompt)

        script = build_launch_script(cmd, duration_hours=duration, session_timeout=session_timeout)
        create_session(name, str(d), script)
        launched += 1
        click.echo(f"  launched: {name}")

    click.echo(f"\n{launched} agent(s) launched.")


@batch.command("kill")
def batch_kill():
    """Stop all running agents."""
    count = kill_all_sessions()
    click.echo(f"Killed {count} agent(s).")


# --------------------------------------------------------------------------- #
# reva debug
# --------------------------------------------------------------------------- #


@main.command()
@click.option("--n", "count", type=int, default=3, help="Number of agents to preview.")
@click.option("--strategy", type=click.Choice(["stratified", "random"]), default="stratified")
@click.option("--seed", type=int, default=42)
@click.pass_context
def debug(ctx, count, strategy, seed):
    """Preview compiled prompts for sampled agents."""
    cfg = _get_config(ctx)

    role_files = sorted(str(f) for f in cfg.roles_dir.glob("*.md") if f.name != "README.md")
    interest_files = sorted(str(f) for f in cfg.interests_dir.glob("**/*.md") if f.name != "README.md")
    persona_files = sorted(str(f) for f in cfg.personas_dir.glob("*.json") if f.name.startswith("all_"))

    # filter out aggregate files
    persona_files = sorted(str(f) for f in cfg.personas_dir.glob("*.json") if not f.name.startswith("all_"))

    if not role_files or not interest_files or not persona_files:
        raise click.ClickException(
            f"Not enough components. Roles: {len(role_files)}, "
            f"Interests: {len(interest_files)}, Personas: {len(persona_files)}"
        )

    samples = sample(role_files, interest_files, persona_files, n=count, strategy=strategy, seed=seed)

    separator = "\n" + "=" * 80 + "\n"
    for i, s in enumerate(samples):
        prompt = compile_agent_prompt(
            role_path=Path(s.role),
            persona_path=Path(s.persona),
            interest_path=Path(s.interests),
            global_rules_path=cfg.global_rules_path,
            platform_skills_path=cfg.platform_skills_path,
        )
        click.echo(separator)
        click.echo(f"Agent {i + 1}/{len(samples)}: {s.name}")
        click.echo(f"  role:      {Path(s.role).name}")
        click.echo(f"  interests: {Path(s.interests).name}")
        click.echo(f"  persona:   {Path(s.persona).name}")
        click.echo(f"  chars:     {len(prompt)}")
        click.echo(separator)
        click.echo(prompt)

    click.echo(separator)


# --------------------------------------------------------------------------- #
# reva watch
# --------------------------------------------------------------------------- #


@main.command()
@click.argument("name", required=False)
@click.option("--all", "watch_all", is_flag=True, help="Watch all running agents (interleaved).")
@click.pass_context
def watch(ctx, name, watch_all):
    """Stream a readable live view of agent activity from agent.log."""
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
        # pick most recently modified agent log
        agents = sorted(
            (d for d in cfg.agents_dir.iterdir() if d.is_dir() and (d / "agent.log").exists()),
            key=lambda d: (d / "agent.log").stat().st_mtime,
            reverse=True,
        )
        if not agents:
            raise click.ClickException("No agent logs found.")
        log_files = [(agents[0].name, agents[0] / "agent.log")]
        click.echo(f"Watching: {agents[0].name}\n")

    handles = {name: open(path, "r") for name, path in log_files}
    # seek to end so we only show new lines (unless file is small)
    for name_, fh in handles.items():
        fh.seek(0)

    prefix = len(log_files) > 1

    try:
        while True:
            activity = False
            for agent_name, fh in handles.items():
                line = fh.readline()
                if not line:
                    continue
                activity = True
                _render_log_line(line.strip(), agent_name if prefix else None)
            if not activity:
                time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        for fh in handles.values():
            fh.close()


def _wrap(text: str, width: int = 100, indent: str = "  ") -> str:
    """Wrap text to width, indenting continuation lines."""
    import textwrap
    lines = text.splitlines()
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width, subsequent_indent=indent) or [""])
    return "\n".join(wrapped)


def _render_log_line(line: str, agent_name: str | None):
    """Parse one stream-json line and print a human-readable summary."""
    if not line:
        return
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        click.echo(line)
        return

    tag = f"[{agent_name[:28]}] " if agent_name else ""
    typ = d.get("type")

    if typ == "system" and d.get("subtype") == "init":
        model = d.get("model", "?")
        click.echo(click.style(f"\n{tag}▶ session started  model={model}", fg="green", bold=True))

    elif typ == "assistant":
        for block in d.get("message", {}).get("content", []):
            btype = block.get("type")

            if btype == "thinking":
                thought = block.get("thinking", "").strip()
                if thought:
                    click.echo(click.style(f"\n{tag}thinking:", fg="bright_black", bold=True))
                    click.echo(click.style(_wrap(thought, indent="  "), fg="bright_black"))

            elif btype == "text":
                text = block.get("text", "").strip()
                if text:
                    click.echo(click.style(f"\n{tag}» ", fg="cyan", bold=True) + _wrap(text, indent="  "))

            elif btype == "tool_use":
                tool = block.get("name", "?")
                inp = block.get("input", {})
                summary = _summarize_tool_input(tool, inp)
                click.echo(click.style(f"\n{tag}⚙ {tool}", fg="yellow", bold=True))
                if summary:
                    click.echo(click.style(_wrap(summary, indent="  "), fg="yellow"))

    elif typ == "user":
        for block in d.get("message", {}).get("content", []):
            if block.get("type") == "tool_result":
                result = block.get("content", "")
                if isinstance(result, list):
                    result = " ".join(r.get("text", "") for r in result if isinstance(r, dict))
                if result and result.strip():
                    click.echo(click.style(f"  ← ", fg="bright_black") +
                               click.style(_wrap(result.strip(), indent="    "), fg="bright_black"))

    elif typ == "result":
        cost = d.get("cost_usd")
        turns = d.get("num_turns")
        cost_str = f"  cost=${cost:.4f}" if cost else ""
        click.echo(click.style(f"\n{tag}■ session ended  turns={turns}{cost_str}\n", fg="red", bold=True))

    elif typ == "rate_limit_event":
        status = d.get("rate_limit_info", {}).get("status", "?")
        if status != "allowed":
            click.echo(click.style(f"{tag}⚠ rate limit: {status}", fg="magenta"))


def _summarize_tool_input(tool: str, inp: dict) -> str:
    if tool == "Bash":
        return inp.get("command", "").strip()
    if tool == "WebFetch":
        return inp.get("url", "")
    if tool in ("Write", "Edit"):
        return inp.get("file_path", "")
    if tool == "Read":
        return inp.get("file_path", "")
    if tool == "Skill":
        return inp.get("skill", "")
    if tool in ("Grep", "Glob"):
        return inp.get("pattern", "") or inp.get("query", "")
    return json.dumps(inp, ensure_ascii=False)[:200]


# --------------------------------------------------------------------------- #
# reva view
# --------------------------------------------------------------------------- #


@main.command()
@click.pass_context
def view(ctx):
    """Launch the interactive TUI viewer (dropdown + tabs)."""
    from reva.viewer import RevaViewer
    cfg = _get_config(ctx)
    app = RevaViewer(cfg=cfg)
    app.run()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _expand_globs(patterns: tuple[str, ...]) -> list[str]:
    """Expand shell glob patterns into a sorted list of paths."""
    result = []
    for pattern in patterns:
        expanded = glob.glob(pattern, recursive=True)
        if expanded:
            result.extend(expanded)
        else:
            # not a glob, treat as literal path
            result.append(pattern)
    return sorted(set(result))
