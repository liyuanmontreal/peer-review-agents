"""
reva view — interactive Textual TUI for watching agent activity.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Label,
    Markdown,
    RichLog,
    Select,
    TabbedContent,
    TabPane,
)

from reva.tmux import list_sessions


# --------------------------------------------------------------------------- #
# helpers (adapted from cli.py)
# --------------------------------------------------------------------------- #


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


def _parse_log_line(line: str) -> list[tuple[str, str]]:
    """Parse one stream-json line into list of (style, text) pairs for RichLog."""
    if not line:
        return []
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        return [("dim", line)]

    typ = d.get("type")
    out: list[tuple[str, str]] = []

    if typ == "system" and d.get("subtype") == "init":
        model = d.get("model", "?")
        out.append(("bold green", f"\n▶ session started  model={model}"))

    elif typ == "assistant":
        for block in d.get("message", {}).get("content", []):
            btype = block.get("type")
            if btype == "thinking":
                thought = block.get("thinking", "").strip()
                if thought:
                    out.append(("bold dim", "\nthinking:"))
                    out.append(("dim", f"  {thought}"))
            elif btype == "text":
                text = block.get("text", "").strip()
                if text:
                    out.append(("bold cyan", "\n» "))
                    out.append(("", text))
            elif btype == "tool_use":
                tool = block.get("name", "?")
                inp = block.get("input", {})
                summary = _summarize_tool_input(tool, inp)
                out.append(("bold yellow", f"\n⚙ {tool}"))
                if summary:
                    out.append(("yellow", f"  {summary}"))

    elif typ == "user":
        for block in d.get("message", {}).get("content", []):
            if block.get("type") == "tool_result":
                result = block.get("content", "")
                if isinstance(result, list):
                    result = " ".join(r.get("text", "") for r in result if isinstance(r, dict))
                if result and result.strip():
                    out.append(("dim", f"  ← {result.strip()[:300]}"))

    elif typ == "result":
        cost = d.get("cost_usd")
        turns = d.get("num_turns")
        cost_str = f"  cost=${cost:.4f}" if cost else ""
        out.append(("bold red", f"\n■ session ended  turns={turns}{cost_str}\n"))

    elif typ == "rate_limit_event":
        status = d.get("rate_limit_info", {}).get("status", "?")
        if status != "allowed":
            out.append(("bold magenta", f"⚠ rate limit: {status}"))

    return out


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #


class RevaViewer(App):
    TITLE = "reva viewer"
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh_agents", "Refresh"),
    ]
    CSS = """
    #toolbar {
        height: 3;
        padding: 0 1;
        background: $panel;
        border-bottom: solid $primary;
    }
    #agent-select {
        width: 1fr;
    }
    #refresh-btn {
        width: 12;
        margin-left: 1;
    }
    #output-log {
        scrollbar-gutter: stable;
    }
    #system-prompt {
        padding: 1 2;
    }
    #agent-table {
        height: 1fr;
    }
    TabbedContent {
        height: 1fr;
    }
    """

    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self._current_agent: str | None = None
        self._tail_running = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="toolbar"):
            yield Label("Agent: ", classes="label")
            yield Select([], id="agent-select", prompt="— pick an agent —")
            yield Button("Refresh", id="refresh-btn", variant="primary")
        with TabbedContent():
            with TabPane("Output", id="tab-output"):
                yield RichLog(id="output-log", highlight=False, markup=False, wrap=True)
            with TabPane("System Prompt", id="tab-prompt"):
                yield Markdown("", id="system-prompt")
            with TabPane("Agent Info", id="tab-info"):
                yield DataTable(id="agent-table", zebra_stripes=True)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#agent-table", DataTable)
        table.add_columns("Field", "Value")
        self._populate_agent_list()
        self.set_interval(5, self._populate_agent_list)

    # ------------------------------------------------------------------ #
    # agent list
    # ------------------------------------------------------------------ #

    def _get_agent_names(self) -> list[str]:
        """All agents with a log file (running or not)."""
        running = {s.agent_name for s in list_sessions()}
        names = set()
        if self.cfg.agents_dir.exists():
            for d in self.cfg.agents_dir.iterdir():
                if d.is_dir() and (d / "agent.log").exists():
                    names.add(d.name)
        # running agents first
        return sorted(running, key=str) + sorted(names - running, key=str)

    def _populate_agent_list(self) -> None:
        names = self._get_agent_names()
        sel = self.query_one("#agent-select", Select)
        options = [(name, name) for name in names]
        sel.set_options(options)
        # restore selection if still valid
        if self._current_agent and self._current_agent in names:
            sel.value = self._current_agent

    # ------------------------------------------------------------------ #
    # events
    # ------------------------------------------------------------------ #

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.value is Select.BLANK:
            return
        name = str(event.value)
        if name == self._current_agent:
            return
        self._current_agent = name
        self._load_agent(name)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "refresh-btn":
            self._populate_agent_list()

    def action_refresh_agents(self) -> None:
        self._populate_agent_list()

    # ------------------------------------------------------------------ #
    # agent loading
    # ------------------------------------------------------------------ #

    def _load_agent(self, name: str) -> None:
        agent_dir = self.cfg.agents_dir / name

        # clear and reload output log
        log_widget = self.query_one("#output-log", RichLog)
        log_widget.clear()
        self._tail_running = False  # signal old worker to stop
        time.sleep(0.05)           # give worker a moment to notice
        log_path = agent_dir / "agent.log"
        if log_path.exists():
            self._tail_log(log_path)

        # system prompt tab
        prompt_widget = self.query_one("#system-prompt", Markdown)
        claude_md = agent_dir / "CLAUDE.md"
        if claude_md.exists():
            self.call_later(prompt_widget.update, claude_md.read_text(encoding="utf-8"))
        else:
            self.call_later(prompt_widget.update, "_No system prompt found._")

        # agent info tab
        table = self.query_one("#agent-table", DataTable)
        table.clear()
        config_path = agent_dir / "config.json"
        if config_path.exists():
            cfg_data = json.loads(config_path.read_text(encoding="utf-8"))
            for key, val in cfg_data.items():
                # shorten long paths to just the filename
                if isinstance(val, str) and "/" in val:
                    val = Path(val).name
                table.add_row(key, str(val))
        # running status
        running = {s.agent_name for s in list_sessions()}
        table.add_row("status", "running" if name in running else "stopped")

    # ------------------------------------------------------------------ #
    # log tailing
    # ------------------------------------------------------------------ #

    @work(thread=True)
    def _tail_log(self, log_path: Path) -> None:
        self._tail_running = True
        log_widget = self.query_one("#output-log", RichLog)
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            while self._tail_running:
                line = fh.readline()
                if not line:
                    time.sleep(0.2)
                    continue
                line = line.strip()
                if not line:
                    continue
                segments = _parse_log_line(line)
                for style, text in segments:
                    if text:
                        self.call_from_thread(log_widget.write, text)
