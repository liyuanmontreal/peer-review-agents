"""
koala.py

Thin HTTP client for the Koala Science MCP endpoint.
All platform tool calls go through here.

Write safety switch
-------------------
By default (KOALA_WRITE_ENABLED unset or not "true") all write operations
(post_comment, post_verdict) are intercepted: the payload is logged locally
and a structured safe-mode result is returned instead of contacting Koala.

Set KOALA_WRITE_ENABLED=true to enable real writes.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx

from reva.env import koala_base_url

from .window import _UUID_RE, koala_window_state

_WRITE_TOOLS = frozenset({"post_comment", "post_verdict"})


class KoalaClient:
    def __init__(self, api_key: str | None = None, on_write=None):
        self.api_key = api_key or os.environ["COALESCENCE_API_KEY"]
        self.mcp_url = f"{koala_base_url()}/mcp"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self._id = 0
        self._on_write = on_write
        self._paper_cache: dict[str, dict] = {}

    def call_tool(self, name: str, arguments: dict) -> str:
        if name in _WRITE_TOOLS:
            safe = not _write_enabled()
            if self._on_write is not None:
                try:
                    self._on_write(name, arguments, safe_mode=safe)
                except Exception as exc:
                    print(f"[koala] on_write hook error (non-fatal): {exc}")
            if safe:
                return _intercept_write(name, arguments)

            paper_id = arguments.get("paper_id", "")
            if not _UUID_RE.match(paper_id):
                print(f"[competition] SKIP paper_id={paper_id!r} not a full UUID")
                return json.dumps({"error": "paper_id must be a full UUID", "paper_id": paper_id})

            cached = self._paper_cache.get(paper_id)
            if cached is not None:
                ws = koala_window_state(cached)
                print(
                    f"[competition] WINDOW paper_id={paper_id} phase={ws['phase']} "
                    f"open={ws['open']} seconds_left={ws['seconds_left']:.0f}"
                )
                if not ws["open"]:
                    tool_label = "POST_COMMENT" if name == "post_comment" else "POST_VERDICT"
                    print(f"[competition] {tool_label} failed paper_id={paper_id} reason=window_closed")
                    return json.dumps({"error": "window_closed", "paper_id": paper_id})

        self._id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        try:
            resp = httpx.post(self.mcp_url, json=payload, headers=self.headers, timeout=30)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 409:
                paper_id = arguments.get("paper_id", "unknown")
                tool_label = "POST_COMMENT" if name == "post_comment" else "POST_VERDICT"
                print(f"[competition] {tool_label} failed paper_id={paper_id} reason=window_closed (409)")
                return json.dumps({"error": "window_closed", "paper_id": paper_id})
            raise
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Koala Science error: {data['error']}")
        content = data.get("result", {}).get("content", [])
        result = "\n".join(
            block.get("text", "") for block in content if block.get("type") == "text"
        )
        if name == "get_paper":
            pid = arguments.get("paper_id", "")
            try:
                paper_data = json.loads(result)
                if isinstance(paper_data, dict):
                    self._paper_cache[pid] = paper_data
            except (json.JSONDecodeError, ValueError):
                pass
        return result


def _write_enabled() -> bool:
    return os.environ.get("KOALA_WRITE_ENABLED", "").lower() == "true"


def _intercept_write(name: str, arguments: dict) -> str:
    paper_id = arguments.get("paper_id", "unknown")
    github_file_url = arguments.get("github_file_url", "")

    _write_safe_mode_log(name, arguments)

    result = {
        "safe_mode": True,
        "would_post": True,
        "tool": name,
        "paper_id": paper_id,
        "github_file_url": github_file_url,
        "payload": arguments,
    }

    print(
        f"[safe-mode] {name} intercepted (paper={paper_id}). "
        f"Set KOALA_WRITE_ENABLED=true to enable real writes."
    )
    if github_file_url:
        print(f"[safe-mode] transparency log URL: {github_file_url}")
    print(f"[safe-mode] payload:\n{json.dumps(arguments, indent=2)}")

    return json.dumps(result)


def _write_safe_mode_log(tool_name: str, arguments: dict) -> None:
    """Write a local audit log for an intercepted write operation."""
    paper_id = arguments.get("paper_id", "unknown")
    log_dir = Path(os.environ.get("KOALA_TRANSPARENCY_LOG_DIR", "./logs"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    action_id = f"safe_mode_{tool_name}_{timestamp}"
    log_path = log_dir / paper_id / f"{action_id}.md"

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Safe-Mode Intercept: {tool_name}",
            "",
            f"- **timestamp**: {datetime.now(timezone.utc).isoformat()}",
            f"- **tool**: {tool_name}",
            f"- **paper_id**: {paper_id}",
            f"- **safe_mode**: true — KOALA_WRITE_ENABLED was not set to true",
            "",
            "## Would-Be Payload",
            "",
            "```json",
            json.dumps(arguments, indent=2),
            "```",
        ]
        log_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[safe-mode] log written: {log_path}")
    except OSError as exc:
        print(f"[safe-mode] could not write log: {exc}")
