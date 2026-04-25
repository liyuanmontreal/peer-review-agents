#!/usr/bin/env python3
"""
Verify safe-mode behavior before launching a real agent.

Checks:
  1. KOALA_WRITE_ENABLED defaults to false (safe)
  2. post_comment is intercepted in safe mode
  3. post_verdict is intercepted in safe mode
  4. Read tools (get_papers, get_paper, etc.) are NOT intercepted
  5. Transparency log helper can write a sample log file

Usage:
    python scripts/check_safe_mode.py

Exit code 0 = all checks passed, safe to test-launch.
Exit code 1 = one or more checks failed.
"""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent

# Ensure reva package (cli/) is importable when running outside pytest.
_cli_path = str(ROOT / "cli")
if _cli_path not in sys.path:
    sys.path.insert(0, _cli_path)


def _load(relative_path: str, name: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ok(label: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    line = f"  [{status}] {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return passed


def _httpx_stub():
    stub = MagicMock()
    resp = MagicMock()
    resp.json.return_value = {"result": {"content": [{"type": "text", "text": "stubbed-read"}]}}
    resp.raise_for_status.return_value = None
    stub.post.return_value = resp
    return stub


def main() -> int:
    print("=" * 60)
    print("Koala safe-mode verification")
    print("=" * 60)
    all_passed = True

    # ── 1. KOALA_WRITE_ENABLED defaults to false ──────────────────────────────
    print("\n1. Environment default")
    env_val = os.environ.get("KOALA_WRITE_ENABLED", "")
    is_safe = env_val.lower() != "true"
    all_passed &= _ok(
        "KOALA_WRITE_ENABLED is not 'true'",
        is_safe,
        f"current value: {env_val!r}",
    )
    if not is_safe:
        print("    WARNING: real writes are enabled — unset KOALA_WRITE_ENABLED to use safe mode")

    # ── 2 & 3. Write tools are intercepted ────────────────────────────────────
    print("\n2. Write tool interception (KOALA_WRITE_ENABLED unset)")
    os.environ.pop("KOALA_WRITE_ENABLED", None)

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["KOALA_TRANSPARENCY_LOG_DIR"] = tmp
        try:
            sys.modules.setdefault("httpx", _httpx_stub())
            koala = _load("agent_definition/harness/koala.py", "_check_koala")
            client = koala.KoalaClient(api_key="check-key")

            for tool_name, args in [
                (
                    "post_comment",
                    {
                        "paper_id": "check-paper",
                        "content_markdown": "Safe-mode test comment.",
                        "github_file_url": "https://example.com/log.md",
                    },
                ),
                (
                    "post_verdict",
                    {
                        "paper_id": "check-paper",
                        "score": 5.0,
                        "content_markdown": "Safe-mode test verdict.",
                        "github_file_url": "https://example.com/log.md",
                    },
                ),
            ]:
                raw = client.call_tool(tool_name, args)
                try:
                    result = json.loads(raw)
                    intercepted = result.get("safe_mode") is True and result.get("would_post") is True
                except (json.JSONDecodeError, AttributeError):
                    intercepted = False
                all_passed &= _ok(f"{tool_name} intercepted in safe mode", intercepted)

        finally:
            os.environ.pop("KOALA_TRANSPARENCY_LOG_DIR", None)

    # ── 4. Read tools are not intercepted ─────────────────────────────────────
    print("\n3. Read tools pass-through")
    read_tools = [
        "get_papers",
        "get_paper",
        "get_comments",
        "get_actor_profile",
        "get_notifications",
        "mark_notifications_read",
        "get_unread_count",
    ]
    write_tools_set = koala._WRITE_TOOLS
    not_blocked = [t for t in read_tools if t not in write_tools_set]
    all_passed &= _ok(
        "read tools are not in _WRITE_TOOLS",
        len(not_blocked) == len(read_tools),
        f"checked: {', '.join(read_tools)}",
    )

    # ── 5. Transparency log helper ─────────────────────────────────────────────
    print("\n4. Transparency log helper")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            transparency = _load(
                "agent_definition/harness/transparency.py", "_check_transparency"
            )
            github_repo = "https://github.com/liyuanmontreal/peer-review-agents"
            url = transparency.write_transparency_log(
                paper_id="check-paper",
                action_id="check-action-001",
                action_type="comment",
                content="Sample comment for safe-mode verification.",
                reasoning="Testing that the transparency helper writes correctly.",
                log_dir=tmp,
                github_repo=github_repo,
            )
            log_file = Path(tmp) / "check-paper" / "check-action-001.md"
            all_passed &= _ok("log file written to disk", log_file.exists(), str(log_file))
            all_passed &= _ok(
                "log returns GitHub blob URL",
                url.startswith("https://github.com/liyuanmontreal/peer-review-agents/blob/main/"),
                url,
            )
            if log_file.exists():
                text = log_file.read_text()
                all_passed &= _ok(
                    "log contains paper_id and action_type",
                    "check-paper" in text and "comment" in text,
                )
        except Exception as exc:
            all_passed &= _ok("transparency log helper", False, str(exc))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_passed:
        print("All checks passed. Safe to launch in read-only / safe mode.")
    else:
        print("SOME CHECKS FAILED. Review output above before launching.")
    print()
    print("To enable real writes when ready:")
    print("  export KOALA_WRITE_ENABLED=true")
    print("  export KOALA_GITHUB_REPO=https://github.com/liyuanmontreal/peer-review-agents")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
