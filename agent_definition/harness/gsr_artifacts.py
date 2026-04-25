"""
gsr_artifacts.py

Automatic GSR artifact hook. Called by KoalaClient on every post_comment /
post_verdict call — before both safe-mode intercept and real writes.

Creates structured artifacts under agent_configs/gsr_agent/reviews/ so that
review evidence is persisted regardless of whether KOALA_WRITE_ENABLED is set.
"""
import re
from datetime import datetime, timezone
from pathlib import Path

from .transparency import write_artifact

GSR_ARTIFACT_DIR = "agent_configs/gsr_agent/reviews"
GSR_LOGS_BASE = "agent_configs/gsr_agent/reviews"


def emit_gsr_artifacts(
    tool_name: str,
    arguments: dict,
    *,
    safe_mode: bool,
    artifact_dir: str = GSR_ARTIFACT_DIR,
    logs_base_path: str = GSR_LOGS_BASE,
) -> None:
    """Emit GSR artifacts for post_comment or post_verdict.

    Safe to call unconditionally — catches all exceptions so it cannot
    block the main write path.
    """
    try:
        if tool_name == "post_comment":
            _emit_comment_artifacts(arguments, safe_mode, artifact_dir, logs_base_path)
        elif tool_name == "post_verdict":
            _emit_verdict_artifacts(arguments, safe_mode, artifact_dir, logs_base_path)
    except Exception as exc:
        print(f"[gsr-artifacts] artifact emission failed (non-fatal): {exc}")


def _short_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _extract_cited_comments(text: str) -> list[str]:
    return re.findall(r"\[\[comment:[^\]]+\]\]", text)


def _emit_comment_artifacts(
    arguments: dict,
    safe_mode: bool,
    artifact_dir: str,
    logs_base_path: str,
) -> None:
    paper_id = arguments.get("paper_id", "unknown")
    content_markdown = arguments.get("content_markdown", "")
    parent_id = arguments.get("parent_id")
    sid = _short_id()
    kw = dict(log_dir=artifact_dir, logs_base_path=logs_base_path)

    summary_path = Path(artifact_dir) / paper_id / f"paper_{paper_id}_summary.md"
    if not summary_path.exists():
        write_artifact(
            paper_id,
            "paper_summary",
            summary=f"Placeholder for {paper_id} — agent should enrich after reading.",
            content=(
                f"Paper `{paper_id}` — auto-created placeholder summary.\n\n"
                "Replace with a real analysis after reading the paper."
            ),
            **kw,
        )

    draft_content = "\n".join([
        f"- **paper_id**: {paper_id}",
        f"- **parent_id**: {parent_id or '(none — top-level thread)'}",
        "",
        "## Comment Content",
        "",
        content_markdown,
    ])
    write_artifact(
        paper_id,
        "comment_draft",
        short_id=sid,
        summary=(content_markdown[:80].replace("\n", " ") or "(empty)"),
        content=draft_content,
        **kw,
    )

    write_artifact(
        paper_id,
        "comment_trace",
        short_id=sid,
        summary=(content_markdown[:80].replace("\n", " ") or "(empty)"),
        payload={
            "action_type": "post_comment",
            "paper_id": paper_id,
            "parent_id": parent_id,
            "safe_mode": safe_mode,
            "tool_arguments": arguments,
        },
        **kw,
    )


def write_local_artifact_smoke(
    agent_name: str = "gsr_agent",
    artifact_dir: str = GSR_ARTIFACT_DIR,
) -> None:
    """Write a local-only artifact smoke task with a realistic comment artifact pair.

    Creates three files under reviews/local_artifact_smoke/:
      paper_local_artifact_smoke_summary.md
      comment_draft_local_artifact_smoke_<shortid>.md
      comment_trace_local_artifact_smoke_<shortid>.json

    No Koala API call is made. No karma consumed. Safe to call unconditionally.
    To disable: remove the call in Agent.run() in harness.py.
    """
    try:
        paper_id = "local_artifact_smoke"
        sid = _short_id()
        placeholder = "Local artifact smoke test only; no Koala comment was attempted."
        kw = dict(log_dir=artifact_dir)

        write_artifact(
            paper_id,
            "paper_summary",
            summary="Local-only smoke artifact — no Koala comment was attempted.",
            content=(
                "Local-only smoke artifact. Verifies the realistic comment artifact format.\n\n"
                f"- **local_only**: true\n"
                f"- **karma_consumed**: none\n"
                f"- **agent_name**: {agent_name}\n"
                "- **note**: no Koala post_comment or post_verdict was called"
            ),
            **kw,
        )

        write_artifact(
            paper_id,
            "comment_draft",
            short_id=sid,
            summary=placeholder,
            content=placeholder,
            **kw,
        )

        write_artifact(
            paper_id,
            "comment_trace",
            short_id=sid,
            summary=placeholder,
            payload={
                "action_type": "local_smoke",
                "paper_id": paper_id,
                "safe_mode": True,
                "local_only": True,
                "note": "no post_comment called; no karma consumed",
                "agent_name": agent_name,
            },
            **kw,
        )

        print(f"[gsr-artifacts] local artifact smoke written: {Path(artifact_dir) / paper_id}")
    except Exception as exc:
        print(f"[gsr-artifacts] local artifact smoke failed (non-fatal): {exc}")


def write_smoke_artifact(
    agent_name: str = "gsr_agent",
    artifact_dir: str = GSR_ARTIFACT_DIR,
) -> None:
    """Write a local smoke artifact at startup to verify the artifact creation path.

    Creates agent_configs/gsr_agent/reviews/smoke_test/paper_smoke_test_summary.md.
    No Koala API call is made. Safe to call unconditionally — never raises.

    To disable: remove the call in Agent.run() in harness.py.
    """
    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        smoke_dir = Path(artifact_dir) / "smoke_test"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        smoke_path = smoke_dir / "paper_smoke_test_summary.md"
        lines = [
            "# Smoke Artifact: startup verification",
            "",
            f"- **timestamp**: {timestamp}",
            f"- **agent_name**: {agent_name}",
            "- **purpose**: local smoke artifact — verifies artifact creation path only",
            "- **koala_post**: no Koala post attempted",
        ]
        smoke_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[gsr-artifacts] smoke artifact written: {smoke_path}")
    except Exception as exc:
        print(f"[gsr-artifacts] smoke artifact failed (non-fatal): {exc}")


def _emit_verdict_artifacts(
    arguments: dict,
    safe_mode: bool,
    artifact_dir: str,
    logs_base_path: str,
) -> None:
    paper_id = arguments.get("paper_id", "unknown")
    content_markdown = arguments.get("content_markdown", "")
    score = arguments.get("score")
    cited = _extract_cited_comments(content_markdown)
    sid = _short_id()
    kw = dict(log_dir=artifact_dir, logs_base_path=logs_base_path)

    draft_lines = [
        f"- **paper_id**: {paper_id}",
        f"- **score**: {score}",
        "",
        "## Verdict Content",
        "",
        content_markdown,
    ]
    if cited:
        draft_lines += ["", "## Cited Comments", ""]
        draft_lines += [f"- {c}" for c in cited]

    write_artifact(
        paper_id,
        "verdict_draft",
        short_id=sid,
        summary=f"Score {score} — {content_markdown[:60].replace(chr(10), ' ')}",
        content="\n".join(draft_lines),
        **kw,
    )

    write_artifact(
        paper_id,
        "verdict_trace",
        short_id=sid,
        summary=f"Score {score}",
        payload={
            "action_type": "post_verdict",
            "paper_id": paper_id,
            "score": score,
            "safe_mode": safe_mode,
            "cited_comments": cited,
            "tool_arguments": arguments,
        },
        **kw,
    )
