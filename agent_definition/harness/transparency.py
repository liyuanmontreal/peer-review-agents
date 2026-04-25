"""
transparency.py

Helper for writing local transparency log files before posting comments or
verdicts. Every Koala comment/verdict requires a github_file_url pointing to
a file in the agent's public repo that documents the reasoning.

Typical usage:

    from agent_definition.harness.transparency import write_transparency_log

    github_url = write_transparency_log(
        paper_id="abc123",
        action_id="comment-2026-04-24-001",
        action_type="comment",
        content="The ablation study on Table 3 is missing a baseline.",
        reasoning="Claim X on p.4 is unsupported without this comparison.",
    )
    # Pass github_url as github_file_url when calling post_comment / post_verdict.

GSR artifact usage:

    from agent_definition.harness.transparency import write_artifact

    url = write_artifact(
        paper_id="abc123",
        artifact_kind="comment_draft",
        short_id="20260424T120000Z",
        source_phase="review",
        summary="Missing baseline in Table 3",
        content="The comparison in §5 omits the strongest published baseline.",
    )
    # Pass url as github_file_url when calling post_comment.

    url = write_artifact(
        paper_id="abc123",
        artifact_kind="comment_trace",
        short_id="20260424T120000Z",
        summary="Missing baseline in Table 3",
        payload={"evidence": "Table 3 row 4", "decision_impact": "major concern"},
    )

Environment variables
---------------------
KOALA_TRANSPARENCY_LOG_DIR  Local directory for log files (default: ./logs)
KOALA_GITHUB_REPO           GitHub repo URL, e.g. https://github.com/owner/repo
KOALA_GITHUB_LOGS_PATH      Logs path within the repo (default: logs)
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path


_ARTIFACT_EXTS: dict[str, str] = {
    "paper_summary": "md",
    "comment_draft": "md",
    "comment_trace": "json",
    "verdict_draft": "md",
    "verdict_trace": "json",
}

_ARTIFACT_DEFAULT_PHASE: dict[str, str] = {
    "paper_summary": "review",
    "comment_draft": "review",
    "comment_trace": "review",
    "verdict_draft": "verdict",
    "verdict_trace": "verdict",
}


def write_transparency_log(
    paper_id: str,
    action_id: str,
    action_type: str,
    content: str,
    *,
    score: float | None = None,
    cited_comments: list[str] | None = None,
    model: str | None = None,
    reasoning: str = "",
    log_dir: str | None = None,
    github_repo: str | None = None,
    logs_base_path: str | None = None,
) -> str:
    """Write a local transparency log and return the expected GitHub blob URL.

    Args:
        paper_id: Koala paper identifier.
        action_id: Unique identifier for this action — used as the filename.
        action_type: "comment" or "verdict".
        content: Markdown body of the comment or verdict.
        score: Verdict score 0.0–10.0 (verdicts only).
        cited_comments: List of [[comment:<uuid>]] citation strings.
        model: LLM model/backend identifier, if available.
        reasoning: Brief rationale for this action.
        log_dir: Override local log directory.
        github_repo: Override GitHub repo URL.
        logs_base_path: Override the logs sub-path within the repo.

    Returns:
        Full GitHub blob URL for the written file, or a TODO placeholder
        when KOALA_GITHUB_REPO is not configured.
    """
    base_dir = Path(log_dir or os.environ.get("KOALA_TRANSPARENCY_LOG_DIR", "./logs"))
    log_path = base_dir / paper_id / f"{action_id}.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()

    lines = [
        f"# Transparency Log: {action_type}",
        "",
        f"- **timestamp**: {timestamp}",
        f"- **action_type**: {action_type}",
        f"- **paper_id**: {paper_id}",
        f"- **action_id**: {action_id}",
    ]
    if model:
        lines.append(f"- **model**: {model}")
    if score is not None:
        lines.append(f"- **score**: {score}")
    lines += [
        "",
        "## Content",
        "",
        content,
    ]
    if cited_comments:
        lines += ["", "## Cited Comments", ""]
        lines += [f"- {c}" for c in cited_comments]
    lines += [
        "",
        "## Reasoning",
        "",
        reasoning or "<!-- TODO: add reasoning before committing -->",
    ]

    log_path.write_text("\n".join(lines), encoding="utf-8")

    repo = (github_repo or os.environ.get("KOALA_GITHUB_REPO", "")).rstrip("/")
    logs_path = logs_base_path or os.environ.get("KOALA_GITHUB_LOGS_PATH", "logs")

    if repo:
        return f"{repo}/blob/main/{logs_path}/{paper_id}/{action_id}.md"
    return f"TODO: set KOALA_GITHUB_REPO — local log at {log_path}"


def write_artifact(
    paper_id: str,
    artifact_kind: str,
    *,
    short_id: str | None = None,
    source_phase: str | None = None,
    summary: str = "",
    payload: dict | None = None,
    content: str = "",
    log_dir: str | None = None,
    github_repo: str | None = None,
    logs_base_path: str | None = None,
) -> str:
    """Write a structured GSR artifact and return the expected GitHub blob URL.

    Supported artifact_kind values and their output formats:
      paper_summary  — markdown overview of a paper (no short_id required)
      comment_draft  — markdown draft of a comment before posting
      comment_trace  — JSON reasoning trace for a comment
      verdict_draft  — markdown draft of a verdict before posting
      verdict_trace  — JSON reasoning trace for a verdict

    JSON artifacts use the schema:
      artifact_kind, paper_id, created_at, source_phase, summary, payload

    Args:
        paper_id: Koala paper identifier.
        artifact_kind: One of the five supported kinds listed above.
        short_id: Short identifier appended to the filename. Auto-generated
            from the current timestamp if omitted. Not used for paper_summary.
        source_phase: "review", "verdict", or "internal". Defaults to the
            natural phase for each kind when omitted.
        summary: One-line description of the artifact's purpose.
        payload: Structured data dict (JSON artifacts only).
        content: Markdown body (markdown artifacts only).
        log_dir: Override local log directory.
        github_repo: Override GitHub repo URL.
        logs_base_path: Override the logs sub-path within the repo.

    Returns:
        Full GitHub blob URL for the written file, or a TODO placeholder
        when KOALA_GITHUB_REPO is not configured.

    Raises:
        ValueError: When artifact_kind is not one of the five supported values.
    """
    if artifact_kind not in _ARTIFACT_EXTS:
        raise ValueError(
            f"Unknown artifact_kind: {artifact_kind!r}. "
            f"Must be one of {list(_ARTIFACT_EXTS)}"
        )

    ext = _ARTIFACT_EXTS[artifact_kind]
    resolved_phase = source_phase or _ARTIFACT_DEFAULT_PHASE[artifact_kind]
    timestamp = datetime.now(timezone.utc).isoformat()
    sid = short_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if artifact_kind == "paper_summary":
        filename = f"paper_{paper_id}_summary.{ext}"
    else:
        filename = f"{artifact_kind}_{paper_id}_{sid}.{ext}"

    base_dir = Path(log_dir or os.environ.get("KOALA_TRANSPARENCY_LOG_DIR", "./logs"))
    artifact_path = base_dir / paper_id / filename
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    if ext == "json":
        data = {
            "artifact_kind": artifact_kind,
            "paper_id": paper_id,
            "created_at": timestamp,
            "source_phase": resolved_phase,
            "summary": summary,
            "payload": payload or {},
        }
        artifact_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        lines = [
            f"# GSR Artifact: {artifact_kind}",
            "",
            f"- **artifact_kind**: {artifact_kind}",
            f"- **paper_id**: {paper_id}",
            f"- **created_at**: {timestamp}",
            f"- **source_phase**: {resolved_phase}",
        ]
        if summary:
            lines += ["", f"**Summary:** {summary}"]
        lines += ["", "## Content", "", content or "<!-- no content provided -->"]
        artifact_path.write_text("\n".join(lines), encoding="utf-8")

    repo = (github_repo or os.environ.get("KOALA_GITHUB_REPO", "")).rstrip("/")
    logs_path = logs_base_path or os.environ.get("KOALA_GITHUB_LOGS_PATH", "logs")

    if repo:
        return f"{repo}/blob/main/{logs_path}/{paper_id}/{filename}"
    return f"TODO: set KOALA_GITHUB_REPO — local artifact at {artifact_path}"
