from __future__ import annotations
from typing import Any
from gsr.config import INVITATION_SUFFIXES
from pathlib import Path
import logging
import openreview
import hashlib
import re

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _download_pdf(client, note_id: str, out_path: Path) -> tuple[str | None, str | None, str | None]:
    try:
        #pdf_bytes = client.get_attachment(note_id, "pdf")
        pdf_bytes = client.get_attachment(field_name='pdf', id=note_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(pdf_bytes)
        return str(out_path), _sha256_bytes(pdf_bytes), None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"



def _normalize_openreview_error(err: str) -> str:
    """
    Reduce noisy OpenReviewException strings into stable buckets.
    Removes reqId and other request-specific tokens.
    """
    # 典型：OpenReviewException: {'name': 'NotFoundError', 'message': '...', 'status': 404, ... 'reqId': '...'}
    # 先去掉 reqId 片段
    err2 = re.sub(r"'reqId'\s*:\s*'[^']+'", "'reqId':'<redacted>'", err)
    return err2


def _safe_value(content: dict | None, field: str) -> Any:
    """Extract ``content[field]['value']``, returning *None* if missing."""
    if content is None:
        return None
    try:
        return content[field]["value"]
    except (KeyError, TypeError):
        return None


def _get_submission_name(client: openreview.api.OpenReviewClient, venue_id: str) -> str:
    """Resolve the submission invitation name for *venue_id*.

    E.g. ``"ICLR.cc/2024/Conference/-/Submission"``
    """
    venue_group = client.get_group(venue_id)
    submission_name = venue_group.content.get("submission_name", {}).get("value", "Submission")
    return f"{venue_id}/-/{submission_name}"


def _classify_reply(reply: dict) -> str | None:
    """Return a logical type for *reply* by matching invitation suffixes.

    Returns one of ``'review'``, ``'rebuttal'``, ``'meta_review'``,
    ``'decision'``, or *None* if no suffix matches.
    """
    invitations = reply.get("invitations", [])
    for inv in invitations:
        for key, suffixes in INVITATION_SUFFIXES.items():
            if any(inv.endswith(s) for s in suffixes):
                return key
    return None


# ---------------------------------------------------------------------------
# Review / rebuttal / meta-review / decision extraction
# ---------------------------------------------------------------------------

def _extract_review(reply: dict) -> dict:
    c = reply.get("content", {})
    return {
        "id": reply["id"],
        "forum": reply.get("forum"),
        "replyto": reply.get("replyto"),
        "signatures": reply.get("signatures", []),
        "rating": _safe_value(c, "rating"),
        "confidence": _safe_value(c, "confidence"),
        "summary": _safe_value(c, "summary"),
        "strengths": _safe_value(c, "strengths"),
        "weaknesses": _safe_value(c, "weaknesses"),
        "questions": _safe_value(c, "questions"),
        "soundness": _safe_value(c, "soundness"),
        "presentation": _safe_value(c, "presentation"),
        "contribution": _safe_value(c, "contribution"),
    }


def _extract_rebuttal(reply: dict) -> dict:
    c = reply.get("content", {})
    return {
        "id": reply["id"],
        "forum": reply.get("forum"),
        "replyto": reply.get("replyto"),
        "signatures": reply.get("signatures", []),
        "comment": _safe_value(c, "comment") or _safe_value(c, "rebuttal"),
    }


def _extract_meta_review(reply: dict) -> dict:
    c = reply.get("content", {})
    return {
        "id": reply["id"],
        "forum": reply.get("forum"),
        "replyto": reply.get("replyto"),
        "signatures": reply.get("signatures", []),
        "recommendation": _safe_value(c, "recommendation"),
        "metareview": _safe_value(c, "metareview"),
        "confidence": _safe_value(c, "confidence"),
    }


def _extract_decision(reply: dict) -> dict:
    c = reply.get("content", {})
    return {
        "id": reply["id"],
        "forum": reply.get("forum"),
        "decision": _safe_value(c, "decision"),
    }


# ---------------------------------------------------------------------------
# Main fetch
# ---------------------------------------------------------------------------

def fetch_venue_data(
    client: openreview.api.OpenReviewClient,
    venue_id: str,
    limit: int | None = None,
    *,
    download_pdfs: bool = False,
    pdf_dir: str | None = None,
) -> dict:


    """Fetch papers and their full review threads for *venue_id*.

    Returns a dict ``{"venue_id": str, "papers": [...]}``.
    """
    submission_inv = _get_submission_name(client, venue_id)
    log.info("Fetching submissions for invitation: %s", submission_inv)

    submissions = client.get_all_notes(invitation=submission_inv, details="replies")

    if limit is not None:
        submissions = submissions[:limit]

    log.info("Processing %d submissions", len(submissions))

    papers = []
    pdf_ok = 0
    pdf_fail = 0
    pdf_fail_reasons: dict[str, int] = {}
    pdf_skipped = 0

    total = len(submissions)
    progress_every = 20  # 每处理 20 篇打印一次进度；

    for i, note in enumerate(submissions, start=1):
        content = note.content or {}

        paper = {
            "id": note.id,
            "forum": note.forum,
            "number": note.number,
            "title": _safe_value(content, "title"),
            "authors": _safe_value(content, "authors"),
            "abstract": _safe_value(content, "abstract"),
            "keywords": _safe_value(content, "keywords"),
            "reviews": [],
            "rebuttals": [],
            "meta_reviews": [],
            "decision": None,
            "pdf_path": None,
            "pdf_sha256": None,
            "pdf_error": None,
        }

        replies = note.details.get("replies", []) if note.details else []
        for reply in replies:
            kind = _classify_reply(reply)
            if kind == "review":
                paper["reviews"].append(_extract_review(reply))
            elif kind == "rebuttal":
                paper["rebuttals"].append(_extract_rebuttal(reply))
            elif kind == "meta_review":
                paper["meta_reviews"].append(_extract_meta_review(reply))
            elif kind == "decision":
                paper["decision"] = _extract_decision(reply)

        log.debug("Note %s pdf field raw: %r", note.id, content.get("pdf"))

        # --- PDF download / resume support ---
        if download_pdfs and pdf_dir:
            safe_venue = venue_id.replace("/", "_").replace(".", "_")
            out_name = f"paper_{note.number}_{note.id}.pdf"
            out_path = Path(pdf_dir) / safe_venue / out_name

            if out_path.exists() and out_path.stat().st_size > 0:
                paper["pdf_path"] = str(out_path)
                paper["pdf_error"] = None
                pdf_ok += 1
                pdf_skipped += 1
            else:
                pdf_path, pdf_sha256, err = _download_pdf(client, note.id, out_path)
                paper["pdf_path"] = pdf_path
                paper["pdf_sha256"] = pdf_sha256
                paper["pdf_error"] = err

                if pdf_path is not None:
                    pdf_ok += 1
                    # ⚠️ 不再每篇都 log.info("PDF saved ...")
                else:
                    pdf_fail += 1
                    reason = _normalize_openreview_error(err or "unknown_error")
                    pdf_fail_reasons[reason] = pdf_fail_reasons.get(reason, 0) + 1
                    log.warning("PDF download failed for paper %s (%s): %s", note.number, note.id, reason)

            # --- progress log ---
            if (i % progress_every == 0) or (i == total):
                pct = (i / total) * 100 if total else 100.0
                log.info(
                    "PDF progress: %.1f%% (%d/%d) | ok=%d (skipped=%d) fail=%d",
                    pct, i, total, pdf_ok, pdf_skipped, pdf_fail
                )

        papers.append(paper)

    # summary log for PDF download results
    if download_pdfs and pdf_dir:
        log.info(
            "PDF download summary: %d succeeded (%d skipped), %d failed. Output dir: %s",
            pdf_ok, pdf_skipped, pdf_fail, pdf_dir
        )
        for reason, cnt in sorted(pdf_fail_reasons.items(), key=lambda x: x[1], reverse=True)[:5]:
            log.info("  - %s: %d", reason, cnt)
 

    return {"venue_id": venue_id, "papers": papers}
