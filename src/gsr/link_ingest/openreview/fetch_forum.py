import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import openreview

from gsr.config import DEFAULT_VENUE_ID
from gsr.config import PDF_DIR as PDF_ROOT
from gsr.data_collection.client import build_client
from gsr.data_collection.fetcher import _download_pdf


def _unwrap(v):
    # OpenReview v2 often wraps values like {"value": "..."}
    if isinstance(v, dict):
        return v.get("value")
    return v

def _invs_text(note) -> str:
    invs = getattr(note, "invitations", None)
    if isinstance(invs, list):
        return " ".join(str(x).lower() for x in invs)
    return str(invs or "").lower()

def _first_present(content: dict, keys: list[str]):
    """Return the first non-empty content[key] (after unwrap) among keys."""
    for k in keys:
        if k in content and content[k] is not None:
            v = _unwrap(content.get(k))
            if v is not None and v != "":
                return v
    return None

def _safe(s: str) -> str:
    return s.replace("/", "_").replace(".", "_")

_NON_SUBMISSION_INV_TOKENS = frozenset({
    "official_review", "review",
    "official_comment", "comment",
    "official_rebuttal", "rebuttal",
    "author_response",
    "official_decision", "decision",
    "meta_review", "metareview",
    "senior_area_chair",
    "withdrawal",
    "desk_rejection",
})


def _is_submission_invitation(note) -> bool:
    """Return True if the note's invitations do NOT look like review/comment/rebuttal/decision."""
    inv_text = _invs_text(note)
    return not any(tok in inv_text for tok in _NON_SUBMISSION_INV_TOKENS)


def _note_content_dict(note) -> dict:
    c = getattr(note, "content", None)
    if isinstance(c, dict):
        return c
    return {}


def _select_submission_note(forum_id: str, thread: list, log) -> object | None:
    """
    Choose the root submission note from a forum thread using a priority order:

    1. Exact id match (n.id == forum_id) — same as the old behavior.
    2. forum matches, has title+abstract, invitation looks like a submission.
    3. forum matches, has title, invitation looks like a submission.
    4. Any note in the thread with a submission-like invitation (last resort).
    5. Returns None so the caller can raise a descriptive error.
    """
    # Priority 1 — exact id match
    for n in thread:
        if n.id == forum_id:
            log.debug("fallback: selected note %s via exact id match", n.id)
            return n

    # Build a filtered list: notes whose forum == forum_id
    same_forum = [n for n in thread if getattr(n, "forum", None) == forum_id]

    # Priority 2 — has title + abstract + submission invitation
    for n in same_forum:
        if not _is_submission_invitation(n):
            continue
        c = _note_content_dict(n)
        has_title = bool(_unwrap(c.get("title")))
        has_abstract = bool(_unwrap(c.get("abstract")))
        if has_title and has_abstract:
            log.info(
                "fallback: selected note %s (forum match + title + abstract, inv=%r)",
                n.id, getattr(n, "invitations", None),
            )
            return n

    # Priority 3 — has title + submission invitation
    for n in same_forum:
        if not _is_submission_invitation(n):
            continue
        c = _note_content_dict(n)
        if bool(_unwrap(c.get("title"))):
            log.info(
                "fallback: selected note %s (forum match + title only, inv=%r)",
                n.id, getattr(n, "invitations", None),
            )
            return n

    # Priority 4 — any submission-like note in the whole thread
    for n in thread:
        if _is_submission_invitation(n):
            log.warning(
                "fallback: conservatively selected note %s from full thread "
                "(no closer match found, inv=%r)",
                n.id, getattr(n, "invitations", None),
            )
            return n

    return None


def fetch_forum_data(forum_id: str) -> dict:
    log = logging.getLogger(__name__)
    client = build_client()

    # 1) Get submission note — fast path first; fall back to forum-thread search
    #    when the direct note fetch is restricted (some venues return 403 for
    #    get_note even though the forum thread itself is accessible).
    try:
        note = client.get_note(id=forum_id)
    except openreview.OpenReviewException as _exc:
        _err = _exc.args[0] if _exc.args else {}
        if isinstance(_err, dict) and _err.get("status") == 403:
            log.warning(
                "get_note(%s) returned 403 (%s); trying forum-thread fallback.",
                forum_id,
                _err.get("name", "ForbiddenError"),
            )
            _thread = client.get_all_notes(forum=forum_id)
            note = _select_submission_note(forum_id, _thread, log)
            if note is None:
                _debug_notes = [n for n in _thread if getattr(n, "forum", None) == forum_id][:5]
                for _dn in _debug_notes:
                    _dc = getattr(_dn, "content", None) or {}
                    log.warning(
                        "fallback candidate: id=%s forum=%s invitation=%r has_title=%s has_abstract=%s",
                        _dn.id,
                        getattr(_dn, "forum", None),
                        getattr(_dn, "invitations", None),
                        bool(_unwrap(_dc.get("title")) if isinstance(_dc, dict) else False),
                        bool(_unwrap(_dc.get("abstract")) if isinstance(_dc, dict) else False),
                    )
                raise ValueError(
                    f"Forum {forum_id!r}: direct fetch returned 403 and the submission "
                    f"note is absent from the forum thread ({len(_thread)} reply notes "
                    f"returned — the paper is likely under access restriction at this venue)."
                ) from _exc
        else:
            raise
    paper_id = f"openreview::{forum_id}"

    # 2) Determine venue_id (OpenReview sometimes stores {"value": ...})
    venue_id = None
    if hasattr(note, "content") and note.content:
        venue_raw = note.content.get("venueid")
        if isinstance(venue_raw, dict):
            venue_id = venue_raw.get("value")
        else:
            venue_id = venue_raw

    if not venue_id:
        venue_id = DEFAULT_VENUE_ID or "unknown"
    venue_id = str(venue_id)
    venue_safe = _safe(venue_id)

    # 3) Download PDF attachment (best-effort)
    pdf_dir = PDF_ROOT / venue_safe
    pdf_dir.mkdir(parents=True, exist_ok=True)

    out_name = f"paper_{forum_id}.pdf"  # forum_id is safe-ish; paper_id contains '::'
    out_path = Path(pdf_dir) / out_name

    pdf_path, pdf_sha256, pdf_error = _download_pdf(client, note.id, out_path)
    if pdf_path is None:
        log.warning("PDF download failed for forum %s: %s", forum_id, pdf_error)

    # 4) Get thread replies
    replies = client.get_all_notes(forum=note.forum or note.id)
    log.info("Total %d replies in forum %s.", len(replies), forum_id)

    reviews: list[dict] = []
    rebuttals: list[dict] = []
    meta_reviews: list[dict] = []
    decision = None

    for r in replies:
        inv_text = _invs_text(r)  # invitations text (ONLY for type detection)

        # Robustly coerce content into a dict (avoid `or {}` swallowing non-dict content objects)
        content_obj = getattr(r, "content", None)
        if content_obj is None:
            content = {}
        elif isinstance(content_obj, dict):
            content = content_obj
        else:
            try:
                content = dict(content_obj)
            except Exception:
                try:
                    content = content_obj.to_json()
                except Exception:
                    content = {}

        rid = f"openreview::{r.id}"
        replyto = getattr(r, "replyto", None)
        signatures = getattr(r, "signatures", []) or []
        signatures_json = json.dumps(signatures)

        # --------
        # Decision (most specific; handle first)
        # --------
        is_decision = ("official_decision" in inv_text) or ("decision" in inv_text and "decision" in content)
        if decision is None and is_decision:
            dec_text = _first_present(content, ["decision", "recommendation", "final_decision"])
            if dec_text is not None:
                decision = {
                    "id": rid,
                    "forum": forum_id,
                    "decision": str(_unwrap(dec_text)),
                }
            continue

        # ------
        # Review (handle early; most important)
        # ------
        is_review = ("official_review" in inv_text) or ("review" in content)
        if is_review:
            rating = _first_present(content, ["rating", "score", "recommendation"])
            confidence = _first_present(content, ["confidence"])

            # Two common schemas:
            # (A) Workshop-style: content['review']
            # (B) Conference-style rubric: summary/strengths/weaknesses/questions/soundness/presentation/contribution
            review_text = _unwrap(content.get("review"))

            summary = _unwrap(content.get("summary"))
            strengths = _unwrap(content.get("strengths"))
            weaknesses = _unwrap(content.get("weaknesses"))
            questions = _unwrap(content.get("questions"))
            soundness = _unwrap(content.get("soundness"))
            presentation = _unwrap(content.get("presentation"))
            contribution = _unwrap(content.get("contribution"))

            # Build a main text blob used for extraction/reporting if `review` is not present
            if not review_text:
                parts = []
                if summary:
                    parts.append(f"SUMMARY:\n{summary}")
                if strengths:
                    parts.append(f"STRENGTHS:\n{strengths}")
                if weaknesses:
                    parts.append(f"WEAKNESSES:\n{weaknesses}")
                if questions:
                    parts.append(f"QUESTIONS:\n{questions}")
                if soundness:
                    parts.append(f"SOUNDNESS:\n{soundness}")
                if presentation:
                    parts.append(f"PRESENTATION:\n{presentation}")
                if contribution:
                    parts.append(f"CONTRIBUTION:\n{contribution}")
                review_text = "\n\n".join(parts).strip()

            if not review_text:
                log.warning(
                    "Skipping official review %s: empty text (keys=%s inv=%r)",
                    rid,
                    list(content.keys()),
                    inv_text[:160],
                )
                continue

            # If workshop-style (only `review` exists), store it in summary for v1 compatibility.
            # If conference-style, store rubric fields separately.
            stored_summary = summary if summary else review_text

            log.debug(
                "Review %s stored summary_len=%d strengths_len=%d weaknesses_len=%d",
                rid,
                len(stored_summary or ""),
                len(strengths or ""),
                len(weaknesses or ""),
            )

            # Capture ALL content fields as raw_fields for heterogeneous schema support.
            # Skip internal/structural keys; coerce to string after unwrapping.
            _SKIP_RAW_KEYS = frozenset({
                "pdf", "paper_link", "html", "_bibtex", "title", "authors",
                "abstract", "keywords", "venueid",
            })
            raw_fields: dict[str, str] = {}
            for k, v in content.items():
                if k.startswith("_") or k in _SKIP_RAW_KEYS:
                    continue
                unwrapped = _unwrap(v)
                if isinstance(unwrapped, str) and unwrapped.strip():
                    raw_fields[k] = unwrapped
                elif isinstance(unwrapped, (int, float)):
                    raw_fields[k] = str(unwrapped)
                # Lists and nested dicts are skipped intentionally

            reviews.append(
                {
                    "id": rid,
                    "paper_id": paper_id,
                    "forum": forum_id,
                    "replyto": replyto,
                    "signatures": signatures_json,
                    "rating": str(_unwrap(rating)) if rating is not None else None,
                    "confidence": str(_unwrap(confidence)) if confidence is not None else None,
                    "summary": stored_summary,
                    "strengths": strengths,
                    "weaknesses": weaknesses,
                    "questions": questions,
                    "soundness": soundness,
                    "presentation": presentation,
                    "contribution": contribution,
                    "raw_fields": json.dumps(raw_fields, ensure_ascii=False),
                }
            )
            continue

        # ------------
        # Meta-Review (conservative to avoid misclassifying reviews)
        # ------------
        is_meta = (
            ("meta_review" in inv_text)
            or ("metareview" in inv_text)
            or ("senior_area_chair" in inv_text)
            or ("official_meta_review" in inv_text)
        )
        if is_meta:
            rec = _first_present(content, ["recommendation", "recommendation_final", "overall_recommendation"])
            mr = _first_present(content, ["metareview", "meta_review", "summary", "comment", "additional_comments_on_reviewer_discussion"])
            conf = _first_present(content, ["confidence", "confidence_final"])

            if rec is not None or mr is not None:
                meta_reviews.append(
                    {
                        "id": rid,
                        "paper_id": paper_id,
                        "forum": forum_id,
                        "replyto": replyto,
                        "signatures": signatures_json,
                        "recommendation": str(_unwrap(rec)) if rec is not None else None,
                        "metareview": _unwrap(mr) if mr is not None else None,
                        "confidence": str(_unwrap(conf)) if conf is not None else None,
                    }
                )
            continue

        # ---------
        # Rebuttal / Author response
        # ---------
        is_rebuttal = (
            ("rebuttal" in inv_text)
            or ("official_rebuttal" in inv_text)
            or ("author_response" in inv_text)
            or ("author" in inv_text and "comment" in inv_text)
        )
        if is_rebuttal:
            comment = _first_present(content, ["comment", "rebuttal", "response", "author_response"])
            comment = _unwrap(comment) if comment is not None else None
            if comment:
                rebuttals.append(
                    {
                        "id": rid,
                        "paper_id": paper_id,
                        "forum": forum_id,
                        "replyto": replyto,
                        "signatures": signatures_json,
                        "comment": comment,
                    }
                )
            continue

        # Otherwise: ignore other note types (official comments, etc.)

    paper_dict = {
        "id": paper_id,
        "forum": forum_id,
        "number": None,
        "venue_id": venue_id,
        "title": _unwrap(note.content.get("title")) if hasattr(note, "content") else None,
        "authors": json.dumps(_unwrap(note.content.get("authors")) or []) if hasattr(note, "content") else "[]",
        "abstract": _unwrap(note.content.get("abstract")) if hasattr(note, "content") else None,
        "keywords": json.dumps(_unwrap(note.content.get("keywords")) or []) if hasattr(note, "content") else "[]",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "pdf_path": str(pdf_path) if pdf_error is None and pdf_path is not None else None,
        "pdf_sha256": pdf_sha256,
        "pdf_error": pdf_error,
        "reviews": reviews,
        "rebuttals": rebuttals,
        "meta_reviews": meta_reviews,
        "decision": decision,
    }

    return paper_dict