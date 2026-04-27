from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

from ..parsing.caption_extractor import extract_captions_from_spans
from .evidence_objects import EvidenceObject, FIGURE, TABLE, TEXT_CHUNK

log = logging.getLogger(__name__)

_NEARBY_LINE_WINDOW = 10
_MAX_CONTEXT_CHARS = 2200

# Caption validity filter — code/SQL context detection
_SQL_CODE_TOKENS_RE = re.compile(
    r"\b(SELECT|FROM\s+\w|WHERE\s+\w|(?:LEFT|RIGHT|INNER|OUTER)\s+JOIN|JOIN\s+\w|WITH\s+\w+\s+AS|"
    r"CASE\s+WHEN|GROUP\s+BY|ORDER\s+BY|INSERT\s+INTO|UPDATE\s+\w|CREATE\s+TABLE|ALTER\s+TABLE)\b"
    r"|(\{\{|\}\}|dbt\.)",
    re.IGNORECASE,
)
_SUSPICIOUS_SECTIONS_RE = re.compile(
    r"^(preamble)\b",  # conservative: only preamble; references/acknowledgments can have real tables
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Inline-reference rejection — patterns that indicate a span starting with
# "Table N" or "Figure N" is a mid-sentence prose reference, not a caption.
# ---------------------------------------------------------------------------

# A) Comma immediately after the label number:
#    "Table 8, respectively."  "Table 1, Table 2 and …"
_INLINE_COMMA_RE = re.compile(
    r"^(?:Table|Tab\.|Figure|Fig\.)\s+\d+(?:\.\d+)?\s*,",
    re.IGNORECASE,
)

# B) "respectively" anywhere within 80 chars of the label start:
#    "Table 1 and Table 2, respectively"  "Table 8, respectively."
_INLINE_RESPECTIVELY_RE = re.compile(
    r"^(?:Table|Tab\.|Figure|Fig\.)\s+\d+(?:\.\d+)?.{0,70}\brespectively\b",
    re.IGNORECASE,
)

# C) "Table X and Table Y" / "Figure X and Figure Y" list references:
#    "Table 1 and Table 2 demonstrate …"
_INLINE_CONJUNCTION_RE = re.compile(
    r"^(?:Table|Tab\.|Figure|Fig\.)\s+\d+(?:\.\d+)?\s+and\s+(?:Table|Tab\.|Figure|Fig\.)\s+\d+",
    re.IGNORECASE,
)

# D) Common 3rd-person verbs directly after the label (no colon/period separator):
#    "Table 1 compares …"  "Figure 3 shows …"
#    An optional single adverb (e.g. "also", "further") is allowed between the
#    label number and the verb to catch patterns like "Fig. 7 also shows …".
#    Verb list is intentionally conservative to avoid false rejections.
_INLINE_PROSE_VERB_RE = re.compile(
    r"^(?:Table|Tab\.|Figure|Fig\.)\s+\d+(?:\.\d+)?\s+"
    r"(?:(?:also|further|additionally|similarly|clearly|notably|explicitly|therefore|thus)\s+)?"
    r"(?:"
    r"compares?|shows?|lists?|presents?|summarizes?|summarises?"
    r"|contains?|depicts?|displays?|reports?|demonstrates?"
    r"|provides?|includes?|highlights?|reveals?|describes?"
    r"|details?|illustrates?|gives?|outlines?|plots?"
    r"|visualizes?|visualises?|indicates?|suggests?"
    r")\b",
    re.IGNORECASE,
)

# layout heuristics for object-region expansion
_MAX_TABLE_SCAN_LINES = 40
_MAX_FIGURE_SCAN_LINES = 28
_MAX_VERTICAL_GAP_FACTOR = 2.8  # gap threshold relative to caption height
_MIN_OBJECT_HEIGHT = 18.0

# P1.9 — table region validation constants
# Max fallback table height as a fraction of inferred page height.
# A region taller than this fraction of the page is almost certainly wrong.
_TABLE_MAX_HEIGHT_PAGE_FRACTION = 0.30   # 30% of page height
# Minimum fraction of lines that must be "short" (table row-like) for region to pass.
# Short = <= 80 chars.  Prose paragraphs have most lines >> 80 chars.
_TABLE_MIN_SHORT_LINE_FRACTION = 0.60
# If >= this many non-caption lines are long prose lines, reject as prose.
_TABLE_PROSE_LONG_LINE_THRESHOLD = 2     # 2 long lines → probably prose
_TABLE_PROSE_LONG_LINE_CHARS = 80        # chars threshold for "long"


def _section_number(section: str | None) -> str | None:
    if not section:
        return None
    m = re.match(r"^(\d+(?:\.\d+)*)\b", section.strip())
    return m.group(1) if m else None


def _clean_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()

def _keep_single_caption_paragraph(text: str | None) -> str:
    if not text:
        return ""

    raw = str(text).strip()
    if not raw:
        return ""

    raw = re.sub(r"\s+", " ", raw).strip()

    # Keep only from the first real figure/table prefix
    m = re.search(r"\b(Figure|Fig\.|Table|Tab\.)\s+\d+(?:\.\d+)*\b", raw, flags=re.I)
    if m:
        raw = raw[m.start():].strip()

    # Stop before obvious spillover into body text / next heading
    stop_patterns = [
        r"\s+\d+(?:\.\d+)*\s+[A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)*\b",  # section heading
        r"\s+(For each domain,|For each prompt,|We construct|We evaluate|In this section,|In our experiments,|To this end,|Recent efforts|Finally,|The following sections)\b",
    ]
    for pat in stop_patterns:
        m2 = re.search(pat, raw)
        if m2:
            raw = raw[:m2.start()].strip()
            break

    return raw

def _collect_figure_region_with_caption(
    spans: list[dict[str, Any]],
    *,
    cap: dict[str, Any],
) -> tuple[list[str], list[float] | None]:
    """
    Conservative figure region:
    include current figure caption + figure body immediately above it,
    but do not climb into previous figure captions, headings, or body paragraphs.
    """
    if not spans:
        return cap.get("span_ids") or [], cap.get("bbox")

    page_num = cap.get("page_num")
    caption_span_ids = cap.get("span_ids") or []
    caption_bbox = cap.get("bbox")
    cap_section = _clean_text(cap.get("section"))

    ordered = _sort_spans(spans)
    by_id = {s["id"]: s for s in ordered}
    caption_spans = [by_id[sid] for sid in caption_span_ids if sid in by_id]
    if not caption_spans:
        return caption_span_ids, caption_bbox

    page_spans = [s for s in ordered if s.get("page_num") == page_num]
    if not page_spans:
        return caption_span_ids, caption_bbox

    page_index_by_id = {s["id"]: i for i, s in enumerate(page_spans)}
    caption_indices = [page_index_by_id[s["id"]] for s in caption_spans if s["id"] in page_index_by_id]
    if not caption_indices:
        return caption_span_ids, caption_bbox

    cap_start = min(caption_indices)
    cap_end = max(caption_indices)

    cap_box = _bbox_union([s.get("bbox") for s in caption_spans]) or caption_bbox
    cap_h = max(_bbox_height(cap_box), _MIN_OBJECT_HEIGHT)

    selected: list[dict[str, Any]] = list(caption_spans)

    def span_top(s: dict[str, Any]) -> float:
        b = s.get("bbox")
        return float(b[1]) if b and len(b) == 4 else 0.0

    def span_bottom(s: dict[str, Any]) -> float:
        b = s.get("bbox")
        return float(b[3]) if b and len(b) == 4 else 0.0

    def span_left(s: dict[str, Any]) -> float:
        b = s.get("bbox")
        return float(b[0]) if b and len(b) == 4 else 0.0

    def span_right(s: dict[str, Any]) -> float:
        b = s.get("bbox")
        return float(b[2]) if b and len(b) == 4 else 0.0

    prev_top = min(span_top(s) for s in caption_spans)
    caption_left = min(span_left(s) for s in caption_spans)
    caption_right = max(span_right(s) for s in caption_spans)

    upward: list[dict[str, Any]] = []
    taken_up = 0
    i = cap_start - 1

    while i >= 0 and taken_up < _MAX_FIGURE_SCAN_LINES:
        s = page_spans[i]
        txt = _clean_text(s.get("text"))
        b = s.get("bbox")

        if not b or len(b) != 4:
            i -= 1
            continue

        if cap_section and _clean_text(s.get("section")) and _clean_text(s.get("section")) != cap_section:
            break

        # hard stops: previous object caption / heading / obvious body paragraph
        if _is_caption_like_text(txt):
            break
        if re.match(r"^\d+(?:\.\d+)*\s+[A-Z]", txt):
            break
        if re.match(
            r"^(For each|We construct|We evaluate|In this section|In our experiments|To this end|Recent efforts|Finally,|The following|To examine|We analyze|Figure \d+ demonstrates)\b",
            txt,
        ):
            break

        gap = max(0.0, prev_top - span_bottom(s))
        if gap > cap_h * 6.0:
            break

        # stay roughly in the same horizontal figure block
        left = span_left(s)
        right = span_right(s)
        if right < caption_left - 220 or left > caption_right + 220:
            break

        upward.append(s)
        prev_top = min(prev_top, span_top(s))
        taken_up += 1
        i -= 1

    upward.reverse()
    selected = upward + selected

    seen: set[str] = set()
    selected_ids: list[str] = []
    selected_boxes: list[list[float]] = []

    for s in selected:
        sid = s.get("id")
        if not sid or sid in seen:
            continue
        seen.add(sid)
        selected_ids.append(sid)
        if s.get("bbox"):
            selected_boxes.append(s["bbox"])

    if len(selected_ids) <= len(caption_span_ids):
        return caption_span_ids, caption_bbox

    object_bbox = _bbox_union(selected_boxes) or caption_bbox
    return selected_ids, object_bbox

def _truncate(text: str, limit: int = _MAX_CONTEXT_CHARS) -> str:
    text = _clean_text(text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _sort_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(spans, key=lambda s: (s.get("page_num", 10**9), s.get("span_index", 10**9)))


def _bbox_union(
    bboxes: list[list[float] | tuple[float, float, float, float] | None]
) -> list[float] | None:
    valid = [b for b in bboxes if b and len(b) == 4]
    if not valid:
        return None
    x0 = min(float(b[0]) for b in valid)
    y0 = min(float(b[1]) for b in valid)
    x1 = max(float(b[2]) for b in valid)
    y1 = max(float(b[3]) for b in valid)
    return [x0, y0, x1, y1]


def _bbox_height(bbox: list[float] | tuple[float, float, float, float] | None) -> float:
    if not bbox or len(bbox) != 4:
        return 0.0
    return max(0.0, float(bbox[3]) - float(bbox[1]))


def _is_caption_like_text(text: str | None) -> bool:
    t = _clean_text(text).lower()
    if not t:
        return False
    return bool(re.match(r"^(table|tab\.?|figure|fig\.?)\s*\d+", t))


def _collect_caption_context(
    spans: list[dict[str, Any]],
    *,
    caption_page: int,
    caption_span_ids: list[str],
    kind: str,
) -> str:
    """
    Pre-multimodal V1 strategy:
    - for tables: prefer caption + following nearby lines on the same page
    - for figures: prefer caption + some nearby lines before/after
    - stop when section changes
    """
    if not spans:
        return ""

    ordered = _sort_spans(spans)
    by_id = {s["id"]: s for s in ordered}
    caption_spans = [by_id[sid] for sid in caption_span_ids if sid in by_id]
    if not caption_spans:
        return ""

    cap_section = _clean_text(caption_spans[0].get("section")) if caption_spans else ""

    caption_indices = [ordered.index(s) for s in caption_spans]
    start_idx = min(caption_indices)
    end_idx = max(caption_indices)

    collected: list[str] = []

    if kind == "table":
        # Caption first
        for i in range(start_idx, end_idx + 1):
            txt = _clean_text(ordered[i].get("text"))
            if txt:
                collected.append(txt)

        # Then grab a window after caption, same page only.
        # Do NOT apply the section-change guard here: table column headers
        # (e.g. "Model", "Results") are short words that often get
        # misclassified as section headings by the heading heuristic,
        # which would incorrectly stop collection at the first data row.
        after_count = 0
        j = end_idx + 1
        while j < len(ordered) and after_count < _NEARBY_LINE_WINDOW:
            s = ordered[j]
            if s.get("page_num") != caption_page:
                break

            txt = _clean_text(s.get("text"))
            if txt:
                collected.append(txt)
                after_count += 1
            j += 1

    else:
        # figure: some before + caption + some after
        before_count = 0
        i = start_idx - 1
        before: list[str] = []
        while i >= 0 and before_count < (_NEARBY_LINE_WINDOW // 2):
            s = ordered[i]
            if s.get("page_num") != caption_page:
                break
            if cap_section and _clean_text(s.get("section")) and _clean_text(s.get("section")) != cap_section:
                break

            txt = _clean_text(s.get("text"))
            if txt:
                before.append(txt)
                before_count += 1
            i -= 1
        before.reverse()
        collected.extend(before)

        for i in range(start_idx, end_idx + 1):
            txt = _clean_text(ordered[i].get("text"))
            if txt:
                collected.append(txt)

        after_count = 0
        j = end_idx + 1
        while j < len(ordered) and after_count < (_NEARBY_LINE_WINDOW // 2):
            s = ordered[j]
            if s.get("page_num") != caption_page:
                break
            if cap_section and _clean_text(s.get("section")) and _clean_text(s.get("section")) != cap_section:
                break

            txt = _clean_text(s.get("text"))
            if txt:
                collected.append(txt)
                after_count += 1
            j += 1

    return _truncate(" ".join(collected))


def _build_object_retrieval_text(
    *,
    label: str | None,
    caption_text: str | None,
    section: str | None,
    context_text: str | None,
    kind: str,
) -> str:
    parts: list[str] = []

    if label:
        parts.append(label)

    if caption_text:
        parts.append(f"Caption: {_clean_text(caption_text)}")

    if section:
        parts.append(f"Section: {_clean_text(section)}")

    if context_text:
        if kind == "table":
            parts.append(f"Table text/context: {_clean_text(context_text)}")
        else:
            parts.append(f"Figure context: {_clean_text(context_text)}")

    return "\n".join(p for p in parts if p).strip()


def _collect_object_region(
    spans: list[dict[str, Any]],
    *,
    cap: dict[str, Any],
) -> tuple[list[str], list[float] | None]:
    """
    Expand from caption to approximate full table/figure region.

    Important:
    - table: caption is usually ABOVE table -> expand downward
    - figure: caption is usually BELOW figure -> expand upward first
    - stop when section changes
    """
    if not spans:
        return cap.get("span_ids") or [], cap.get("bbox")

    kind = cap["kind"]
    page_num = cap.get("page_num")
    caption_span_ids = cap.get("span_ids") or []
    caption_bbox = cap.get("bbox")
    cap_section = _clean_text(cap.get("section"))

    ordered = _sort_spans(spans)
    by_id = {s["id"]: s for s in ordered}
    caption_spans = [by_id[sid] for sid in caption_span_ids if sid in by_id]
    if not caption_spans:
        return caption_span_ids, caption_bbox

    # same-page spans only
    page_spans = [s for s in ordered if s.get("page_num") == page_num]
    if not page_spans:
        return caption_span_ids, caption_bbox

    page_index_by_id = {s["id"]: i for i, s in enumerate(page_spans)}
    caption_page_indices = [page_index_by_id[s["id"]] for s in caption_spans if s["id"] in page_index_by_id]
    if not caption_page_indices:
        return caption_span_ids, caption_bbox

    cap_start = min(caption_page_indices)
    cap_end = max(caption_page_indices)

    cap_box = _bbox_union([s.get("bbox") for s in caption_spans]) or caption_bbox
    cap_h = max(_bbox_height(cap_box), _MIN_OBJECT_HEIGHT)
    if kind == "table":
        max_gap = cap_h * 6.5
    else:
        max_gap = cap_h * 6.0


    selected: list[dict[str, Any]] = list(caption_spans)

    def span_top(s: dict[str, Any]) -> float:
        b = s.get("bbox")
        return float(b[1]) if b and len(b) == 4 else 0.0

    def span_bottom(s: dict[str, Any]) -> float:
        b = s.get("bbox")
        return float(b[3]) if b and len(b) == 4 else 0.0

    def valid_candidate_text(s: dict[str, Any]) -> bool:
        txt = _clean_text(s.get("text"))
        if not txt:
            return False
        if _is_caption_like_text(txt):
            return False
        return True

    if kind == "table":
        # Expand downward from caption
        prev_bottom = max(span_bottom(s) for s in caption_spans)
        taken = 0

        j = cap_end + 1
        while j < len(page_spans) and taken < _MAX_TABLE_SCAN_LINES:
            s = page_spans[j]
            b = s.get("bbox")
            if not b or len(b) != 4:
                j += 1
                continue

            if cap_section and _clean_text(s.get("section")) and _clean_text(s.get("section")) != cap_section:
                break

            gap = max(0.0, span_top(s) - prev_bottom)
            if gap > max_gap:
                break
            
            
            txt = _clean_text(s.get("text"))
            if re.match(
                r"^(For each domain,|For each prompt,|We construct|We evaluate|In this section,|In our experiments,|To this end,|Recent efforts|Finally,|The following sections)\b",
                txt,
            ):
                break

            if _is_caption_like_text(txt):
                break
            selected.append(s)
            prev_bottom = max(prev_bottom, span_bottom(s))
            taken += 1
            j += 1

    else:
        # figure: expand upward first (figure body usually above caption)
        prev_top = min(span_top(s) for s in caption_spans)
        taken_up = 0
        upward: list[dict[str, Any]] = []

        i = cap_start - 1
        while i >= 0 and taken_up < _MAX_FIGURE_SCAN_LINES:
            s = page_spans[i]
            b = s.get("bbox")
            if not b or len(b) != 4:
                i -= 1
                continue

            if cap_section and _clean_text(s.get("section")) and _clean_text(s.get("section")) != cap_section:
                break

            gap = max(0.0, prev_top - span_bottom(s))
            if gap > max_gap:
                break


            txt = _clean_text(s.get("text"))
            if _is_caption_like_text(txt):
                break

            # stop before normal body paragraph under the figure            
            if re.match(r"^(For each|We construct|We evaluate|In this section|In our experiments|To this end|Recent efforts|Finally,|The following)\b", txt):
                break

            upward.append(s)
            prev_top = min(prev_top, span_top(s))
            taken_up += 1
            i -= 1

        upward.reverse()
        selected = upward + selected

    # allow at most one short line immediately below caption
    prev_bottom = max(span_bottom(s) for s in caption_spans)
    taken_down = 0
    j = cap_end + 1
    while j < len(page_spans) and taken_down < 1:
        s = page_spans[j]
        txt = _clean_text(s.get("text"))
        b = s.get("bbox")

        if not b or len(b) != 4:
            j += 1
            continue

        if cap_section and _clean_text(s.get("section")) and _clean_text(s.get("section")) != cap_section:
            break

        if _is_caption_like_text(txt):
            break

        if re.match(r"^\d+(?:\.\d+)*\s+[A-Z]", txt):
            break

        gap = max(0.0, span_top(s) - prev_bottom)
        if gap > cap_h * 2.0:
            break

        selected.append(s)
        prev_bottom = max(prev_bottom, span_bottom(s))
        taken_down += 1
        j += 1


    # dedupe by id while preserving order
    seen: set[str] = set()
    selected_ids: list[str] = []
    selected_boxes: list[list[float]] = []

    for s in selected:
        sid = s.get("id")
        if not sid or sid in seen:
            continue
        seen.add(sid)
        selected_ids.append(sid)
        if s.get("bbox"):
            selected_boxes.append(s["bbox"])

    # If expansion failed, fall back to caption
    if len(selected_ids) <= len(caption_span_ids):
        return caption_span_ids, caption_bbox

    object_bbox = _bbox_union(selected_boxes) or caption_bbox
    return selected_ids, object_bbox


def _is_inline_reference_not_caption(caption_text: str) -> bool:
    """Return True when caption_text looks like a mid-sentence prose reference.

    These are cases where a PDF span *starts* with "Table N" or "Figure N" but
    the surrounding text reveals it is an inline citation rather than a real
    figure/table caption.  Detecting them early prevents fake evidence objects.

    Four high-confidence patterns are checked (all anchored at the start):

    A) Comma immediately after the label number — "Table 8, respectively."
    B) "respectively" within 80 chars of the start — "Table 1 and Table 2, respectively"
    C) Conjunction list reference — "Table 1 and Table 2 show …"
    D) 3rd-person prose verb after the label — "Table 1 compares …"

    The check is intentionally conservative: only patterns with very high
    confidence of being inline references are rejected.
    """
    t = _clean_text(caption_text)
    if not t:
        return False
    if _INLINE_COMMA_RE.match(t):
        return True
    if _INLINE_RESPECTIVELY_RE.match(t):
        return True
    if _INLINE_CONJUNCTION_RE.match(t):
        return True
    if _INLINE_PROSE_VERB_RE.match(t):
        return True
    return False


def _is_valid_caption_candidate(
    cap: dict[str, Any],
    spans: list[dict[str, Any]],
    all_numbers_for_kind: list[int],
) -> tuple[bool, str | None]:
    """Conservative identity-layer filter for caption candidates.

    Returns (is_valid, rejection_reason).  Rejects only when >= 2 independent
    suspicious signals are present — precision over recall.

    Signals checked:
      bare_caption        — no descriptive text after "Table N" / "Figure N"
      suspicious_section  — span is in a preamble-like section
      code_sql_context    — nearby spans contain SQL / template tokens (>= 2 hits)
      extreme_number      — table/figure number is >= 10× max of other candidates
    """
    signals: list[str] = []

    # --- Signal 1: bare caption (no meaningful description after label) ---
    caption_text = _clean_text(cap.get("caption_text") or "")
    label = _clean_text(cap.get("label") or "")
    body = caption_text
    if label and body.lower().startswith(label.lower()):
        body = body[len(label):].lstrip(":. ").strip()
    if len(body) < 5:
        signals.append("bare_caption")

    # --- Signal 2: suspicious section ---
    section = _clean_text(cap.get("section") or "")
    if section and _SUSPICIOUS_SECTIONS_RE.match(section):
        signals.append("suspicious_section")

    # --- Signal 3: nearby spans look like SQL / template code ---
    page = cap.get("page_num")
    cap_span_ids = set(cap.get("span_ids") or [])
    by_id = {s["id"]: s for s in spans}

    cap_indices = [
        by_id[sid].get("span_index", 0)
        for sid in cap_span_ids
        if sid in by_id
    ]
    if cap_indices and spans:
        lo = min(cap_indices)
        hi = max(cap_indices)
        nearby = [
            s for s in spans
            if s.get("page_num") == page
            and lo - 5 <= s.get("span_index", 0) <= hi + 20
        ]
        nearby_text = " ".join(_clean_text(s.get("text")) for s in nearby)
        sql_hits = len(_SQL_CODE_TOKENS_RE.findall(nearby_text))
        if sql_hits >= 2:
            signals.append(f"code_sql_context ({sql_hits} tokens)")

    # --- Signal 4: number wildly outside distribution for this paper ---
    try:
        number = int(str(cap.get("number") or "").split(".")[0])
    except (ValueError, AttributeError):
        number = None

    if number is not None and len(all_numbers_for_kind) >= 3:
        max_normal = max(all_numbers_for_kind)
        # Only flag if clearly extreme: >= 10x the largest normal number AND > 20
        if number > max_normal * 10 and number > 20:
            signals.append(f"extreme_number ({number}, max_normal={max_normal})")

    if len(signals) >= 2:
        return False, "; ".join(signals)
    return True, None


# ---------------------------------------------------------------------------
# P1.9 — Table region validation helpers
# ---------------------------------------------------------------------------

# Heading-like patterns: appendix sections (C.2.3), numbered sections (3.1),
# or short ALL-CAPS / Title-Case lines that look like section titles.
_HEADING_LIKE_RE = re.compile(
    r"^(?:[A-Z]\.?\d*\.)?(?:\d+\.)*\d+\s+[A-Z]"  # "3.1 Method", "C.2 Details"
    r"|^[A-Z]\.\d+(?:\.\d+)*\b"                    # "A.1", "B.2.3" alone
    r"|^Appendix\b",                                # bare "Appendix" line
    re.UNICODE,
)


def _is_heading_like_span(text: str) -> bool:
    """Return True if ``text`` looks like a section/appendix heading.

    Conservative — only matches clear numbered or appendix patterns.
    """
    t = text.strip()
    if not t or len(t) > 120:
        return False
    return bool(_HEADING_LIKE_RE.match(t))


def _looks_like_prose_block(lines: list[str]) -> bool:
    """Return True if the list of text lines looks like a prose paragraph.

    Rules (any one sufficient for rejection):
    1. First line is heading-like (section title immediately after caption).
    2. >= _TABLE_PROSE_LONG_LINE_THRESHOLD lines are longer than
       _TABLE_PROSE_LONG_LINE_CHARS — typical prose sentences.
    3. Short-line fraction < _TABLE_MIN_SHORT_LINE_FRACTION — not row-dense.
    """
    if not lines:
        return False

    # Rule 1: heading-like first line
    if _is_heading_like_span(lines[0]):
        return True

    # Rule 2 + 3: long-line count and short-line fraction
    long_count = sum(
        1 for ln in lines if len(ln) > _TABLE_PROSE_LONG_LINE_CHARS
    )
    if long_count >= _TABLE_PROSE_LONG_LINE_THRESHOLD:
        return True

    short_count = sum(
        1 for ln in lines if len(ln) <= _TABLE_PROSE_LONG_LINE_CHARS
    )
    total = len(lines)
    if total > 0 and short_count / total < _TABLE_MIN_SHORT_LINE_FRACTION:
        return True

    return False


def _validate_table_region(
    object_span_ids: list[str],
    caption_span_ids: list[str],
    object_bbox: list[float] | None,
    caption_bbox: list[float] | None,
    span_by_id: dict[str, dict],
    inferred_page_height: float,
) -> tuple[bool, str | None]:
    """Validate a candidate table expansion region (P1.9).

    Returns (valid: bool, rejection_reason: str | None).

    A region is REJECTED (→ degrade to caption-only) when:
      R1: No expansion occurred (object_span_ids == caption_span_ids).
      R2: Expanded bbox is taller than _TABLE_MAX_HEIGHT_PAGE_FRACTION of page.
      R3: Non-caption spans look like a prose block (_looks_like_prose_block).

    Fails open when metadata is missing: unknown spans are treated as short
    lines (conservative — prefer keeping a real table over false rejection).
    """
    cap_id_set = set(caption_span_ids)
    extra_ids = [sid for sid in object_span_ids if sid not in cap_id_set]

    # R1: no expansion — already handled upstream as caption_only; pass through
    if not extra_ids:
        return True, None

    # R2: height guardrail
    if inferred_page_height > 0 and object_bbox and len(object_bbox) == 4:
        expanded_height = max(0.0, float(object_bbox[3]) - float(object_bbox[1]))
        if expanded_height > _TABLE_MAX_HEIGHT_PAGE_FRACTION * inferred_page_height:
            return False, (
                f"table_height_exceeds_page_fraction "
                f"(height={expanded_height:.0f} "
                f"limit={_TABLE_MAX_HEIGHT_PAGE_FRACTION * inferred_page_height:.0f})"
            )

    # R3: prose block check on non-caption spans
    extra_texts = []
    for sid in extra_ids:
        span = span_by_id.get(sid)
        if span:
            t = _clean_text(span.get("text"))
            if t:
                extra_texts.append(t)

    if extra_texts and _looks_like_prose_block(extra_texts):
        first_line = extra_texts[0][:60] if extra_texts else ""
        return False, f"prose_block_detected (first_line={first_line!r})"

    return True, None


# ---------------------------------------------------------------------------
# Phase A — Figure crop generation
# ---------------------------------------------------------------------------

_FS_ILLEGAL_RE = re.compile(r'[<>:"/\\|?*]')


def _safe_fs_name(value: str) -> str:
    """Return a filesystem-safe version of *value* for use in paths.

    Replaces Windows-illegal characters (< > : " / \\ | ? *) with underscores,
    strips surrounding whitespace, and falls back to "unknown" if the result
    would be empty.
    """
    sanitized = _FS_ILLEGAL_RE.sub("_", value).strip()
    return sanitized or "unknown"


_CROP_PADDING_PT: int = 8   # padding in PDF points around the figure bbox
_CROP_RENDER_SCALE: float = 2.0  # 144 DPI
_CROP_MIN_DIM: float = 4.0  # minimum bbox dimension (PDF points)

# caption_only asset-recovery heuristic constants (asset crop only; never changes UI bbox)
_HEURISTIC_H_MARGIN_PT: float = 20.0          # horizontal expansion past caption edges
_HEURISTIC_EXPAND_UP_FRACTION: float = 0.35   # upward expansion as fraction of page height
_HEURISTIC_EXPAND_UP_MAX_PT: float = 280.0    # hard cap on upward expansion
_HEURISTIC_EXPAND_UP_MIN_PT: float = 60.0     # minimum upward expansion to try
_HEURISTIC_MIN_BOX_HEIGHT: float = 60.0       # reject if candidate box is smaller than this
_HEURISTIC_MIN_HEIGHT_RATIO: float = 3.0      # candidate must be >= N× caption height
_HEURISTIC_PROSE_LONG_LINE_CHARS: int = 80    # char threshold for "long" (prose-like) line
_HEURISTIC_PROSE_REJECT_LONG_LINES: int = 3   # >= this many long lines → reject as prose


def _crop_figure_to_file(
    pdf_path: str | Path,
    page: int,
    bbox: list[float],
    evidence_id: str,
    paper_id: str,
) -> str | None:
    """Render a figure region from a PDF page and save it as PNG.

    Returns the saved file path on success, None on any failure.
    Never raises — all errors degrade gracefully.
    """
    if not pdf_path or not bbox or len(bbox) != 4:
        return None

    x0, y0, x1, y1 = bbox
    if (x1 - x0) < _CROP_MIN_DIM or (y1 - y0) < _CROP_MIN_DIM:
        return None

    # Apply padding
    x0 = max(0.0, x0 - _CROP_PADDING_PT)
    y0 = max(0.0, y0 - _CROP_PADDING_PT)
    x1 = x1 + _CROP_PADDING_PT
    y1 = y1 + _CROP_PADDING_PT

    try:
        import fitz
    except ImportError:
        log.debug("[figure-crop] fitz not available — skipping crop for %s", evidence_id)
        return None

    try:
        doc = fitz.open(str(pdf_path))
        if page < 1 or page > len(doc):
            doc.close()
            log.warning("[figure-crop] page %s out of range for %s", page, evidence_id)
            return None
        fitz_page = doc[page - 1]
        clip = fitz.Rect(x0, y0, x1, y1)
        mat = fitz.Matrix(_CROP_RENDER_SCALE, _CROP_RENDER_SCALE)
        pix = fitz_page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)
        img_bytes: bytes = pix.tobytes("png")
        doc.close()
    except Exception as exc:
        log.warning("[figure-crop] render failed evidence_id=%s page=%s: %s", evidence_id, page, exc)
        return None

    try:
        from gsr.config import WORKSPACE_DIR
        safe_paper_id = _safe_fs_name(paper_id)
        safe_evidence_id = _safe_fs_name(evidence_id)
        out_dir: Path = WORKSPACE_DIR / "assets" / "figures" / safe_paper_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{safe_evidence_id}.png"
        out_path.write_bytes(img_bytes)
        log.debug(
            "[figure-crop] saved evidence_id=%s paper_id=%s path=%s",
            evidence_id, paper_id, out_path,
        )
        return str(out_path)
    except Exception as exc:
        log.warning("[figure-crop] save failed evidence_id=%s: %s", evidence_id, exc)
        return None


def _try_caption_expand_heuristic(
    caption_bbox: list[float],
    page_spans: list[dict[str, Any]],
    caption_span_id_set: set[str],
    page_height: float,
    page_width: float,
) -> tuple[list[float] | None, str]:
    """Asset-only heuristic: attempt to recover a crop bbox for a caption_only figure.

    Assumes the figure body is above the caption. Builds a candidate rectangle by
    expanding upward from the caption top, then runs cheap plausibility checks.

    Returns (candidate_bbox, reason).  bbox is None when plausibility fails.
    Never raises. Does NOT modify the evidence/UI bbox — asset crop use only.
    """
    if not caption_bbox or len(caption_bbox) != 4:
        return None, "no_caption_bbox"

    cap_x0, cap_y0, cap_x1, cap_y1 = [float(v) for v in caption_bbox]
    cap_h = max(cap_y1 - cap_y0, 1.0)

    if page_height <= 0.0:
        return None, "unknown_page_height"

    expand_up = min(
        _HEURISTIC_EXPAND_UP_MAX_PT,
        max(_HEURISTIC_EXPAND_UP_MIN_PT, page_height * _HEURISTIC_EXPAND_UP_FRACTION),
    )

    if page_width <= 0.0:
        xs = [float(s["bbox"][2]) for s in page_spans if s.get("bbox") and len(s["bbox"]) == 4]
        page_width = max(xs, default=0.0)
    if page_width <= 0.0:
        return None, "unknown_page_width"

    cand_x0 = max(0.0, cap_x0 - _HEURISTIC_H_MARGIN_PT)
    cand_x1 = min(page_width, cap_x1 + _HEURISTIC_H_MARGIN_PT)
    cand_y0 = max(0.0, cap_y0 - expand_up)
    cand_y1 = cap_y1  # include caption line at bottom of crop

    cand_w = cand_x1 - cand_x0
    cand_h = cand_y1 - cand_y0

    # Check 1: degenerate box
    if cand_w < 40.0 or cand_h < _HEURISTIC_MIN_BOX_HEIGHT:
        return None, f"degenerate_box(w={cand_w:.0f},h={cand_h:.0f})"

    # Check 2: candidate must be substantially taller than the caption alone
    if cand_h < cap_h * _HEURISTIC_MIN_HEIGHT_RATIO:
        return None, f"not_larger_than_caption(ratio={cand_h / cap_h:.1f})"

    # Check 3: text-density check — reject if above-caption region is prose-heavy
    above_spans = [
        s for s in page_spans
        if s.get("id") not in caption_span_id_set
        and s.get("bbox") and len(s["bbox"]) == 4
        and float(s["bbox"][1]) >= cand_y0 - 2.0
        and float(s["bbox"][3]) <= cap_y0 + 2.0
        and float(s["bbox"][0]) >= cand_x0 - 10.0
        and float(s["bbox"][2]) <= cand_x1 + 10.0
    ]
    long_lines = sum(
        1 for s in above_spans
        if len(_clean_text(s.get("text"))) > _HEURISTIC_PROSE_LONG_LINE_CHARS
    )
    if long_lines >= _HEURISTIC_PROSE_REJECT_LONG_LINES:
        return None, f"prose_dense(long_lines={long_lines})"

    return [cand_x0, cand_y0, cand_x1, cand_y1], "ok"


def _classify_bbox_confidence(
    object_span_ids: list[str],
    caption_span_ids: list[str],
) -> str:
    """Classify how reliable the evidence object's bounding box is.

    "high"         — meaningful expansion beyond caption (>= 5 object spans)
    "inferred"     — some expansion happened, but small (> caption but < 5 spans)
    "caption_only" — no expansion; bbox covers only the caption anchor
    """
    n_obj = len(object_span_ids)
    n_cap = len(caption_span_ids)

    if n_obj <= n_cap:
        return "caption_only"
    if n_obj >= 5:
        return "high"
    return "inferred"


def _build_docling_table_lookup(docling_tables: list) -> dict:
    """Build two-level lookup from Docling tables for caption-defined enrichment.

    Returns a dict with:
      "by_label_page": (label_norm, page) → NormalizedTable  (exact match)
      "by_page":       page → list[NormalizedTable]           (page-only fallback)

    Only tables whose label looks like a real caption label (e.g. "table 4") are
    indexed in by_label_page.  Docling-assigned fallback labels like "Table 495"
    are NOT indexed there — they would never match a caption-defined object and
    their absence keeps the lookup clean.
    """
    by_label_page: dict[tuple[str, int], Any] = {}
    by_page: dict[int, list[Any]] = {}

    # Pattern for real caption labels: "table 4", "table 4.3", "figure 2.1"
    _real_label_re = re.compile(
        r"^(table|tab\.?|figure|fig\.?)\s+\d+(?:\.\d+){0,2}$", re.IGNORECASE
    )

    for t in docling_tables or []:
        label_raw = _clean_text(t.label or "")
        label_norm = label_raw.lower()
        page = t.page or 0

        if label_norm and _real_label_re.match(label_norm):
            by_label_page[(label_norm, page)] = t

        by_page.setdefault(page, []).append(t)

    return {"by_label_page": by_label_page, "by_page": by_page}


def _is_good_docling_table_text(
    text: str | None,
    caption_text: str | None,
) -> tuple[bool, str | None]:
    """Quality gate for Docling table content_text.

    Returns (passes: bool, rejection_reason: str | None).
    A Docling table is considered good only when it is meaningfully better
    than the caption-only fallback.

    Rules (conservative — precision over recall):
    - text must exist and be >= 80 chars after cleaning
    - text must not be nearly identical to caption_text
    - at least one of:
        * contains a pipe character (markdown table indicator)
        * has >= 3 non-empty lines (multi-row structure)
        * length > 1.8x caption length (substantially more content)
    """
    t = _clean_text(text)
    if not t:
        return False, "empty_text"

    if len(t) < 80:
        return False, f"too_short ({len(t)} chars)"

    cap = _clean_text(caption_text)
    if cap and t == cap:
        return False, "identical_to_caption"

    has_pipe = "|" in t
    non_empty_lines = [ln for ln in t.split("\n") if ln.strip()]
    has_multirow = len(non_empty_lines) >= 3
    substantially_larger = bool(cap) and len(t) > len(cap) * 1.8

    if not (has_pipe or has_multirow or substantially_larger):
        return False, "not_substantially_better_than_caption"

    return True, None


def build_evidence_objects_for_paper(
    *,
    paper_id: str,
    chunks: list[dict[str, Any]],
    spans: list[dict[str, Any]] | None = None,
    docling_tables: list | None = None,
    pdf_path: str | None = None,
    use_figure_ocr: bool = False,
) -> list[EvidenceObject]:
    """Build evidence objects for one paper.

    Args:
        paper_id: Paper identifier.
        chunks: V2 span-window chunks from chunk_paper_v2_from_spans().
        spans: V2 PyMuPDF spans — source of truth for bbox/grounding.
        docling_tables: Optional list of NormalizedTable objects from Docling.
            When provided, table evidence objects use Docling content_text as the
            primary substantive content (structured rows/markdown). PyMuPDF spans
            remain the grounding layer (bbox, span_ids). caption_extractor is the
            fallback when no Docling match exists for a given label+page.
        pdf_path: Path to the source PDF file.  Required when use_figure_ocr=True.
        use_figure_ocr: Phase 1 figure text recovery toggle (A/B evaluation flag).
            If True and pdf_path is provided, run LightOnOCR-2 on each canonical
            figure object whose bbox_confidence is "high" or "inferred".
            Results are stored in metadata using the figure_ocr_* schema.
    """
    # Resolve OCR availability once, outside the loop.
    _do_ocr = False
    if use_figure_ocr and pdf_path:
        from ..vision import ocr_lighton as _ocr_mod
        from ..vision.ocr_lighton import lighton_ocr_available
        _OCR_MAX_NEW_TOKENS: int = _ocr_mod._DEFAULT_MAX_NEW_TOKENS
        _OCR_TIMEOUT_SECONDS: float = _ocr_mod._DEFAULT_TIMEOUT_SECONDS
        _do_ocr = lighton_ocr_available()
        if not _do_ocr:
            log.warning(
                "use_figure_ocr=True but lightonai is not installed — "
                "figure OCR will be skipped. Run: pip install lightonai"
            )
        else:
            log.info(
                "[figure-ocr] guardrails: max_new_tokens=%d timeout_s=%.0f crop_max_side=1536",
                _OCR_MAX_NEW_TOKENS, _OCR_TIMEOUT_SECONDS,
            )
    out: list[EvidenceObject] = []

    # 1) text chunks remain first-class evidence objects
    for chunk in chunks:
        text = _clean_text(chunk.get("text"))
        out.append(
            EvidenceObject(
                id=chunk["id"],
                paper_id=paper_id,
                object_type=TEXT_CHUNK,
                retrieval_text=text,
                page=chunk.get("page"),
                page_start=chunk.get("page_start"),
                page_end=chunk.get("page_end"),
                section=chunk.get("section"),
                section_number=_section_number(chunk.get("section")),
                content_text=text,
                span_ids=chunk.get("span_ids"),
                metadata={"chunk_index": chunk.get("chunk_index")},
            )
        )

    # 2) caption-derived table / figure objects
    if spans:
        # P1.9 — pre-build span lookup and per-page heights for table validation
        _span_by_id: dict[str, dict] = {s["id"]: s for s in spans if s.get("id")}
        _page_heights: dict[int, float] = {}
        _page_widths: dict[int, float] = {}
        for _s in spans:
            _pg = _s.get("page_num")
            _bb = _s.get("bbox")
            if _pg is not None and _bb and len(_bb) == 4:
                _y1 = float(_bb[3])
                if _y1 > _page_heights.get(_pg, 0.0):
                    _page_heights[_pg] = _y1
                _x1 = float(_bb[2])
                if _x1 > _page_widths.get(_pg, 0.0):
                    _page_widths[_pg] = _x1

        raw_captions = extract_captions_from_spans(spans)

        # Build number distribution for the extreme-number signal (per kind, before filtering).
        numbers_by_kind: dict[str, list[int]] = {"table": [], "figure": []}
        for _cap in raw_captions:
            try:
                _n = int(str(_cap.get("number") or "").split(".")[0])
                numbers_by_kind[_cap["kind"]].append(_n)
            except (ValueError, AttributeError):
                pass

        # Filter out false-positive caption candidates.
        captions: list[dict[str, Any]] = []
        rejected_log: list[tuple[str, str]] = []
        _rejected_inline = 0
        _fig_inline_rejected = 0
        for _cap in raw_captions:
            # Hard reject: inline prose references like "Table 8, respectively."
            # These are spans that start with "Table N" / "Figure N" but are
            # mid-sentence references, not real captions.  Skip entirely — do
            # NOT create an evidence object.
            _cap_text = _cap.get("caption_text") or ""
            if _is_inline_reference_not_caption(_cap_text):
                _rejected_inline += 1
                if _cap.get("kind") == "figure":
                    _fig_inline_rejected += 1
                rejected_log.append((_cap.get("label", "?"), "inline_reference"))
                log.debug(
                    "Caption candidate rejected (inline_reference): "
                    "label=%r page=%s text=%r",
                    _cap.get("label"),
                    _cap.get("page_num"),
                    _cap_text[:100],
                )
                continue

            valid, reason = _is_valid_caption_candidate(
                _cap, spans, numbers_by_kind[_cap["kind"]]
            )
            if valid:
                captions.append(_cap)
            else:
                rejected_log.append((_cap.get("label", "?"), reason or "unknown"))
                log.debug(
                    "Caption candidate rejected: label=%r page=%s reason=%s",
                    _cap.get("label"),
                    _cap.get("page_num"),
                    reason,
                )

        log.info(
            "Caption candidates for paper '%s': %d total, %d kept, %d rejected "
            "(inline_ref=%d%s)",
            paper_id,
            len(raw_captions),
            len(captions),
            len(rejected_log),
            _rejected_inline,
            f" top_other_rejected={[r for r in rejected_log if r[1] != 'inline_reference'][:3]}"
            if any(r[1] != "inline_reference" for r in rejected_log) else "",
        )

        # Light dedupe: same (kind, label) should not produce two evidence objects.
        # When duplicates exist, keep the candidate with more caption text — a real
        # caption ("Figure 7: Results on ...") is longer than a prose snippet that
        # slipped through the inline-reference filter.
        _deduped: list[dict[str, Any]] = []
        _seen_label_keys: dict[tuple[str, str], int] = {}
        _fig_deduped_dropped = 0
        for _cap in captions:
            _key = (_cap["kind"], (_cap.get("label") or "").lower())
            _existing_idx = _seen_label_keys.get(_key)
            if _existing_idx is None:
                _seen_label_keys[_key] = len(_deduped)
                _deduped.append(_cap)
            else:
                _existing = _deduped[_existing_idx]
                if len(_cap.get("caption_text") or "") > len(_existing.get("caption_text") or ""):
                    log.debug(
                        "Caption dedupe: replacing shorter candidate for label=%r "
                        "(was %d chars, now %d chars)",
                        _cap.get("label"),
                        len(_existing.get("caption_text") or ""),
                        len(_cap.get("caption_text") or ""),
                    )
                    _deduped[_existing_idx] = _cap
                else:
                    if _cap.get("kind") == "figure":
                        _fig_deduped_dropped += 1
                    log.debug(
                        "Caption dedupe: discarding shorter duplicate for label=%r "
                        "(%d chars vs kept %d chars)",
                        _cap.get("label"),
                        len(_cap.get("caption_text") or ""),
                        len(_existing.get("caption_text") or ""),
                    )
        if len(_deduped) < len(captions):
            log.info(
                "Caption dedupe for paper '%s': removed %d duplicate(s)",
                paper_id, len(captions) - len(_deduped),
            )
        captions = _deduped

        docling_lookup = _build_docling_table_lookup(docling_tables)

        # Per-figure OCR progress counters (only meaningful when use_figure_ocr=True)
        _n_fig_total = sum(1 for c in captions if c["kind"] == "figure") if use_figure_ocr else 0
        _fig_ocr_num = 0
        _ocr_phase_start = time.monotonic()
        _fig_crop_caption_only_skipped = 0
        _fig_crop_caption_only_attempted = 0
        _fig_crop_caption_only_recovered = 0

        for idx, cap in enumerate(captions):
            kind = cap["kind"]
            obj_type = TABLE if kind == "table" else FIGURE

            # IMPORTANT:
            # Keep caption text compact for UI card display.
            short_caption = _keep_single_caption_paragraph(cap.get("caption_text"))

            # Determine context_text for retrieval / verification.
            # Caption-first design:
            #   - evidence object identity (label, page, bbox, span_ids) always
            #     comes from the caption extractor / PyMuPDF grounding layer.
            #   - Docling is ONLY used to enrich content_text when it matches a
            #     caption-defined table by label+page (or page-only as fallback)
            #     AND passes the quality gate.
            #   - Docling-only tables (no matching caption object) are ignored.
            #   - Figures are not enriched from Docling (visual objects).
            context_text: str | None = None

            # Per-object audit metadata (populated below for tables).
            docling_meta: dict[str, Any] = {
                "source_used": "caption_extractor",
                "caption_label_raw": cap.get("label"),
                "docling_label_raw": None,
                "docling_match_mode": "none",
                "docling_quality_pass": None,
                "docling_rejected_reason": None,
                "type_conflict": False,
            }

            if kind == "table":
                cap_label_raw = _clean_text(cap.get("label") or "")
                cap_label_norm = cap_label_raw.lower()
                cap_page = cap.get("page_num") or 0

                # Keep table retrieval_text compact (label+caption+section only).
                # No eager context injection for tables in either branch — Docling
                # markdown is populated lazily by enrich_topk_tables_with_docling().
                context_text = None
                if docling_tables is not None:
                    docling_meta.update({
                        "docling_enriched": False,
                        "docling_source_page": cap_page or None,
                        "docling_match_mode": "none",
                        "docling_markdown_chars": 0,
                        "docling_error": None,
                    })

            retrieval_text = _build_object_retrieval_text(
                label=cap.get("label"),
                caption_text=short_caption,
                section=cap.get("section"),
                context_text=context_text,
                kind=kind,
            )

            # IMPORTANT:
            # PDF highlight should cover object body + caption, not caption alone.
            if kind == "figure":
                object_span_ids, object_bbox = _collect_figure_region_with_caption(
                    spans,
                    cap=cap,
                )
            else:
                object_span_ids, object_bbox = _collect_object_region(
                    spans,
                    cap=cap,
                )

                # P1.9 — validate table expansion region before accepting it.
                # If the collected region looks like prose / a next section /
                # is too tall, degrade to caption-only to prevent fake table boxes.
                cap_page = cap.get("page_num") or 0
                _ph = _page_heights.get(cap_page, 0.0)
                _tbl_valid, _tbl_reject_reason = _validate_table_region(
                    object_span_ids=object_span_ids,
                    caption_span_ids=cap.get("span_ids") or [],
                    object_bbox=object_bbox,
                    caption_bbox=cap.get("bbox"),
                    span_by_id=_span_by_id,
                    inferred_page_height=_ph,
                )
                if not _tbl_valid:
                    log.debug(
                        "P1.9 table region rejected — degrading to caption-only: "
                        "label=%r page=%s reason=%s",
                        cap.get("label"), cap_page, _tbl_reject_reason,
                    )
                    object_span_ids = cap.get("span_ids") or []
                    object_bbox = cap.get("bbox")
                    # Record rejection in docling_meta for audit (non-breaking addition)
                    docling_meta["table_region_validation"] = "rejected"
                    docling_meta["table_fallback_rejected_reason"] = _tbl_reject_reason
                else:
                    docling_meta["table_region_validation"] = "passed"

            # Fallback conservatively to caption if region inference fails
            if not object_span_ids:
                object_span_ids = cap.get("span_ids") or []
            if not object_bbox:
                object_bbox = cap.get("bbox")

            bbox_confidence = _classify_bbox_confidence(
                object_span_ids, cap.get("span_ids") or []
            )

            # --- LightOnOCR-2 figure text recovery (Phase 1) ---
            # Every figure gets explicit figure_ocr_* metadata — even when skipped.
            # This ensures audit reporting is always reliable regardless of mode.
            # Toggle: use_figure_ocr=True (A/B evaluation flag — do not auto-enable).
            ocr_meta: dict[str, Any] = {}
            if kind == "figure":
                if use_figure_ocr:
                    _fig_ocr_num += 1
                    log.info(
                        "[figure-ocr] (%d/%d) %s page=%s bbox_conf=%s",
                        _fig_ocr_num, _n_fig_total,
                        cap.get("label", "?"), cap.get("page_num", "?"), bbox_confidence,
                    )

                if not use_figure_ocr:
                    ocr_meta = {
                        "figure_ocr_attempted": False,
                        "figure_ocr_skip_reason": "disabled",
                        "figure_ocr_source": "lightonocr2",
                    }
                elif not _do_ocr:
                    # use_figure_ocr=True but dependency missing or pdf_path not provided
                    ocr_meta = {
                        "figure_ocr_attempted": False,
                        "figure_ocr_skip_reason": "dependency_missing",
                        "figure_ocr_source": "lightonocr2",
                    }
                    log.info(
                        "[figure-ocr] (%d/%d) skipped reason=dependency_missing",
                        _fig_ocr_num, _n_fig_total,
                    )
                elif bbox_confidence == "caption_only":
                    ocr_meta = {
                        "figure_ocr_attempted": False,
                        "figure_ocr_skip_reason": "caption_only",
                        "figure_ocr_source": "lightonocr2",
                    }
                    log.info(
                        "[figure-ocr] (%d/%d) skipped reason=caption_only",
                        _fig_ocr_num, _n_fig_total,
                    )
                else:
                    from ..vision.ocr_lighton import ocr_figure_region
                    _fig_t0 = time.monotonic()
                    ocr_result = ocr_figure_region(
                        pdf_path,  # type: ignore[arg-type]  # guarded by _do_ocr
                        page=cap.get("page_num") or 1,
                        bbox=object_bbox,
                        caption_text=short_caption or None,
                        max_new_tokens=_OCR_MAX_NEW_TOKENS,
                        timeout_seconds=_OCR_TIMEOUT_SECONDS,
                    )
                    _fig_elapsed = time.monotonic() - _fig_t0
                    # ocr_result already uses the figure_ocr_* schema from ocr_lighton.py
                    ocr_meta = ocr_result
                    # Persist elapsed and outcome into metadata for audit reporting.
                    ocr_meta["figure_ocr_elapsed_s"] = round(_fig_elapsed, 2)
                    _label = cap.get("label", "?")
                    _page = cap.get("page_num", "?")
                    if ocr_meta.get("figure_ocr_text"):
                        ocr_meta["figure_ocr_outcome"] = "accepted"
                        log.info(
                            "[figure-ocr] (%d/%d) %s page=%s bbox=%s elapsed=%.1fs "
                            "outcome=accepted chars=%d",
                            _fig_ocr_num, _n_fig_total,
                            _label, _page, bbox_confidence,
                            _fig_elapsed, len(ocr_meta["figure_ocr_text"]),
                        )
                    elif ocr_meta.get("figure_ocr_skip_reason") == "inference_failed":
                        ocr_meta["figure_ocr_outcome"] = "inference_failed"
                        _err_type = ocr_meta.get("figure_ocr_error_type", "")
                        log.info(
                            "[figure-ocr] (%d/%d) %s page=%s bbox=%s elapsed=%.1fs "
                            "outcome=inference_failed error_type=%s",
                            _fig_ocr_num, _n_fig_total,
                            _label, _page, bbox_confidence,
                            _fig_elapsed, _err_type,
                        )
                    else:
                        ocr_meta["figure_ocr_outcome"] = "rejected_after_inference"
                        _skip_reason = ocr_meta.get("figure_ocr_skip_reason") or "unknown"
                        log.info(
                            "[figure-ocr] (%d/%d) %s page=%s bbox=%s elapsed=%.1fs "
                            "outcome=rejected_after_inference reason=%s",
                            _fig_ocr_num, _n_fig_total,
                            _label, _page, bbox_confidence,
                            _fig_elapsed, _skip_reason,
                        )

            # IMPORTANT:
            # content_text should stay compact for evidence cards.
            # All table evidence gets empty content_text at build time; it is
            # populated lazily by enrich_topk_tables_with_docling() before verification
            # (falls back to caption_text in load_cached_evidence_mixed if unenriched).
            if kind == "table":
                content_text = ""
            else:
                content_text = retrieval_text

            # --- Phase A: Figure crop generation ---
            # Only for figures; tables are excluded in this phase.
            # caption_only figures are skipped — the bbox covers only the caption
            # text line, which produces a misleading narrow crop with no figure content.
            # Failures degrade silently — text-only evidence is always preserved.
            _crop_asset_path: str | None = None
            _crop_meta: dict[str, Any] = {}
            if kind == "figure" and pdf_path and object_bbox:
                _evidence_id = f"{paper_id}_{kind}_{idx}"
                _fig_page = cap.get("page_num") or 0
                if bbox_confidence == "caption_only":
                    # Heuristic recovery: try to build an asset crop from caption bbox
                    # without touching the evidence/UI bbox.
                    _fig_crop_caption_only_attempted += 1
                    _heur_page_spans = [s for s in spans if s.get("page_num") == _fig_page]
                    _heur_cap_ids = set(cap.get("span_ids") or [])
                    _heur_bbox, _heur_reason = _try_caption_expand_heuristic(
                        caption_bbox=cap.get("bbox"),
                        page_spans=_heur_page_spans,
                        caption_span_id_set=_heur_cap_ids,
                        page_height=_page_heights.get(_fig_page, 0.0),
                        page_width=_page_widths.get(_fig_page, 0.0),
                    )
                    if _heur_bbox is not None:
                        _crop_asset_path = _crop_figure_to_file(
                            pdf_path=pdf_path,
                            page=_fig_page,
                            bbox=_heur_bbox,
                            evidence_id=_evidence_id,
                            paper_id=paper_id,
                        )
                        if _crop_asset_path:
                            _fig_crop_caption_only_recovered += 1
                            _crop_meta = {
                                "has_image": True,
                                "crop_source": "caption_expand_heuristic",
                                "crop_quality": "low",
                                "crop_padding": _CROP_PADDING_PT,
                                "asset_bbox_source": "caption_expand_heuristic",
                                "asset_recovered_from_caption_only": True,
                            }
                            log.debug(
                                "[figure-crop] heuristic_recovered evidence_id=%s page=%s",
                                _evidence_id, _fig_page,
                            )
                        else:
                            _crop_meta = {
                                "has_image": False,
                                "crop_source": "caption_anchor",
                                "crop_quality": "low",
                                "crop_padding": _CROP_PADDING_PT,
                                "crop_skipped_reason": "caption_only",
                                "asset_bbox_source": "caption_expand_heuristic",
                                "asset_recovered_from_caption_only": False,
                            }
                    else:
                        _fig_crop_caption_only_skipped += 1
                        log.debug(
                            "[figure-crop] skipped evidence_id=%s page=%s: "
                            "caption_only heuristic_reason=%s",
                            _evidence_id, _fig_page, _heur_reason,
                        )
                        _crop_meta = {
                            "has_image": False,
                            "crop_source": "caption_anchor",
                            "crop_quality": "low",
                            "crop_padding": _CROP_PADDING_PT,
                            "crop_skipped_reason": "caption_only",
                        }
                else:
                    _crop_asset_path = _crop_figure_to_file(
                        pdf_path=pdf_path,
                        page=_fig_page,
                        bbox=object_bbox,
                        evidence_id=_evidence_id,
                        paper_id=paper_id,
                    )
                    _crop_quality = {
                        "high": "high",
                        "inferred": "medium",
                    }.get(bbox_confidence, "low")
                    _crop_source = {
                        "high": "object_bbox",
                        "inferred": "inferred_bbox",
                    }.get(bbox_confidence, "caption_anchor")
                    _crop_meta = {
                        "has_image": _crop_asset_path is not None,
                        "crop_source": _crop_source,
                        "crop_quality": _crop_quality if _crop_asset_path else "low",
                        "crop_padding": _CROP_PADDING_PT,
                    }
                    log.debug(
                        "[figure-crop] evidence_id=%s page=%s saved=%s quality=%s source=%s",
                        _evidence_id, _fig_page, _crop_asset_path is not None,
                        _crop_meta["crop_quality"], _crop_source,
                    )

            out.append(
                EvidenceObject(
                    id=f"{paper_id}_{kind}_{idx}",
                    paper_id=paper_id,
                    object_type=obj_type,
                    label=cap.get("label"),
                    page=cap.get("page_num"),
                    page_start=cap.get("page_num"),
                    page_end=cap.get("page_num"),
                    section=cap.get("section"),
                    section_number=cap.get("section_number"),
                    caption_text=short_caption,
                    retrieval_text=retrieval_text,
                    content_text=content_text,
                    bbox=object_bbox,
                    span_ids=object_span_ids,
                    asset_path=_crop_asset_path,
                    metadata={
                        "number": cap.get("number"),
                        "kind": kind,
                        # Docling-specific audit fields (populated for tables only).
                        **(docling_meta if kind == "table" else {"source_used": "caption_extractor"}),
                        "caption_bbox": cap.get("bbox"),
                        "caption_span_ids": cap.get("span_ids"),
                        # Grounding confidence: "high" / "inferred" / "caption_only"
                        "object_bbox": object_bbox,
                        "object_bbox_inferred": bbox_confidence != "high",
                        "object_bbox_confidence": bbox_confidence,
                        # LightOnOCR-2 enrichment (figures only)
                        **ocr_meta,
                        # Phase A: figure crop signals (figures only; empty for tables/text)
                        **_crop_meta,
                    },
                )
            )

        # Aggregate OCR logging for this paper (figures only).
        if use_figure_ocr:
            fig_objects = [e for e in out if e.object_type == FIGURE]
            n_figs = len(fig_objects)
            if n_figs:
                from collections import Counter
                statuses = Counter(
                    (e.metadata or {}).get("figure_ocr_skip_reason") or (
                        "accepted" if (e.metadata or {}).get("figure_ocr_text") else "rejected_after_inference"
                    ) if (e.metadata or {}).get("figure_ocr_attempted") else (
                        (e.metadata or {}).get("figure_ocr_skip_reason") or "unknown_skip"
                    )
                    for e in fig_objects
                )
                n_attempted = sum(1 for e in fig_objects if (e.metadata or {}).get("figure_ocr_attempted"))
                n_accepted = sum(1 for e in fig_objects if (e.metadata or {}).get("figure_ocr_text"))
                n_rejected = n_attempted - n_accepted
                n_never_attempted = n_figs - n_attempted
                n_inference_failed = sum(
                    1 for e in fig_objects
                    if (e.metadata or {}).get("figure_ocr_skip_reason") == "inference_failed"
                )
                n_timeout = sum(
                    1 for e in fig_objects
                    if (e.metadata or {}).get("figure_ocr_error_type") == "TimeoutError"
                )
                _total_elapsed = time.monotonic() - _ocr_phase_start
                _timeout_suffix = f" timeout={n_timeout}" if n_timeout > 0 else ""
                log.info(
                    "[figure-ocr] summary total=%d attempted=%d accepted=%d rejected=%d "
                    "never_attempted=%d%s total_elapsed=%.1fs",
                    n_figs, n_attempted, n_accepted, n_rejected, n_never_attempted,
                    _timeout_suffix, _total_elapsed,
                )
                if n_inference_failed > 0:
                    error_types = Counter(
                        (e.metadata or {}).get("figure_ocr_error_type") or "unknown"
                        for e in fig_objects
                        if (e.metadata or {}).get("figure_ocr_skip_reason") == "inference_failed"
                    )
                    # Log first sample message to give a concrete hint
                    sample_msg = next(
                        (
                            (e.metadata or {}).get("figure_ocr_error_message", "")
                            for e in fig_objects
                            if (e.metadata or {}).get("figure_ocr_skip_reason") == "inference_failed"
                            and (e.metadata or {}).get("figure_ocr_error_message")
                        ),
                        None,
                    )
                    log.warning(
                        "Figure OCR inference_failed for paper '%s': error_types=%s sample=%s",
                        paper_id, dict(error_types), sample_msg,
                    )

        # Figure caption fix — audit summary (no-op when no figures processed)
        _fig_out = [e for e in out if e.object_type == FIGURE]
        _n_fig_out = len(_fig_out)
        if _n_fig_out or _fig_inline_rejected or _fig_deduped_dropped:
            _n_fig_with_image = sum(
                1 for e in _fig_out if (e.metadata or {}).get("has_image")
            )
            log.info(
                "[figure-filter] paper=%s inline_prose_rejected=%d",
                paper_id, _fig_inline_rejected,
            )
            log.info(
                "[figure-dedupe] paper=%s duplicates_dropped=%d",
                paper_id, _fig_deduped_dropped,
            )
            log.info(
                "[figure-crop] paper=%s caption_only_skipped=%d "
                "caption_only_attempted=%d caption_only_recovered=%d",
                paper_id, _fig_crop_caption_only_skipped,
                _fig_crop_caption_only_attempted, _fig_crop_caption_only_recovered,
            )
            log.info(
                "[figure-summary] paper=%s total=%d with_image=%d without_image=%d",
                paper_id, _n_fig_out, _n_fig_with_image, _n_fig_out - _n_fig_with_image,
            )

    return out