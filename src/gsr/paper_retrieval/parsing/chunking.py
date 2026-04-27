"""Chunking utilities for parsed papers – Module 3, Step 2.

Converts parsed paper structures into retrieval-ready chunks.

Supported modes:

V1:
    - ``chunk_paper(parsed, ...)``
    - text-based chunking over section text
    - outputs char offsets within each section

V2 (layout-aware):
    - ``chunk_paper_v2_from_spans(spans, ...)``
    - span-based chunking over layout-aware line spans
    - outputs span_ids + page range for bbox-aware downstream rendering

P1.7 — Boundary-aware V2 chunking
    Adds a lightweight segmentation step *before* the sliding window so that
    chunks cannot cross obvious layout / semantic boundaries:
      A) page change
      B) section-label change
      C) heading-like span (starts a new section title line)
      D) large vertical gap on the same page
      E) header / footer exclusion band (spans removed before segmentation)

    Helpers added (all in "V2 internals" section below):
      _looks_like_heading        — heuristic: short numbered / lettered title
      _is_in_header_footer_band  — top/bottom 6% of page height
      _has_large_vertical_gap    — gap > 1.8× max adjacent line height
      _segment_spans_for_chunking — applies A–D, returns list[list[span]]

P1.7b — Docling boundary hints (opportunistic, backward-compatible)
    When a NormalizedDocument is passed via the ``docling_doc`` parameter,
    Docling-derived structural signals augment the segmentation:
      F) Docling title/heading boundary — span text matches a Docling section
         heading; takes precedence over heuristic heading detection (C).
      G) Docling caption exclusion — spans identified by Docling as figure/table
         caption regions are excluded from body-text chunks and act as hard
         segment breaks (body before ≠ body after a caption).

    New helpers:
      _extract_docling_hints   — extracts (caption_ids, heading_texts) from doc
      _docling_role            — returns role tag for a span if Docling-labelled
      _has_docling_boundary    — boundary-F check against Docling heading set

    If ``docling_doc`` is None (default), P1.7 behavior is unchanged.
"""
from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API (V1)
# ---------------------------------------------------------------------------

def chunk_paper(
    parsed: dict,
    *,
    chunk_size: int = 120,
    chunk_overlap: int = 30,
) -> list[dict]:
    """Split a parsed paper into overlapping text chunks (V1).

    Args:
        parsed: Output of ``parse_paper_pdf()``.
        chunk_size: Number of tokens (approx. words) per chunk.
        chunk_overlap: Number of overlapping tokens between consecutive chunks.

    Returns:
        list of chunk dicts with keys:
            ``id``, ``paper_id``, ``chunk_index``, ``section``, ``page``,
            ``text``, ``char_start``, ``char_end``.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    paper_id = parsed["paper_id"]
    sections = parsed.get("sections", [])
    chunks: list[dict] = []
    chunk_index = 0

    for sec in sections:
        section_name = sec["heading"]
        page = sec["page"]
        text = sec["text"].strip()
        if not text:
            continue

        words = _tokenize_words(text)
        if not words:
            continue

        step = chunk_size - chunk_overlap

        # Map token index -> char span in original text
        word_spans = list(_iter_word_spans(text))
        n = len(words)

        for start in range(0, n, step):
            end = min(start + chunk_size, n)

            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words).strip()
            if not chunk_text:
                continue

            char_start = word_spans[start][0]
            char_end = word_spans[end - 1][1]

            chunks.append(
                {
                    "id": f"{paper_id}_chunk_{chunk_index}",
                    "paper_id": paper_id,
                    "chunk_index": chunk_index,
                    "section": section_name,
                    "page": page,
                    "text": chunk_text,
                    "char_start": char_start,
                    "char_end": char_end,
                }
            )
            chunk_index += 1

            if end == n:
                break

    log.info(
        "Chunked paper '%s': %d sections → %d chunks (size=%d, overlap=%d)",
        paper_id,
        len(sections),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


# ---------------------------------------------------------------------------
# Public API (V2)
# ---------------------------------------------------------------------------

def chunk_paper_v2_from_spans(
    spans: list[dict[str, Any]],
    *,
    chunk_size: int = 8,
    chunk_overlap: int = 2,
    min_chars: int = 40,
    docling_doc: Any = None,
) -> list[dict[str, Any]]:
    """Build retrieval chunks from line-level spans (V2, P1.7 boundary-aware).

    This function uses a sliding window over *line spans* instead of words.
    P1.7 adds a segmentation pass before the window so that chunks cannot
    cross page boundaries, section changes, heading-like lines, large vertical
    gaps, or page header/footer bands.

    P1.7b (optional): when ``docling_doc`` is a NormalizedDocument, Docling
    structural signals are used to add boundary hints F (heading) and G
    (caption exclusion).  Falls back silently to P1.7 if ``docling_doc``
    is None or carries no usable signals.

    Args:
        spans:
            Output ``parsed_v2["spans"]`` from ``parse_paper_pdf_v2()``.
            Each span should contain:
                - id
                - paper_id
                - page_num
                - span_index
                - text
                - bbox          [x0, y0, x1, y1] in PDF coordinates
                - optional section
        chunk_size:
            Number of spans (lines) per chunk.
        chunk_overlap:
            Number of overlapping spans between adjacent chunks.
        min_chars:
            Skip chunks whose concatenated text is too short.
        docling_doc:
            Optional NormalizedDocument (or any object with .figures, .tables,
            .sections attributes).  When provided, Docling-derived boundary
            hints F and G are activated.  Pass None (default) to keep plain
            P1.7 behaviour.

    Returns:
        list of chunk dicts compatible with ``save_chunks()``, including:
            ``id``, ``paper_id``, ``chunk_index``, ``section``, ``page``,
            ``page_start``, ``page_end``, ``text``, ``char_start``,
            ``char_end``, ``span_ids``.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    if not spans:
        return []

    spans = [s for s in spans if s.get("text", "").strip()]
    if not spans:
        return []

    # Ensure deterministic order
    spans = sorted(spans, key=lambda s: (s["page_num"], s["span_index"]))

    paper_id = spans[0]["paper_id"]
    step = chunk_size - chunk_overlap
    chunks: list[dict[str, Any]] = []
    chunk_index = 0

    # ------------------------------------------------------------------
    # P1.7 — pre-process: header/footer exclusion + boundary segmentation
    # ------------------------------------------------------------------

    # Estimate per-page height from the maximum y1 seen across all spans.
    # This is a proxy for page height — accurate enough for 6%-band detection.
    page_heights: dict[int, float] = {}
    for s in spans:
        bbox = s.get("bbox")
        if bbox and len(bbox) >= 4:
            pg = s["page_num"]
            y1 = float(bbox[3])
            if y1 > page_heights.get(pg, 0.0):
                page_heights[pg] = y1

    # Remove spans that fall inside the page header or footer band.
    # These produce misleading red boxes (running titles, page numbers, etc.)
    # when included in body-text chunks.  They are not deleted from the DB —
    # only excluded from the chunking input.
    n_original = len(spans)
    spans = [
        s for s in spans
        if not _is_in_header_footer_band(s, page_heights.get(s["page_num"], 0.0))
    ]
    n_removed = n_original - len(spans)

    if not spans:
        return []

    # ------------------------------------------------------------------
    # P1.7b — extract Docling boundary hints (opportunistic, fail-open)
    # ------------------------------------------------------------------
    docling_caption_ids, docling_heading_texts = _extract_docling_hints(docling_doc)

    # Pre-compute stats for logging (counts before segmentation).
    n_caption_excluded = sum(
        1 for s in spans if s.get("id") in docling_caption_ids
    )
    n_heading_matched = sum(
        1 for s in spans
        if (s.get("text") or "").strip().lower() in docling_heading_texts
    )

    # Segment the remaining spans into boundary-respecting groups, then run
    # the sliding window independently within each segment.
    segments, n_docling_title_boundaries = _segment_spans_for_chunking(
        spans,
        docling_heading_texts=docling_heading_texts,
        docling_caption_ids=docling_caption_ids,
    )

    log.info(
        "Segmented V2 spans for '%s': %d spans → %d usable spans"
        " (%d header/footer removed) → %d segments",
        paper_id, n_original, len(spans), n_removed, len(segments),
    )

    if docling_caption_ids or docling_heading_texts:
        log.info(
            "Docling boundary hints for '%s': role_spans=%d"
            " title_boundaries=%d caption_excluded=%d block_boundaries=0",
            paper_id,
            n_caption_excluded + n_heading_matched,
            n_docling_title_boundaries,
            n_caption_excluded,
        )

    for seg in segments:
        n_seg = len(seg)
        for start in range(0, n_seg, step):
            end = min(start + chunk_size, n_seg)
            window = seg[start:end]

            chunk = _build_span_window_chunk(
                paper_id=paper_id,
                chunk_index=chunk_index,
                window=window,
                min_chars=min_chars,
            )
            if chunk is not None:
                chunks.append(chunk)
                chunk_index += 1

            if end == n_seg:
                break

    log.info(
        "Chunked paper V2 '%s': %d spans → %d chunks (size=%d spans, overlap=%d spans)",
        paper_id,
        len(spans),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


# ---------------------------------------------------------------------------
# V2 internals — P1.7b Docling boundary hint helpers
# ---------------------------------------------------------------------------

def _extract_docling_hints(
    docling_doc: Any,
) -> tuple[frozenset[str], frozenset[str]]:
    """Extract boundary hint sets from a NormalizedDocument.

    Returns:
        (caption_span_ids, heading_texts_lower)
        caption_span_ids — span IDs that belong to a Docling figure/table region.
        heading_texts_lower — lowercased heading strings from Docling sections.
        Both are empty frozensets when ``docling_doc`` is None.
    """
    if docling_doc is None:
        return frozenset(), frozenset()

    caption_ids: set[str] = set()
    for fig in getattr(docling_doc, "figures", None) or []:
        caption_ids.update(getattr(fig, "span_ids", None) or [])
    for tbl in getattr(docling_doc, "tables", None) or []:
        caption_ids.update(getattr(tbl, "span_ids", None) or [])

    heading_texts: set[str] = set()
    for sec in getattr(docling_doc, "sections", None) or []:
        h = (getattr(sec, "heading", None) or "").strip()
        if h:
            heading_texts.add(h.lower())

    return frozenset(caption_ids), frozenset(heading_texts)


def _docling_role(span: dict[str, Any], caption_span_ids: frozenset[str]) -> str | None:
    """Return 'caption' if the span falls in a Docling figure/table region, else None."""
    if span.get("id") in caption_span_ids:
        return "caption"
    return None


def _has_docling_boundary(
    curr: dict[str, Any],
    heading_texts: frozenset[str],
) -> bool:
    """Return True if ``curr`` span text matches a Docling section heading (Boundary F)."""
    if not heading_texts:
        return False
    text = (curr.get("text") or "").strip()
    return bool(text) and text.lower() in heading_texts


# ---------------------------------------------------------------------------
# V2 internals — P1.7 boundary helpers
# ---------------------------------------------------------------------------

# Heading patterns: optional letter prefix for appendix sections (A.1, B.2.3),
# or digit-only section numbers (1, 2.3, 3.1.2), followed by an uppercase word.
_HEADING_RE = re.compile(
    r"^(?:[A-Z]\d*\.)?(?:\d+\.)*\d+\s+[A-Z]"  # e.g. "3.1 Ablation" / "C.2 Details"
    r"|^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$"       # Title-Case short line fallback
)

# Fraction of page height treated as header/footer exclusion band.
_HEADER_FOOTER_BAND = 0.06  # top 6% and bottom 6%

# Vertical-gap multiplier for triggering a new segment.
_GAP_MULTIPLIER = 1.8


def _looks_like_heading(span: dict[str, Any]) -> bool:
    """Return True if the span text looks like a section heading.

    Heuristics (all must pass):
      1. Stripped text is non-empty and ≤ 120 chars (headings are short).
      2. Text does not end with typical sentence-ending punctuation.
      3. Text matches a numbered-section or short-title-case pattern.
    """
    text = span.get("text", "").strip()
    if not text or len(text) > 120:
        return False
    # Sentence-ending punctuation → likely body text, not a heading
    if text[-1] in ".?!,;:":
        return False
    return bool(_HEADING_RE.match(text))


def _is_in_header_footer_band(span: dict[str, Any], page_height: float) -> bool:
    """Return True if the span's bbox falls in the top or bottom 6% of the page.

    If bbox is missing or page_height is zero, returns False (fail-open).
    """
    if not page_height:
        return False
    bbox = span.get("bbox")
    if not bbox or len(bbox) < 4:
        return False
    y0 = float(bbox[1])
    y1 = float(bbox[3])
    band = _HEADER_FOOTER_BAND * page_height
    # Span is fully inside the header band or the footer band
    return (y1 <= band) or (y0 >= page_height - band)


def _has_large_vertical_gap(prev: dict[str, Any], curr: dict[str, Any]) -> bool:
    """Return True if the vertical gap between two same-page spans is unusually large.

    Uses _GAP_MULTIPLIER × max(line-height of the two spans) as the threshold.
    Fails open (returns False) if bbox data is missing or malformed.
    """
    prev_bbox = prev.get("bbox")
    curr_bbox = curr.get("bbox")
    if not prev_bbox or not curr_bbox or len(prev_bbox) < 4 or len(curr_bbox) < 4:
        return False
    prev_y1 = float(prev_bbox[3])
    prev_y0 = float(prev_bbox[1])
    curr_y0 = float(curr_bbox[1])
    curr_y1 = float(curr_bbox[3])

    gap = curr_y0 - prev_y1
    if gap <= 0:
        return False  # overlapping or adjacent — not a gap

    prev_h = max(0.0, prev_y1 - prev_y0)
    curr_h = max(0.0, curr_y1 - curr_y0)
    ref_height = max(prev_h, curr_h)
    if ref_height <= 0:
        return False

    return gap > _GAP_MULTIPLIER * ref_height


def _segment_spans_for_chunking(
    spans: list[dict[str, Any]],
    *,
    docling_heading_texts: frozenset[str] | None = None,
    docling_caption_ids: frozenset[str] | None = None,
) -> tuple[list[list[dict[str, Any]]], int]:
    """Split a sorted span list into boundary-respecting segments (P1.7 + P1.7b).

    A new segment starts whenever any of these conditions holds:

      A) page_num changes
      B) section label changes (normalised; missing → "unknown")
      F) Docling heading boundary — span text matches a Docling section heading
         (P1.7b; takes precedence over heuristic heading detection C)
      C) current span looks like a section heading (_looks_like_heading)
      D) large vertical gap on the same page (_has_large_vertical_gap)

    Docling caption spans (G) are excluded from all segments and act as hard
    segment breaks: the segment before is closed and the segment after starts
    fresh.  This is only active when ``docling_caption_ids`` is non-empty.

    The loop uses ``current[-1]`` as *prev* so that skipped caption spans do
    not corrupt boundary comparisons.

    Returns:
        (segments, n_docling_title_boundaries)
        segments — list of non-empty span lists
        n_docling_title_boundaries — count of boundaries triggered by rule F
    """
    if not spans:
        return [], 0

    _heading_texts: frozenset[str] = docling_heading_texts or frozenset()
    _caption_ids: frozenset[str] = docling_caption_ids or frozenset()

    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    n_docling_title = 0

    for span in spans:
        # Boundary G — Docling caption/figure region: exclude and break segment
        if span.get("id") in _caption_ids:
            if current:
                segments.append(current)
                current = []
            continue  # caption span not added to any segment

        if not current:
            current.append(span)
            continue

        prev = current[-1]
        prev_page = prev["page_num"]
        curr_page = span["page_num"]
        prev_sec = (prev.get("section") or "unknown").strip()
        curr_sec = (span.get("section") or "unknown").strip()

        # Boundary A — page change
        if curr_page != prev_page:
            segments.append(current)
            current = [span]
            continue

        # Boundary B — section change
        if curr_sec != prev_sec:
            segments.append(current)
            current = [span]
            continue

        # Boundary F — Docling heading (stronger than heuristic; checked first)
        if _has_docling_boundary(span, _heading_texts):
            segments.append(current)
            current = [span]
            n_docling_title += 1
            continue

        # Boundary C — heuristic heading-like span starts a new segment
        if _looks_like_heading(span):
            segments.append(current)
            current = [span]
            continue

        # Boundary D — large vertical gap (same page implied by reaching here)
        if _has_large_vertical_gap(prev, span):
            segments.append(current)
            current = [span]
            continue

        current.append(span)

    if current:
        segments.append(current)

    return segments, n_docling_title


# ---------------------------------------------------------------------------
# V2 internals — chunk builder
# ---------------------------------------------------------------------------

def _build_span_window_chunk(
    *,
    paper_id: str,
    chunk_index: int,
    window: list[dict[str, Any]],
    min_chars: int,
) -> dict[str, Any] | None:
    """Construct one V2 chunk from a span window."""
    if not window:
        return None

    lines = [s["text"].strip() for s in window if s.get("text", "").strip()]
    if not lines:
        return None

    chunk_text = "\n".join(lines).strip()
    if len(chunk_text) < min_chars:
        return None

    span_ids = [s["id"] for s in window]
    pages = [s["page_num"] for s in window]

    page_start = min(pages)
    page_end = max(pages)

    # Keep legacy fields for compatibility with existing storage / retrieval / reports
    page = page_start
    char_start = 0
    char_end = len(chunk_text)

    section = _majority_section(window)

    return {
        "id": f"{paper_id}_chunk_{chunk_index}",
        "paper_id": paper_id,
        "chunk_index": chunk_index,
        "section": section,
        "page": page,
        "page_start": page_start,
        "page_end": page_end,
        "text": chunk_text,
        "char_start": char_start,
        "char_end": char_end,
        "span_ids": span_ids,
    }


def _majority_section(window: list[dict[str, Any]]) -> str:
    """Choose the most representative section label in a span window."""
    counts: dict[str, int] = {}

    for s in window:
        sec = s.get("section") or "unknown"
        counts[sec] = counts.get(sec, 0) + 1

    # Deterministic tie-break: first encountered section in the window
    best_section = None
    best_count = -1
    for s in window:
        sec = s.get("section") or "unknown"
        cnt = counts[sec]
        if cnt > best_count:
            best_count = cnt
            best_section = sec

    return best_section or "unknown"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\S+")


def _tokenize_words(text: str) -> list[str]:
    """Simple whitespace tokenizer (good enough for retrieval chunking)."""
    return text.split()


def _iter_word_spans(text: str):
    """Yield (start, end) character spans for each non-whitespace token."""
    for m in _WORD_RE.finditer(text):
        yield m.start(), m.end()