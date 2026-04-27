from __future__ import annotations

from typing import Any
import re

TABLE_PAT = re.compile(r"^(Table|Tab\.)\s+(\d+(?:\.\d+)*)\b[:.]?", re.IGNORECASE)
FIG_PAT = re.compile(r"^(Figure|Fig\.)\s+(\d+(?:\.\d+)*)\b[:.]?", re.IGNORECASE)
SECTION_NUM_PAT = re.compile(r"^(\d+(?:\.\d+)*)\b")

# Heuristics for caption continuation
_MAX_CAPTION_LINES = 4
_MAX_CAPTION_CHARS = 500
_MAX_VERTICAL_GAP_FACTOR = 1.8
_MAX_LEFT_SHIFT = 24.0
_MAX_RIGHT_EXPAND = 80.0


def _match_caption_prefix(text: str) -> tuple[str, str, str] | None:
    text = (text or "").strip()
    m = TABLE_PAT.match(text)
    if m:
        return "table", f"Table {m.group(2)}", m.group(2)
    m = FIG_PAT.match(text)
    if m:
        return "figure", f"Figure {m.group(2)}", m.group(2)
    return None


def _infer_section_number(text: str | None) -> str | None:
    if not text:
        return None
    m = SECTION_NUM_PAT.match(text.strip())
    return m.group(1) if m else None


def _clean(text: str | None) -> str:
    return " ".join((text or "").split()).strip()


def _bbox_height(span: dict[str, Any]) -> float:
    bbox = span.get("bbox")
    if not bbox or len(bbox) != 4:
        return 0.0
    return max(0.0, float(bbox[3]) - float(bbox[1]))


def _bbox_top(span: dict[str, Any]) -> float:
    bbox = span.get("bbox")
    if not bbox or len(bbox) != 4:
        return 0.0
    return float(bbox[1])


def _bbox_bottom(span: dict[str, Any]) -> float:
    bbox = span.get("bbox")
    if not bbox or len(bbox) != 4:
        return 0.0
    return float(bbox[3])


def _bbox_left(span: dict[str, Any]) -> float:
    bbox = span.get("bbox")
    if not bbox or len(bbox) != 4:
        return 0.0
    return float(bbox[0])


def _bbox_right(span: dict[str, Any]) -> float:
    bbox = span.get("bbox")
    if not bbox or len(bbox) != 4:
        return 0.0
    return float(bbox[2])


def _looks_like_section_heading(text: str) -> bool:
    t = _clean(text)
    if not t:
        return False

    # e.g. "3.1 Chat", "4 Experiments"
    if re.match(r"^\d+(?:\.\d+)*\s+[A-Z]", t):
        return True

    # all caps short heading
    if t.isupper() and len(t) <= 80:
        return True

    return False


def _looks_like_body_paragraph_start(text: str) -> bool:
    t = _clean(text)
    if not t:
        return False

    # only very strong paragraph starters
    if re.match(
        r"^(For each domain,|For each prompt,|For each task,|We construct|We evaluate|In this section,|In our experiments,|To this end,|Recent efforts|Finally,|The following sections)\b",
        t,
    ):
        return True

    return False



def _should_continue_caption(
    collected: list[dict[str, Any]],
    nxt: dict[str, Any],
    kind: str,
) -> bool:
    if not collected:
        return False

    prev = collected[-1]
    prev_text = _clean(prev.get("text"))
    nxt_text = _clean(nxt.get("text"))

    if not nxt_text:
        return False

    # Hard stops
    if _match_caption_prefix(nxt_text):
        return False
    if _looks_like_section_heading(nxt_text):
        return False
    if len(nxt_text) > 260:
        return False

    # Limit total size
    total_chars = sum(len(_clean(s.get("text"))) for s in collected)
    if total_chars + len(nxt_text) > _MAX_CAPTION_CHARS:
        return False
    if len(collected) >= _MAX_CAPTION_LINES:
        return False

    # Same page only
    if nxt.get("page_num") != prev.get("page_num"):
        return False

    # Section drift usually means we've left the caption block
    prev_sec = _clean(prev.get("section"))
    nxt_sec = _clean(nxt.get("section"))
    if prev_sec and nxt_sec and prev_sec != nxt_sec:
        return False

    # Span index jump too large
    if int(nxt.get("span_index", 10**9)) - int(prev.get("span_index", 10**9)) > 2:
        return False

    # Geometry-based stop
    prev_h = max(_bbox_height(prev), 1.0)
    gap = max(0.0, _bbox_top(nxt) - _bbox_bottom(prev))
    if gap > prev_h * _MAX_VERTICAL_GAP_FACTOR:
        return False

    left_shift = abs(_bbox_left(nxt) - _bbox_left(collected[0]))
    if left_shift > _MAX_LEFT_SHIFT:
        return False

    # If next line suddenly becomes much wider / different block, stop
    right_expand = _bbox_right(nxt) - _bbox_right(prev)
    if right_expand > _MAX_RIGHT_EXPAND and _looks_like_body_paragraph_start(nxt_text):
        return False

    
    if prev_text.endswith((".", "!", "?")) and _looks_like_body_paragraph_start(nxt_text) and len(collected) >= 3:
        return False

    if _looks_like_body_paragraph_start(nxt_text) and len(collected) >= 3:
        return False

    # Otherwise allow continuation
    return True


def extract_captions_from_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not spans:
        return []

    spans = sorted(spans, key=lambda s: (s["page_num"], s["span_index"]))
    out: list[dict[str, Any]] = []

    i = 0
    while i < len(spans):
        cur = spans[i]
        cur_text = _clean(cur.get("text"))
        matched = _match_caption_prefix(cur_text)
        if not matched:
            i += 1
            continue

        kind, label, number = matched
        page_num = cur["page_num"]

        collected = [cur]
        j = i + 1

        while j < len(spans):
            nxt = spans[j]
            if nxt.get("page_num") != page_num:
                break

            if not _should_continue_caption(collected, nxt, kind):
                break

            collected.append(nxt)
            j += 1

        bbox = [
            min(float(s["bbox"][0]) for s in collected if s.get("bbox")),
            min(float(s["bbox"][1]) for s in collected if s.get("bbox")),
            max(float(s["bbox"][2]) for s in collected if s.get("bbox")),
            max(float(s["bbox"][3]) for s in collected if s.get("bbox")),
        ]

        caption_text = " ".join(_clean(s.get("text")) for s in collected).strip()

        out.append(
            {
                "kind": kind,
                "label": label,
                "number": number,
                "page_num": page_num,
                "bbox": bbox,
                "span_ids": [s["id"] for s in collected],
                "caption_text": caption_text,
                "section": cur.get("section"),
                "section_number": _infer_section_number(cur.get("section")),
            }
        )

        i = j

    return out