from __future__ import annotations

import re
import sqlite3
from difflib import SequenceMatcher
from typing import Any


def _normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching."""
    text = text or ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def _token_overlap_score(a: str, b: str) -> float:
    """Simple Jaccard-like overlap on alnum tokens."""
    toks_a = set(re.findall(r"[a-z0-9]+", a.lower()))
    toks_b = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not toks_a or not toks_b:
        return 0.0
    inter = len(toks_a & toks_b)
    union = len(toks_a | toks_b)
    return inter / union if union else 0.0


def _combined_similarity(a: str, b: str) -> float:
    """
    Blend character-level and token-level similarity.
    Good enough for short scientific evidence quotes.
    """
    a_norm = _normalize_text(a)
    b_norm = _normalize_text(b)

    if not a_norm or not b_norm:
        return 0.0

    char_score = SequenceMatcher(None, a_norm, b_norm).ratio()
    tok_score = _token_overlap_score(a_norm, b_norm)

    # Weighted blend
    return 0.7 * char_score + 0.3 * tok_score


def load_spans_by_ids(
    conn: sqlite3.Connection,
    span_ids: list[str],
) -> list[dict[str, Any]]:
    """Load span rows from pdf_spans for a given list of span IDs."""
    if not span_ids:
        return []

    placeholders = ",".join("?" for _ in span_ids)
    query = f"""
        SELECT id, paper_id, page_num, span_index, text, bbox_json, block_index, line_index
        FROM pdf_spans
        WHERE id IN ({placeholders})
    """

    conn.row_factory = sqlite3.Row
    rows = conn.execute(query, span_ids).fetchall()
    conn.row_factory = None

    # Preserve DB order first, then sort below
    spans = [dict(r) for r in rows]
    spans.sort(key=lambda s: (s["page_num"], s["span_index"]))
    return spans


def select_best_span_ids_for_supporting_quote(
    *,
    conn: sqlite3.Connection,
    supporting_quote: str,
    candidate_span_ids: list[str],
    top_n: int = 3,
    min_score: float = 0.25,
    expand_neighbors: bool = True,
) -> list[str]:
    """
    Align a supporting_quote to the best matching span_ids among candidate spans.

    Args:
        conn: SQLite connection.
        supporting_quote: verification supporting_quote text.
        candidate_span_ids: candidate span ids from retrieval evidence.
        top_n: max number of aligned spans to return.
        min_score: minimum similarity threshold.
        expand_neighbors: if True, may include adjacent span when quote spans multiple lines.

    Returns:
        Ordered list of best-matching span_ids.
    """
    if not supporting_quote or not candidate_span_ids:
        return []

    spans = load_spans_by_ids(conn, candidate_span_ids)
    if not spans:
        return []

    scored: list[tuple[float, dict[str, Any]]] = []
    for sp in spans:
        score = _combined_similarity(supporting_quote, sp.get("text", ""))
        scored.append((score, sp))

    scored.sort(key=lambda x: x[0], reverse=True)

    best = [(score, sp) for score, sp in scored if score >= min_score]
    if not best:
        return []

    selected = best[:top_n]
    selected_ids = [sp["id"] for _, sp in selected]

    if not expand_neighbors:
        return selected_ids

    # Optionally include one adjacent span if it looks like the quote continues
    id_to_idx = {sp["id"]: i for i, sp in enumerate(spans)}
    expanded: list[str] = []
    seen = set()

    for sid in selected_ids:
        if sid not in seen:
            expanded.append(sid)
            seen.add(sid)

        idx = id_to_idx[sid]

        # Try next span first
        for neighbor_idx in (idx + 1, idx - 1):
            if neighbor_idx < 0 or neighbor_idx >= len(spans):
                continue
            nsp = spans[neighbor_idx]
            nid = nsp["id"]
            if nid in seen:
                continue

            # Only include if same page and somewhat relevant
            if nsp["page_num"] == spans[idx]["page_num"]:
                nscore = _combined_similarity(supporting_quote, nsp.get("text", ""))
                if nscore >= min_score * 0.75:
                    expanded.append(nid)
                    seen.add(nid)
                    break

    return expanded[: max(top_n, len(expanded))]