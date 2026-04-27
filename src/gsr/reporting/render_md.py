from __future__ import annotations


from datetime import datetime
from typing import Any
from .utils import md_escape, as_quote_block, parse_chunk_keys, file_url, file_url_page
import re
from html import escape as html_escape
from collections import Counter, defaultdict


_SENT_SPLIT = re.compile(r"(?<=[\.\?\!。！？])\s+|\n+")

_COLOR_PALETTE = [
    "#fff3b0",  # light yellow
    "#bde0fe",  # light blue
    "#caffbf",  # light green
    "#ffc6ff",  # light pink
    "#ffd6a5",  # light orange
    "#d0f4de",  # mint
    "#e7c6ff",  # lavender
    "#cdeac0",  # pale green
]


def _normalize_verdict(v: str | None) -> str:
    v = (v or "unknown").lower()
    if v in {"contradicted", "refuted"}:
        return "refuted"
    if v in {"supported", "support"}:
        return "supported"
    if v in {"insufficient_evidence", "insufficient"}:
        return "insufficient_evidence"
    if v in {"not_verifiable", "non_verifiable"}:
        return "not_verifiable"
    return "unknown"

def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0

def _highlight_many_sentences(
    raw: str,
    sent_to_labels: list[tuple[str, list[str]]],
    *,
    multi_color: bool = True,
) -> str:
    raw = raw or ""
    if not raw:
        return ""

    hits = []
    for idx, (sent, labels) in enumerate(sent_to_labels):
        sent = (sent or "").strip()
        if not sent:
            continue
        pos = raw.find(sent)
        if pos < 0:
            continue
        hits.append((pos, pos + len(sent), sent, labels, idx))

    # Remove overlaps (keep earliest found by longer span first)
    hits.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    non_overlap = []
    last_end = -1
    for h in hits:
        if h[0] >= last_end:
            non_overlap.append(h)
            last_end = h[1]

    # Build output
    out_parts = []
    cur = 0
    for (s, e, sent, labels, color_idx) in non_overlap:
        out_parts.append(html_escape(raw[cur:s]))
        label_txt = ",".join(labels)
        body = html_escape(raw[s:e])
        if multi_color:
            color = _COLOR_PALETTE[color_idx % len(_COLOR_PALETTE)]
            out_parts.append(f'<mark style="background-color: {color}">[{label_txt}] {body}</mark>')
        else:
            out_parts.append(f"<mark>[{label_txt}] {body}</mark>")
        cur = e
    out_parts.append(html_escape(raw[cur:]))
    return "".join(out_parts)

def _openreview_url_from_forum(forum: str) -> str:
    forum = (forum or "").strip()
    return f"https://openreview.net/forum?id={forum}" if forum else ""

def _file_url(path: str) -> str:
    p = (path or "").strip()
    if not p:
        return ""
    # Windows path -> file URL (basic)
    p = p.replace("\\", "/")
    if not p.startswith("/"):
        # e.g. C:/...
        return "file:///" + p
    return "file://" + p


def _tokens(s: str) -> set[str]:
    s = (s or "").lower()
    parts = re.split(r"[^a-z0-9]+", s)
    return {p for p in parts if len(p) >= 3}

def _best_sentence(raw: str, claim: str) -> tuple[str, float]:
    raw = (raw or "").strip()
    claim = (claim or "").strip()
    if not raw or not claim:
        return ("", 0.0)

    claim_tok = _tokens(claim)
    if not claim_tok:
        return ("", 0.0)

    sents = [s.strip() for s in _SENT_SPLIT.split(raw) if s.strip()]
    best = ""
    best_score = 0.0
    for s in sents:
        st = _tokens(s)
        if not st:
            continue
        inter = len(claim_tok & st)
        score = inter / (len(claim_tok) ** 0.5)
        if score > best_score:
            best_score = score
            best = s
    return (best, best_score)

def _highlight_sentence(raw: str, sent: str) -> str:
    raw = raw or ""
    sent = (sent or "").strip()
    if not raw or not sent:
        return html_escape(raw)

    idx = raw.find(sent)
    if idx < 0:
        return html_escape(raw)

    before = html_escape(raw[:idx])
    mid = html_escape(raw[idx:idx + len(sent)])
    after = html_escape(raw[idx + len(sent):])
    return before + "<mark>" + mid + "</mark>" + after

def _collect_sentence_highlights_for_field(
    raw: str,
    claim_items: list[dict[str, Any]],
) -> list[tuple[str, list[str]]]:
    """
    Return list of (sentence, labels) where labels are claim indices like C3, C7.
    Deduplicate by sentence text.
    """
    sent_map: dict[str, list[str]] = {}
    for it in claim_items:
        claim_idx = it.get("claim_index")
        claim_text = (it.get("claim_text") or "").strip()
        sent, score = _best_sentence(raw, claim_text)
        sent = (sent or "").strip()
        if not sent:
            continue
        label = f"C{claim_idx}"
        sent_map.setdefault(sent, [])
        if label not in sent_map[sent]:
            sent_map[sent].append(label)

    # keep deterministic order by first appearance in raw
    items = []
    for sent, labels in sent_map.items():
        pos = raw.find(sent)
        items.append((pos if pos >= 0 else 10**9, sent, labels))
    items.sort(key=lambda x: x[0])

    return [(sent, labels) for _, sent, labels in items]

def _shorten(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    return t if len(t) <= max_chars else (t[:max_chars] + "…")

def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _mean(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return (sum(vals) / len(vals)) if vals else None


def _claim_key(c: dict[str, Any]) -> tuple[str, str, str]:
    """
    Key used for diffing across runs.
    Using (normalized text, source_field, review_id) gives far fewer false matches
    than only claim_text.
    """
    return (
        _norm(c.get("claim_text", "")),
        _norm(c.get("source_field", "")),
        str(c.get("review_id") or ""),
    )

def _format_claim_card(c: dict[str, Any]) -> list[str]:
    ct = (c.get("claim_text") or "").strip()

    lines: list[str] = []
    lines.append(f"- **{ct}**")

    meta = []
    # lineage 
    if c.get("id"):
        meta.append(f"claim_id=`{c.get('id')}`")
    if c.get("review_id") is not None:
        meta.append(f"review=`{c.get('review_id')}`")
    if c.get("source_field") is not None:
        meta.append(f"field=`{c.get('source_field')}`")
    if c.get("extraction_run_id") is not None:
        meta.append(f"run=`{c.get('extraction_run_id')}`")

    # scores
    if c.get("confidence") is not None:
        meta.append(f"conf={c.get('confidence')}")
    if c.get("challengeability") is not None:
        meta.append(f"chal={c.get('challengeability')}")

    # tags
    if c.get("category"):
        meta.append(f"cat=`{c.get('category')}`")
    if c.get("claim_type"):
        meta.append(f"type=`{c.get('claim_type')}`")

    if meta:
        lines.append("  - " + ", ".join(meta))

    # optional explainability fields
    if c.get("binary_question"):
        lines.append(f"  - binary_question: {c.get('binary_question')}")
    if c.get("why_challengeable"):
        lines.append(f"  - why_challengeable: {c.get('why_challengeable')}")
    if c.get("verbatim_quote"):
        lines.append(f"  - verbatim_quote: {c.get('verbatim_quote')}")

    return lines


def render_extraction_comparison(
    paper_id: str,
    variants: list[dict[str, Any]],
    experiment_claims: dict[Any, list[dict[str, Any]]],
    *,
    max_items_per_bucket: int = 12,
    show_presence_matrix: bool = True,
) -> str:
    """
    Compare extraction runs for the same paper.

    variants: list of dicts, must include:
      run_id, min_confidence, min_challengeability, fields, claim_count
      (optionally avg_conf/avg_chal computed earlier)
    experiment_claims: { run_id: [claim_row_dict, ...] }
      claim_row_dict follows your claims table schema:
        id, review_id, paper_id, source_field, claim_index, claim_text,
        confidence, challengeability, category, claim_type, ...
        extraction_run_id (important for lineage)
    """
    lines: list[str] = []
    lines.append(f"# Extraction Comparison — Paper {paper_id}")
    lines.append("")

    # ----------------------------
    # Variant Summary
    # ----------------------------
    lines.append("## Variant Summary")
    lines.append("")
    lines.append("| Experiment ID | min_conf | min_chal | fields | #claims | avg_conf | avg_chal |")
    lines.append("|--------|----------|----------|--------|---------|----------|----------|")

    # stable ordering (newest/highest first)
    #variants_sorted = sorted(variants, key=lambda v: str(v.get("run_id")), reverse=True)
    variants_sorted = sorted(variants, key=lambda v: str(v.get("experiment_id")), reverse=True)

    for v in variants_sorted:
        #rid = v.get("run_id")
        eid = v.get("experiment_id")  
        claims = experiment_claims.get(eid, [])

        avg_conf = v.get("avg_conf")
        if avg_conf is None:
            avg_conf = _mean([c.get("confidence") for c in claims])

        avg_chal = v.get("avg_chal")
        if avg_chal is None:
            avg_chal = _mean([c.get("challengeability") for c in claims])

        lines.append(
            f"| {eid} | "
            f"{v.get('min_confidence')} | "
            f"{v.get('min_challengeability')} | "
            f"`{v.get('fields')}` | "
            f"{v.get('claim_count', len(claims))} | "
            f"{(round(avg_conf, 3) if avg_conf is not None else '')} | "
            f"{(round(avg_chal, 3) if avg_chal is not None else '')} |"
          #f"{_fmt_num(e.get('avg_cal_score'))} |"
        )

    lines.append("")

    # ----------------------------
    # Build maps: run -> key -> claim
    # ----------------------------
    run_to_key_to_claim: dict[Any, dict[tuple, dict[str, Any]]] = {}
    run_to_keys: dict[Any, set[tuple]] = {}

    for v in variants_sorted:
        #rid = v.get("run_id")
        eid = v.get("experiment_id")
        key_map: dict[tuple, dict[str, Any]] = {}

        for c in experiment_claims.get(eid, []):
            k = _claim_key(c)
            # if duplicates, keep the one with higher confidence
            if k not in key_map:
                key_map[k] = c
            else:
                prev = key_map[k]
                if float(c.get("confidence") or 0.0) > float(prev.get("confidence") or 0.0):
                    key_map[k] = c

        run_to_key_to_claim[eid] = key_map
        run_to_keys[eid] = set(key_map.keys())

    if not run_to_keys:
        lines.append("> No claims found for any run.")
        return "\n".join(lines)


    # ----------------------------
    # Presence Matrix (ALL claims)
    # ----------------------------
    if show_presence_matrix and len(variants_sorted) >= 2:
        lines.append("## Claim Presence Matrix (All Claims)")
        lines.append("")
        lines.append("> ✓ indicates the claim appears in the experiment.")
        lines.append("")

        all_keys = set().union(*run_to_keys.values())
        n_runs = len(variants_sorted)

        # frequency across experiments
        freq = {k: sum(1 for eid in run_to_keys if k in run_to_keys[eid]) for k in all_keys}

        # sort: stable first (freq desc), then by text
        keys = sorted(list(all_keys), key=lambda k: (-freq[k], k[0], k[1], k[2]))

        header = ["claim (norm)", "field", "review", "freq"] + [f"exp {v['experiment_id']}" for v in variants_sorted]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")

        for k in keys:
            norm_text, field, review = k
            row = [
                norm_text[:80] + ("…" if len(norm_text) > 80 else ""),
                field,
                review,
                str(freq[k]),
            ]
            for v in variants_sorted:
                eid = v["experiment_id"]
                row.append("✓" if k in run_to_keys[eid] else "")
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")

    # ----------------------------
    # Stability Analysis
    # ----------------------------
    if len(variants_sorted) >= 2:
        lines.append("## Claim Stability Analysis")
        lines.append("")

        all_keys = set().union(*run_to_keys.values())
        n_runs = len(variants_sorted)

        stability = {
            k: sum(1 for eid in run_to_keys if k in run_to_keys[eid]) / n_runs
            for k in all_keys
        }

        # Fully stable (appear in all runs)
        stable = [k for k, s in stability.items() if s == 1.0]
        # Highly unstable (appear in exactly one run)
        unstable = [k for k, s in stability.items() if s == 1 / n_runs]

        lines.append(f"- Fully stable claims: **{len(stable)}**")
        lines.append(f"- Highly unstable claims: **{len(unstable)}**")
        lines.append("")

        # Show top unstable examples
        lines.append("### Highly Unstable Claims (appear in only one experiment)")
        lines.append("")
        for k in unstable[:10]:
            # find which run contains it
            for eid in run_to_keys:
                if k in run_to_keys[eid]:
                    c = run_to_key_to_claim[eid][k]
                    lines.extend(_format_claim_card(c))
                    break
        lines.append("")

        # Show stable examples
        lines.append("### Fully Stable Claims (appear in all experiments)")
        lines.append("")
        for k in stable[:10]:
            #rid = variants_sorted[0]["run_id"]
            eid =  variants_sorted[0]["experiment_id"]
            c = run_to_key_to_claim[eid][k]
            lines.extend(_format_claim_card(c))
        lines.append("")

        avg_stability = sum(stability.values()) / len(stability) if stability else 0
        lines.append(f"- Average claim stability: **{round(avg_stability, 3)}**")
        lines.append("")

    # ----------------------------
    # Pairwise diff (base vs others)
    # ----------------------------
    if len(variants_sorted) >= 2:
        lines.append("## Claim Differences (Base vs Others)")
        lines.append("")

        base = variants_sorted[0]
        base_id = base["experiment_id"]
        base_keys = run_to_keys.get(base_id, set())

        def _conf(eid: Any, k: tuple) -> float:
            c = run_to_key_to_claim.get(eid, {}).get(k, {})
            return float(c.get("confidence") or 0.0)

        for other in variants_sorted[1:]:
            other_id = other["experiment_id"]
            other_keys = run_to_keys.get(other_id, set())

            only_base = sorted(list(base_keys - other_keys), key=lambda k: _conf(base_id, k), reverse=True)
            only_other = sorted(list(other_keys - base_keys), key=lambda k: _conf(other_id, k), reverse=True)
            overlap = sorted(list(base_keys & other_keys), key=lambda k: max(_conf(base_id, k), _conf(other_id, k)), reverse=True)

            lines.append(f"### Experiment {base_id} vs Experiment {other_id}")
            lines.append("")
            lines.append(f"- overlap: **{len(overlap)}**")
            lines.append(f"- only in experiment {base_id}: **{len(only_base)}**")
            lines.append(f"- only in experiment {other_id}: **{len(only_other)}**")
            lines.append("")

            # Only in base
            lines.append(f"**Only in Experiment {base_id} (top {min(max_items_per_bucket, len(only_base))})**")
            lines.append("")
            for k in only_base[:max_items_per_bucket]:
                lines.extend(_format_claim_card(run_to_key_to_claim[base_id][k]))
            lines.append("")

            # Only in other
            lines.append(f"**Only in Experiment {other_id} (top {min(max_items_per_bucket, len(only_other))})**")
            lines.append("")
            for k in only_other[:max_items_per_bucket]:
                lines.extend(_format_claim_card(run_to_key_to_claim[other_id][k]))
            lines.append("")

            # Overlap sample
            lines.append(f"**Overlap (top {min(max_items_per_bucket, len(overlap))})**")
            lines.append("")
            for k in overlap[:max_items_per_bucket]:
                cb = run_to_key_to_claim[base_id].get(k, {})
                co = run_to_key_to_claim[other_id].get(k, {})
                c = cb if float(cb.get("confidence") or 0.0) >= float(co.get("confidence") or 0.0) else co
                lines.extend(_format_claim_card(c))
            lines.append("")

    return "\n".join(lines)

def render_verification_markdown(
    *,
    paper_id: str,
    paper_meta: dict[str, Any],
    rows: list[dict[str, Any]],
    chunk_map: dict[str, dict[str, Any]],
    evidence_mode: str,
    db_path_str: str,
    evidence_max_chars: int = 1400,
    assume_page_0_indexed: bool = False,
) -> str:
    """
    Verification report v2 (review-centric):
      - Group by review_id, then by source_field
      - Show raw review text with claim highlights
      - For each claim: verdict/conf/reasoning + evidence chunks
    """

    def _verdict_badge(v: str | None) -> str:
        v = _normalize_verdict(v)
        if v == "refuted":
            return "🔴 refuted"
        if v == "insufficient_evidence":
            return "🟠 insufficient_evidence"
        if v == "supported":
            return "🟢 supported"
        if v == "not_verifiable":
            return "⚪ not_verifiable"
        return f"⚫ {v}"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf_path = paper_meta.get("pdf_path")

    verdict_counter = Counter(_normalize_verdict(r.get("verdict")) for r in rows)
    status_counter = Counter((r.get("status") or "unknown") for r in rows)
    source_field_counter = Counter((r.get("source_field") or "unknown") for r in rows)

    # --- Map source_field -> review column name (best-effort) ---
    # This matches your extraction report mapping style. If your verification query
    # joins reviews and selects these columns, the renderer will use them.
    field_to_col = {
        "summary": "review_summary",
        "strengths": "review_strengths",
        "weaknesses": "review_weaknesses",
        "questions": "review_questions",
        "soundness": "review_soundness",
        "presentation": "review_presentation",
        "contribution": "review_contribution",
        # fallback possibilities (if you store main review text differently)
        "review": "review_text",
        "main_review_text": "review_text",
    }

    # --- Group rows: review_id -> source_field -> [rows...] ---
    by_review: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        rid = str(r.get("claim_review_id") or r.get("verification_review_id") or r.get("review_id") or "unknown")
        by_review[rid].append(r)

    # Order reviews deterministically
    review_ids = sorted(by_review.keys(), key=lambda x: (x == "unknown", x))

    lines: list[str] = []
    head = f"# Verification Report — {paper_id}"
    if paper_meta.get("title"):
        head += f" — {paper_meta['title']}"
    lines.append(head)
    lines.append("")
    lines.append(f"- Generated at: {now}")
    lines.append(f"- DB: `{db_path_str}`")
    lines.append(f"- Rows: {len(rows)}")
    lines.append(f"- Evidence key mode (STRICT): `{evidence_mode}`")
    lines.append(f"- Paper title: `{paper_meta.get('title') or '(no title)'}`")

    if pdf_path:
        lines.append(f"- PDF path: `{pdf_path}`")
        pdf_u = file_url(pdf_path)
        lines.append(f"- PDF link: [{pdf_u}]({pdf_u})")
    else:
        lines.append("- PDF path: (not available)")

    if paper_meta.get("pdf_sha256"):
        lines.append(f"- PDF sha256: `{paper_meta['pdf_sha256']}`")

    if rows:
        exp_id = rows[0].get("extraction_experiment_id")
        if exp_id:
            lines.append(f"- Experiment: `{exp_id}`")

    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("### Verdicts")
    lines.append("")
    for k, v in verdict_counter.most_common():
        lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("### Status")
    lines.append("")
    for k, v in status_counter.most_common():
        lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("### Source fields")
    lines.append("")
    for k, v in source_field_counter.most_common():
        lines.append(f"- {k}: {v}")

    # Quick index: show top "bad" claims (contradicted high confidence)
    bad = [
        r for r in rows
        if _normalize_verdict(r.get("verdict")) == "refuted" and (r.get("status") == "success")
    ]
    bad_sorted = sorted(bad, key=lambda r: float(r.get("verification_confidence") or 0.0), reverse=True)[:10]
    if bad_sorted:
        lines.append("")
        lines.append("### Quick links (top contradicted)")
        lines.append("")
        for r in bad_sorted:
            rid = str(r.get("claim_review_id") or r.get("verification_review_id") or "unknown")
            cidx = r.get("claim_index")
            anchor = f"review-{rid}-c{cidx}"
            lines.append(f"- {rid} / C{cidx}: [{_verdict_badge(r.get('verdict'))}](#{anchor})")

    lines.append("")
    lines.append("## Reviews")
    lines.append("")

    # Preferred display order for fields
    field_order = ["summary", "strengths", "weaknesses", "questions", "soundness", "presentation", "contribution", "unknown"]

    for review_id in review_ids:
        rrows = by_review[review_id]
        lines.append("---")
        lines.append(f"## Review `{review_id}`")
        lines.append("")

        # --- Mini summary for this review (success rows only) ---
        vr = [x for x in rrows if (x.get("status") or "success") == "success"]
        vcounter = Counter(_normalize_verdict(x.get("verdict")) for x in vr)
        total = sum(vcounter.values())

        def _v(name: str) -> int:
            return int(vcounter.get(name, 0))

        lines.append("**Mini summary**")
        lines.append("")
        lines.append(
            f"- total verified claims: **{total}** "
            f"(🔴 refuted: **{_v('refuted')}**, "
            f"🟠 insufficient_evidence: **{_v('insufficient_evidence')}**, "
            f"🟢 supported: **{_v('supported')}**, "
            f"⚪ not_verifiable: **{_v('not_verifiable')}**, "
            f"unknown: **{_v('unknown')}**)"
        )
        lines.append("")

        worst = sorted(
            [x for x in vr if _normalize_verdict(x.get("verdict")) == "refuted"],
            key=lambda x: float(x.get("verification_confidence") or 0.0),
            reverse=True,
        )[:3]
        if worst:            
            lines.append("### Quick links (top refuted)")
            for x in worst:
                cidx = x.get("claim_index")
                anchor = f"review-{review_id}-c{cidx}"
                conf = x.get("verification_confidence")
                lines.append(f"  - C{cidx} (conf={conf}): [jump](#{anchor})")
            lines.append("")

        # Basic metadata from the first row (if present)
        first = rrows[0]
        lines.append("**Metadata**")
        # If your query later adds review rating/confidence/signatures, these will show
        if first.get("review_rating") is not None:
            lines.append(f"- rating: `{first.get('review_rating')}`")
        if first.get("review_confidence") is not None:
            lines.append(f"- confidence: `{first.get('review_confidence')}`")
        if first.get("review_signatures"):
            lines.append(f"- signatures: `{first.get('review_signatures')}`")
        lines.append("")

        # Group by source_field
        by_field: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rrows:
            sf = (r.get("source_field") or "unknown").strip() or "unknown"
            by_field[sf].append(r)

        # stable field ordering
        fields = sorted(by_field.keys(), key=lambda f: (field_order.index(f) if f in field_order else 999, f))

        for source_field in fields:
            items = by_field[source_field]

            items = sorted(
                items,
                key=lambda x: (
                    int(x["claim_index"]) if x.get("claim_index") is not None else 10**9,
                    str(x.get("claim_id") or "")
                ),
            )

            lines.append(f"### Field: `{source_field}`")
            lines.append("")

            # -------- Raw review text retrieval (best-effort) --------
            raw_full = ""
            # 1) explicit review_text column if your query supplies it
            if items and items[0].get("review_text"):
                raw_full = (items[0].get("review_text") or "").strip()

            # 2) field-based review_* columns (if your query supplies them)
            if not raw_full:
                col = field_to_col.get(source_field)
                if col:
                    raw_full = (items[0].get(col) or "").strip()

            # If still missing, show a hint
            if not raw_full:
                lines.append("> Raw review text is not available for this field (query did not include reviews text columns).")
                lines.append("")
            else:
                # Highlight all claim-linked sentences in this field
                # Prefer verbatim_quote if it appears; else best-sentence match for claim_text
                # We'll build sentence-level highlights by mapping each claim to a best sentence.
                sent_map: dict[str, list[str]] = {}

                for it in items:
                    label = f"C{it.get('claim_index')}"
                    # Try verbatim_quote as an anchor; if it appears, highlight that exact sentence region by best sentence
                    vq = (it.get("verbatim_quote") or "").strip()
                    if vq and vq in raw_full:
                        # best sentence containing vq (fall back to vq itself)
                        s_best, _ = _best_sentence(raw_full, vq)
                        s = s_best.strip() if s_best else vq
                    else:
                        ct = (it.get("claim_text") or "").strip()
                        s, _ = _best_sentence(raw_full, ct)

                    s = (s or "").strip()
                    if not s:
                        continue
                    sent_map.setdefault(s, [])
                    if label not in sent_map[s]:
                        sent_map[s].append(label)

                # deterministic ordering by appearance
                sent_to_labels = []
                for s, labels in sent_map.items():
                    pos = raw_full.find(s)
                    sent_to_labels.append((pos if pos >= 0 else 10**9, s, labels))
                sent_to_labels.sort(key=lambda x: x[0])
                sent_to_labels_final = [(s, labels) for _, s, labels in sent_to_labels]

                highlighted_full = _highlight_many_sentences(raw_full, sent_to_labels_final, multi_color=True)

                lines.append("<details>")
                lines.append("<summary>Raw review text (claims highlighted)</summary>")
                lines.append("")
                lines.append(highlighted_full)
                lines.append("")
                lines.append("</details>")
                lines.append("")

            # -------- Claims & verification --------
            lines.append("#### Claims & Verification")
            lines.append("")

            for it in items:
                cidx = it.get("claim_index")
                anchor = f"review-{review_id}-c{cidx}"
                lines.append(f"<a id=\"{anchor}\"></a>")
                lines.append(f"##### C{cidx} — {_verdict_badge(it.get('verdict'))}")
                lines.append("")
                lines.append(f"- status: `{it.get('status')}`")
                lines.append(f"- verification_confidence: `{it.get('verification_confidence')}`")
                lines.append(f"- verification_model_id: `{it.get('verification_model_id')}`")
                lines.append(f"- verified_at: `{it.get('verified_at')}`")
                if it.get("status") != "success" and it.get("error_message"):
                    lines.append(f"- error: `{md_escape(it.get('error_message'))}`")

                lines.append("")
                lines.append("**Claim**")
                lines.append("")
                lines.append(as_quote_block(md_escape(it.get("claim_text"))).rstrip())
                lines.append("")

                if it.get("verbatim_quote"):
                    lines.append("**Verbatim quote (from review)**")
                    lines.append("")
                    lines.append(as_quote_block(md_escape(it.get("verbatim_quote"))).rstrip())
                    lines.append("")

                # show extraction run meta (useful lineage)
                lines.append("**Extraction lineage**")
                lines.append("")
                lines.append(f"- extraction_run_id: `{it.get('extraction_run_id')}`")
                if it.get("extraction_experiment_id"):
                    lines.append(f"- experiment_id: `{it.get('extraction_experiment_id')}`")
                lines.append(f"- config_hash: `{it.get('extraction_config_hash')}`")
                lines.append(f"- fields: `{it.get('extraction_fields')}`")
                lines.append(
                    f"- thresholds: min_confidence=`{it.get('extraction_min_confidence')}`, "
                    f"min_challengeability=`{it.get('extraction_min_challengeability')}`"
                )
                lines.append(f"- extracted_at(run): `{it.get('extraction_finished_at') or it.get('extracted_at')}`")
                lines.append("")

                if it.get("reasoning"):
                    lines.append("**Reasoning**")
                    lines.append("")
                    lines.append(md_escape(it.get("reasoning")))
                    lines.append("")

                if it.get("supporting_quote"):
                    lines.append("**Supporting quote (LLM output)**")
                    lines.append("")
                    lines.append(as_quote_block(md_escape(it.get("supporting_quote"))).rstrip())
                    lines.append("")

                # Evidence chunks
                raw = it.get("evidence_chunk_ids")
                keys = parse_chunk_keys(raw)

                lines.append("**Evidence (paper chunks)**")
                lines.append("")
                lines.append(f"- evidence_chunk_ids raw: `{md_escape(raw)}`")
                lines.append("")

                if not keys:
                    lines.append("_No parseable evidence_chunk_ids._")
                    lines.append("")
                else:
                    for rank, k in enumerate(keys, 1):
                        ch = chunk_map.get(k)
                        if not ch:
                            lines.append(f"**Evidence #{rank}** (key={k}) — not found for this paper_id")
                            lines.append("")
                            continue

                        sec = ch.get("section")
                        page = ch.get("page")

                        page_link = None
                        page_display = page
                        try:
                            if pdf_path and page is not None:
                                p = int(page)
                                if assume_page_0_indexed:
                                    p += 1
                                page_display = p
                                page_link = file_url_page(pdf_path, p)
                        except Exception:
                            page_link = None

                        if evidence_mode == "id":
                            id_info = f"chunk_id={ch.get('id')}"
                        else:
                            id_info = f"chunk_index={ch.get('chunk_index')} (chunk_id={ch.get('id')})"

                        if page_link:
                            lines.append(f"**Evidence #{rank}** — [page {page_display}]({page_link}), section={sec}, {id_info}")
                        else:
                            lines.append(f"**Evidence #{rank}** — page {page_display}, section={sec}, {id_info}")
                        lines.append("")

                        text = (ch.get("text") or "").strip()
                        if len(text) > evidence_max_chars:
                            text = text[:evidence_max_chars] + "…"
                        lines.append(as_quote_block(text).rstrip())
                        lines.append("")

                lines.append("")  # spacing between claims

            lines.append("")  # spacing between fields

    return "\n".join(lines)

def render_extraction_markdown(
    *,
    paper_id: str,
    paper_meta: dict[str, Any] | None,
    rows: list[dict[str, Any]],
    review_count: int,
    raw_max_chars: int = 2200,
) -> str:
    


    paper_meta = paper_meta or {}
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
 
    title = paper_meta.get("title") or ""
    forum = paper_meta.get("forum") or ""
    pdf_path = paper_meta.get("pdf_path") or ""
    pdf_error = paper_meta.get("pdf_error") or ""

    lines: list[str] = []
    openreview_url = (
        f"https://openreview.net/forum?id={forum}"
            if forum else ""
        )

    pdf_url = ""
    if pdf_path:
        pdf_url = "file:///" + pdf_path.replace("\\", "/")

    head = f"# Extraction Report — {paper_id}"
    if title:
        head += f" — {title}"
    lines.append(head)
    lines.append("")
    lines.append(f"- Total reviews: **{review_count}**")
    lines.append(f"- Total extracted claims: **{len(rows)}**")

    if openreview_url:
        lines.append(f"- Paper URL: {openreview_url}")
    if pdf_url:
        lines.append(f"- Local PDF: {pdf_url}")
    if pdf_error:
        lines.append(f"- PDF error: `{pdf_error}`")

    lines.append("")

    from collections import defaultdict

    def _mean(xs: list[float]) -> float:
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else 0.0

    # ---- summary by run ----
    run_groups = defaultdict(list)
    field_counts = defaultdict(int)
    run_field_counts = defaultdict(int)

    for r in rows:
        eid = r.get("experiment_id") or r.get("extraction_run_id")
        run_groups[eid].append(r)
        field = r.get("source_field") or "unknown"
        field_counts[field] += 1
        run_field_counts[(eid, field)] += 1

    lines.append("## Overview")
    lines.append("")
    lines.append("### Extraction runs")
    lines.append("")
    lines.append("| run_id | model | min_conf | min_chal | fields | #claims | avg_conf | avg_chal |")
    lines.append("|---:|---|---:|---:|---|---:|---:|---:|")

    for eid, items in run_groups.items():
        avg_conf = _mean([x.get("claim_confidence") for x in items])
        avg_chal = _mean([x.get("claim_challengeability") for x in items])
        first = items[0]
        lines.append(
            f"| {eid} | `{first.get('run_model_id')}` | {first.get('run_min_confidence')} | "
            f"{first.get('run_min_challengeability')} | `{first.get('run_fields')}` | "
            f"{len(items)} | {avg_conf:.3f} | {avg_chal:.3f} |"
        )

    lines.append("")
    lines.append("### Fields")
    lines.append("")
    lines.append("| field | #claims |")
    lines.append("|---|---:|")
    for f, c in sorted(field_counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"| `{f}` | {c} |")
    lines.append("")


    # map source_field -> reviews table column (your schema)
    field_to_col = {
        "summary": "review_summary",
        "strengths": "review_strengths",
        "weaknesses": "review_weaknesses",
        "questions": "review_questions",
        "soundness": "review_soundness",
        "presentation": "review_presentation",
        "contribution": "review_contribution",
    }

    cur_review = None
    cur_field = None
    cur_run = None

    for r in rows:
        review_id = r.get("review_id")
        source_field = r.get("source_field") or "unknown"
        experiment_id = r.get("experiment_id") or r.get("extraction_run_id")

        if review_id != cur_review:
            cur_review = review_id
            cur_field = None
            cur_run = None

            lines.append("---")
            lines.append(f"## Review `{review_id}`")
            lines.append("")


            # metadata (your schema)
            rating = r.get("review_rating")
            conf = r.get("review_confidence")
            sig = r.get("review_signatures")
            if rating is not None or conf is not None or sig:
                lines.append("**Metadata**")
                if rating is not None:
                    lines.append(f"- rating: `{rating}`")
                if conf is not None:
                    lines.append(f"- confidence: `{conf}`")
                if sig:
                    lines.append(f"- signatures: `{sig}`")
                lines.append("")

        if source_field != cur_field:
            cur_field = source_field
            cur_run = None

            lines.append(f"### Source field: `{source_field}`")
            raw_col = field_to_col.get(source_field)
            raw_text_full = (r.get(raw_col) or "").strip() if raw_col else ""
            raw_text_short = _shorten(raw_text_full, raw_max_chars)

            if raw_text_short:

                # gather all items for this review_id + source_field (across runs)
                field_items = [x for x in rows if x["review_id"] == review_id and x["source_field"] == source_field]

                sent_to_labels = _collect_sentence_highlights_for_field(raw_text_full, field_items)
                highlighted_full = _highlight_many_sentences(raw_text_full, sent_to_labels, multi_color=True)
                
                lines.append("")
                lines.append("<details>")
                lines.append("<summary>Raw review text (all claim-linked sentences highlighted)</summary>")
                lines.append("")
                lines.append(highlighted_full)
                lines.append("")
                lines.append("</details>")

            else:
                lines.append("")
                lines.append("> Raw review text not available for this field (empty or missing column).")
            lines.append("")
            lines.append("---")

        if experiment_id != cur_run:
            cur_run = experiment_id
            lines.append(f"#### Extraction run `{experiment_id}`")
            lines.append("")
            lines.append(f"- model_id: `{r.get('run_model_id')}`  ")
            lines.append(f"- config_hash: `{r.get('config_hash')}`  ")
            lines.append(f"- fields: `{r.get('run_fields')}`  ")
            lines.append(
                f"- thresholds: min_confidence=`{r.get('run_min_confidence')}`, "
                f"min_challengeability=`{r.get('run_min_challengeability')}`"
            )
            lines.append("")
            lines.append("---")

        claim_text = (r.get("claim_text") or "").strip()
        raw_col = field_to_col.get(source_field)
        raw_full = (r.get(raw_col) or "").strip() if raw_col else ""

        best_sent, score = _best_sentence(raw_full, claim_text)
        highlighted = _highlight_sentence(_shorten(raw_full, raw_max_chars), best_sent) if best_sent else html_escape(_shorten(raw_full, raw_max_chars))

        # claim card
        lines.append(
            f"- **Claim #{r.get('claim_index')}** "
            f"(conf={r.get('claim_confidence')}, chal={r.get('claim_challengeability')}"           
        )
        lines.append(f"  - {claim_text}")

        if r.get("verbatim_quote"):
            lines.append(f"  - verbatim_quote: {r.get('verbatim_quote')}")
        if r.get("category"):
            lines.append(f"  - category: `{r.get('category')}`")
        if r.get("claim_type"):
            lines.append(f"  - claim_type: `{r.get('claim_type')}`")

        # challenge info (very useful for discussion)
        if r.get("binary_question"):
            lines.append(f"  - binary_question: {r.get('binary_question')}")
        if r.get("why_challengeable"):
            lines.append(f"  - why_challengeable: {r.get('why_challengeable')}")

        # evidence sentence + highlighted raw excerpt
        if best_sent:
            lines.append(f"  - most_relevant_sentence (score={round(score, 3)}): {best_sent}")
        else:
            lines.append("  - most_relevant_sentence: *(not found)*")

        lines.append("")
        lines.append("  <details>")
        lines.append("  <summary>Raw text excerpt with highlight</summary>")
        lines.append("")
        lines.append("  " + highlighted.replace("\n", "\n  "))
        lines.append("")
        lines.append("  </details>")
        lines.append("")
        lines.append("---")

    return "\n".join(lines)




# ----------------------------
# Renderer: review-first
# ----------------------------

def render_extraction_markdown_review_first(
    *,
    paper_id: str,
    paper_meta: dict[str, Any] | None,
    review_count: int,
    reviews: list[dict[str, Any]],
    claims_by_review: dict[str, list[dict[str, Any]]],
    experiment_overview: list[dict[str, Any]] | None = None,
    selected_experiment_id: str | None = None,
    raw_max_chars: int = 2200,
    multi_color: bool = True,
) -> str:
    """
    Review-first extraction report (experiment-aware):

    - Header: paper meta + ALL experiments overview (even 0-claim experiments).
    - Body: shows ALL reviews.
    - Claims shown are expected to already be filtered to selected_experiment_id
      by the exporter/query layer (recommended).
    """

    from datetime import datetime
    from collections import defaultdict

    paper_meta = paper_meta or {}
    experiment_overview = experiment_overview or []

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    title = (paper_meta.get("title") or "").strip()
    forum = (paper_meta.get("forum") or "").strip()
    pdf_path = (paper_meta.get("pdf_path") or "").strip()
    pdf_error = (paper_meta.get("pdf_error") or "").strip()

    paper_url = _openreview_url_from_forum(forum)
    pdf_url = _file_url(pdf_path) if pdf_path else ""

    def _fmt_num(x: Any, nd: int = 3) -> str:
        if x is None:
            return ""
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return str(x)

    def _get_conf(row: dict[str, Any]) -> Any:
        return row.get("claim_confidence", row.get("confidence"))

    def _get_chal(row: dict[str, Any]) -> Any:
        return row.get("claim_challengeability", row.get("challengeability"))

    def _get_claim_id(row: dict[str, Any]) -> Any:
        return row.get("claim_id", row.get("id"))

    def _get_run_id(row: dict[str, Any]) -> Any:
        # Prefer extraction_run_id if present; fallback to whatever exists
        return row.get("extraction_run_id") or row.get("run_id") or "unknown_run"
    
    def _get_cal_score(row: dict[str, Any]) -> Any:
        return row.get("calibrated_score")

    def _get_cal_score_norm(row: dict[str, Any]) -> Any:
        return row.get("calibrated_score_norm")

    def _get_cal_error(row: dict[str, Any]) -> Any:
        return row.get("calibration_error")

    def _get_cal_error_detail(row: dict[str, Any]) -> Any:
        return row.get("calibration_error_detail")

    def _score_badge(row: dict[str, Any]) -> str:
        score = _get_cal_score(row)
        err = _get_cal_error(row)

        if err:
            return "⚪"
        if score is None:
            return "⚪"
        try:
            score = int(score)
        except Exception:
            return "⚪"

        if score <= 2:
            return "🔴"
        if score == 3:
            return "🟠"
        if score >= 4:
            return "🟢"
        return "⚪"

    def _fmt_cal_score(row: dict[str, Any]) -> str:
        score = _get_cal_score(row)
        if score is None:
            return "N/A"
        return str(score)

    def _score_label(row: dict[str, Any]) -> str:
        score = _get_cal_score(row)
        err = _get_cal_error(row)

        if err:
            return "score unavailable"
        if score is None:
            return "score unavailable"

        try:
            score = int(score)
        except Exception:
            return "score unavailable"

        if score == 5:
            return "highly faithful"
        if score == 4:
            return "mostly faithful"
        if score == 3:
            return "borderline / paraphrase drift"
        if score == 2:
            return "substantial drift"
        if score == 1:
            return "unsupported / hallucinated"
        return "score unavailable"

    # ----------------------------
    # Header
    # ----------------------------
    lines: list[str] = []
    head = f"# Extraction Report — {paper_id}"
    if title:
        head += f" — {title}"
    lines.append(head)
    lines.append("")
    lines.append(f"- Generated at: {now}")
    lines.append(f"- Total reviews in DB: **{review_count}**")
    lines.append(f"- Reviews shown in this report: **{len(reviews)}**")

    total_claims_in_report = sum(len(v) for v in claims_by_review.values())
    lines.append(f"- Total extracted claims shown: **{total_claims_in_report}**")


    all_claim_rows = [c for xs in claims_by_review.values() for c in xs]

    scored_claims = [c for c in all_claim_rows if _get_cal_score(c) is not None]
    score_errors = [c for c in all_claim_rows if _get_cal_error(c)]

    lines.append(f"- Claims with calibrated score: **{len(scored_claims)}**")
    lines.append(f"- Claims without calibrated score: **{len(all_claim_rows) - len(scored_claims)}**")

    if scored_claims:
        avg_cal = sum(float(_get_cal_score(c)) for c in scored_claims) / len(scored_claims)
        lines.append(f"- Avg calibrated score: **{avg_cal:.3f}**")

        dist = {k: 0 for k in [1, 2, 3, 4, 5]}
        for c in scored_claims:
            try:
                dist[int(_get_cal_score(c))] += 1
            except Exception:
                pass
        lines.append(
            "- Calibrated score distribution: "
            f"`1:{dist[1]}` `2:{dist[2]}` `3:{dist[3]}` `4:{dist[4]}` `5:{dist[5]}`"
        )

    if score_errors:
        from collections import Counter
        err_counter = Counter(str(_get_cal_error(c)) for c in score_errors if _get_cal_error(c))
        err_parts = [f"`{k}:{v}`" for k, v in sorted(err_counter.items(), key=lambda x: str(x[0]))]
        lines.append(f"- Calibration errors: {' '.join(err_parts)}")


    if selected_experiment_id:
        lines.append(f"- Selected experiment: `{selected_experiment_id}`")

    if paper_url:
        lines.append(f"- Paper URL: {paper_url}")
    if pdf_url:
        lines.append(f"- Local PDF: {pdf_url}")
    if pdf_error:
        lines.append(f"- PDF error: `{pdf_error}`")
    lines.append("")

    # ----------------------------
    # Experiments overview (ALL)
    # ----------------------------
    lines.append("## Experiments")
    lines.append("")
    if not experiment_overview:
        lines.append("> No experiment records found for this paper (experiment_id may be NULL).")
        lines.append("")
    else:
        lines.append("| experiment_id | model_id | min_conf | min_chal | fields | #reviews | #claims | avg_conf | avg_chal | last_started_at |")
        lines.append("|---|---|---:|---:|---|---:|---:|---:|---:|---|")
        for e in experiment_overview:
            lines.append(
                f"| `{e.get('experiment_id')}` | `{e.get('model_id')}` | "
                f"{e.get('min_confidence') if e.get('min_confidence') is not None else ''} | "
                f"{e.get('min_challengeability') if e.get('min_challengeability') is not None else ''} | "
                f"`{e.get('fields')}` | "
                f"{e.get('reviews_processed') or 0} | {e.get('claims_written') or 0} | "
                f"{_fmt_num(e.get('avg_conf'))} | {_fmt_num(e.get('avg_chal'))} | "
                f"{e.get('last_started_at') or ''} |"
            )
        lines.append("")

    # ----------------------------
    # Reviews
    # ----------------------------
    field_order = ["summary", "strengths", "weaknesses", "questions", "soundness", "presentation", "contribution"]



    for rv in reviews:
        review_id = rv.get("id")
        lines.append("---")
        lines.append(f"## Review `{review_id}`")
        lines.append("")

        # metadata (review table fields)
        lines.append("**Metadata**")
        if rv.get("rating") is not None:
            lines.append(f"- rating: `{rv.get('rating')}`")
        if rv.get("confidence") is not None:
            lines.append(f"- confidence: `{rv.get('confidence')}`")
        if rv.get("signatures"):
            lines.append(f"- signatures: `{rv.get('signatures')}`")
        if rv.get("replyto"):
            lines.append(f"- replyto: `{rv.get('replyto')}`")
        if rv.get("forum"):
            lines.append(f"- forum: `{rv.get('forum')}`")
        lines.append("")

        # claims for this review (already filtered to selected experiment by exporter ideally)
        review_claims = claims_by_review.get(str(review_id), []) or claims_by_review.get(review_id, []) or []

        if review_claims:
            review_scored = [c for c in review_claims if _get_cal_score(c) is not None]
            review_errs = [c for c in review_claims if _get_cal_error(c)]

            if review_scored or review_errs:
                dist = {k: 0 for k in [1, 2, 3, 4, 5]}
                for c in review_scored:
                    try:
                        dist[int(_get_cal_score(c))] += 1
                    except Exception:
                        pass

                lines.append(
                    f"> Calibrated summary: "
                    f"1={dist[1]}, 2={dist[2]}, 3={dist[3]}, 4={dist[4]}, 5={dist[5]}, "
                    f"errors={len(review_errs)}"
                )
                lines.append("")

        if not review_claims:
            lines.append("> No claims extracted for this review in the selected experiment.")
            lines.append("")

        for field in field_order:
            raw_full = (rv.get(field) or "").strip()
            if not raw_full:
                continue

            lines.append(f"### Field: `{field}`")
            lines.append("")

            field_claims = [c for c in review_claims if (c.get("source_field") or "").strip() == field]

            # Raw text with multi-claim highlights
            if field_claims:
                sent_to_labels = _collect_sentence_highlights_for_field(raw_full, field_claims)
                highlighted_full = _highlight_many_sentences(raw_full, sent_to_labels, multi_color=multi_color)
                highlighted_full = _shorten(highlighted_full, raw_max_chars * 3)

                lines.append("<details>")
                lines.append("<summary>Raw review text (all claim-linked sentences highlighted)</summary>")
                lines.append("")
                lines.append(highlighted_full)
                lines.append("")
                lines.append("</details>")
                lines.append("")
            else:
                lines.append("<details>")
                lines.append("<summary>Raw review text</summary>")
                lines.append("")
                lines.append(html_escape(_shorten(raw_full, raw_max_chars)))
                lines.append("")
                lines.append("</details>")
                lines.append("")
                lines.append("> No claims extracted from this field.")
                lines.append("")
                continue



            # Group claims by extraction_run_id (review-level run). Usually 1 group per review per experiment.
            run_map: dict[Any, list[dict[str, Any]]] = defaultdict(list)
            for c in field_claims:
                run_map[_get_run_id(c)].append(c)

            for run_id, items in sorted(run_map.items(), key=lambda x: str(x[0])):
                lines.append(f"#### Extraction run `{run_id}`")
                lines.append("")

                # If your query provides run metadata, show it (optional)
                first = items[0]
                if first.get("config_hash"):
                    lines.append(f"- config_hash: `{first.get('config_hash')}`  ")
                if first.get("run_fields"):
                    lines.append(f"- fields: `{first.get('run_fields')}`  ")
                if first.get("run_model_id"):
                    lines.append(f"- model_id: `{first.get('run_model_id')}`  ")
                if first.get("run_min_confidence") is not None or first.get("run_min_challengeability") is not None:
                    lines.append(
                        f"- thresholds: min_confidence=`{first.get('run_min_confidence')}`, "
                        f"min_challengeability=`{first.get('run_min_challengeability')}`"
                    )
                lines.append("")

                for c in items:
                    claim_text = (c.get("claim_text") or "").strip()
                    conf = _get_conf(c)
                    chal = _get_chal(c)
                    cal_score = _get_cal_score(c)
                    cal_score_norm = _get_cal_score_norm(c)
                    cal_error = _get_cal_error(c)
                    cal_error_detail = _get_cal_error_detail(c)
                    badge = _score_badge(c)

                    best_sent, score = _best_sentence(raw_full, claim_text)

                    meta_parts = []
                    if conf is not None:
                        meta_parts.append(f"conf={conf}")
                    if chal is not None:
                        meta_parts.append(f"chal={chal}")
                    meta_parts.append(f"cal_score={_fmt_cal_score(c)}")
                    if cal_score_norm is not None:
                        meta_parts.append(f"cal_score_norm={_fmt_num(cal_score_norm)}")

                    lines.append(
                        f"- **Claim #{c.get('claim_index')}** {badge} "
                        f"({', '.join(meta_parts)})"
                    )

                    cid = _get_claim_id(c)
                    if cid is not None:
                        lines.append(f"  - claim_id: `{cid}`")

                    lines.append(f"  - {claim_text}")
                    lines.append(f"  - calibrated_label: {_score_label(c)}")

                    if c.get("verbatim_quote"):
                        lines.append(f"  - verbatim_quote: {c.get('verbatim_quote')}")
                    if c.get("category"):
                        lines.append(f"  - category: `{c.get('category')}`")
                    if c.get("claim_type"):
                        lines.append(f"  - claim_type: `{c.get('claim_type')}`")

                    if c.get("binary_question"):
                        lines.append(f"  - binary_question: {c.get('binary_question')}")
                    if c.get("why_challengeable"):
                        lines.append(f"  - why_challengeable: {c.get('why_challengeable')}")

                    if cal_error:
                        lines.append(f"  - calibration_error: `{cal_error}`")
                    if cal_error_detail:
                        lines.append(f"  - calibration_error_detail: `{html_escape(str(cal_error_detail))}`")
                    if c.get("calibrated_score_raw_text"):
                        lines.append(f"  - calibrated_score_raw_text: `{html_escape(str(c.get('calibrated_score_raw_text')))}`")

                    if best_sent:
                        lines.append(f"  - most_relevant_sentence (score={_fmt_num(score)}): {best_sent}")
                    else:
                        lines.append("  - most_relevant_sentence: *(not found)*")

                    if best_sent:
                        excerpt = _highlight_many_sentences(
                            raw_full,
                            [(best_sent, [f"C{c.get('claim_index')}"])],
                            multi_color=False,
                        )
                        excerpt = _shorten(excerpt, raw_max_chars * 2)
                        lines.append("")
                        lines.append("  <details>")
                        lines.append("  <summary>Raw excerpt with claim highlight</summary>")
                        lines.append("")
                        lines.append("  " + excerpt.replace("\n", "\n  "))
                        lines.append("")
                        lines.append("  </details>")

                    lines.append("")

    return "\n".join(lines)





# ----------------------------
# Renderer: retrieval
# ----------------------------

def render_retrieval_markdown(
    *,
    paper_id: str,
    paper_meta: dict[str, Any] | None,
    parsing: dict[str, Any],
    chunking: dict[str, Any],
    indexing: dict[str, Any],
    retrieval_preview: list[dict[str, Any]],
) -> str:
    """
    Retrieval report:
      - Parsing
      - Chunking
      - Index / Embedding
      - Retrieval preview
    """
    from datetime import datetime
    from collections import Counter, defaultdict

    paper_meta = paper_meta or {}
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    title = (paper_meta.get("title") or "").strip()
    forum = (paper_meta.get("forum") or "").strip()
    pdf_path = (paper_meta.get("pdf_path") or "").strip()

    lines: list[str] = []

    head = f"# Retrieval Report — {paper_id}"
    if title:
        head += f" — {title}"
    lines.append(head)
    lines.append("")
    lines.append(f"- Generated at: {now}")
    if title:
        lines.append(f"- Paper title: `{title}`")
    if forum:
        lines.append(f"- Forum: `{forum}`")
    if pdf_path:
        lines.append(f"- PDF path: `{pdf_path}`")
        pdf_u = file_url(pdf_path)
        lines.append(f"- PDF link: [{pdf_u}]({pdf_u})")
    lines.append("")

    # ----------------------------
    # Pipeline Summary
    # ----------------------------
    lines.append("## Pipeline Summary")
    lines.append("")
    lines.append(f"- Pages parsed: **{parsing.get('num_pages', 0)}**")
    lines.append(f"- Sections detected: **{parsing.get('num_sections', 0)}**")
    lines.append(f"- Chunks generated: **{chunking.get('num_chunks', 0)}**")
    lines.append(f"- Chunk size: **{chunking.get('chunk_size', '')}**")
    lines.append(f"- Chunk overlap: **{chunking.get('chunk_overlap', '')}**")
    lines.append(f"- Indexed chunks: **{indexing.get('indexed_chunks', 0)}**")
    lines.append(f"- Embedded chunks: **{indexing.get('embedded_chunks', 0)}**")
    lines.append(f"- Embedding model: **{indexing.get('embedding_model') or '(none)'}**")
    lines.append("")

    # ----------------------------
    # Parsing
    # ----------------------------
    lines.append("## 1. Parsing")
    lines.append("")
    sections = parsing.get("sections", []) or []

    if not sections:
        lines.append("> No parsed sections found.")
        lines.append("")
    else:
        lines.append("### Detected Sections")
        lines.append("")
        lines.append("| # | Section | Start Page | Length (chars) |")
        lines.append("|---:|---|---:|---:|")
        for idx, sec in enumerate(sections, start=1):
            heading = sec.get("heading") or "unknown"
            page = sec.get("page") or ""
            text = sec.get("text") or ""
            lines.append(f"| {idx} | `{heading}` | {page} | {len(text)} |")
        lines.append("")

        lines.append("### Paper Structure Map")
        lines.append("")
        page_to_sections: dict[int, list[str]] = defaultdict(list)
        for sec in sections:
            page = sec.get("page")
            heading = sec.get("heading") or "unknown"
            if isinstance(page, int):
                page_to_sections[page].append(heading)

        if page_to_sections:
            lines.append("```")
            for page in sorted(page_to_sections.keys()):
                joined = " | ".join(page_to_sections[page])
                lines.append(f"p{page:<3} [{joined}]")
            lines.append("```")
            lines.append("")

    # ----------------------------
    # Chunking
    # ----------------------------
    lines.append("## 2. Chunking")
    lines.append("")
    lines.append(f"- chunk_size: `{chunking.get('chunk_size')}`")
    lines.append(f"- chunk_overlap: `{chunking.get('chunk_overlap')}`")
    lines.append(f"- Total chunks: **{chunking.get('num_chunks', 0)}**")
    lines.append("")

    section_chunk_counts = chunking.get("section_chunk_counts", {}) or {}
    if section_chunk_counts:
        lines.append("### Chunk Distribution by Section")
        lines.append("")
        lines.append("| Section | #Chunks |")
        lines.append("|---|---:|")
        for sec, cnt in sorted(section_chunk_counts.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"| `{sec}` | {cnt} |")
        lines.append("")

        lines.append("```")
        for sec, cnt in sorted(section_chunk_counts.items(), key=lambda x: (-x[1], x[0])):
            bar = "█" * min(int(cnt), 30)
            lines.append(f"{sec:<20} {bar}")
        lines.append("```")
        lines.append("")

    example_chunks = chunking.get("example_chunks", []) or []
    if example_chunks:
        lines.append("### Example Chunks")
        lines.append("")
        for ch in example_chunks:
            lines.append(f"#### Chunk #{ch.get('chunk_index')}")
            lines.append("")
            lines.append(f"- chunk_id: `{ch.get('id')}`")
            lines.append(f"- section: `{ch.get('section')}`")
            lines.append(f"- page: `{ch.get('page')}`")
            lines.append(f"- chars: `{ch.get('char_start')}–{ch.get('char_end')}`")
            lines.append("")
            lines.append("```")
            lines.append((ch.get("text") or "").strip())
            lines.append("```")
            lines.append("")

    # ----------------------------
    # Index / Embedding
    # ----------------------------
    lines.append("## 3. Index / Embedding")
    lines.append("")
    lines.append(f"- Indexed chunks: **{indexing.get('indexed_chunks', 0)}**")
    lines.append(f"- Embedded chunks: **{indexing.get('embedded_chunks', 0)}**")
    lines.append(f"- Embedding model: **{indexing.get('embedding_model') or '(none)'}**")
    lines.append("")

    indexed_examples = indexing.get("indexed_examples", []) or []
    if indexed_examples:
        lines.append("### Indexed Chunk Examples")
        lines.append("")
        lines.append("| Chunk ID | Section | Page | Embedded | Preview |")
        lines.append("|---|---|---:|---|---|")
        for ch in indexed_examples:
            preview = (ch.get("text") or "").replace("\n", " ").strip()
            if len(preview) > 80:
                preview = preview[:80] + "…"
            lines.append(
                f"| `{ch.get('id')}` | `{ch.get('section')}` | {ch.get('page')} | "
                f"{'yes' if ch.get('embedded') else 'no'} | {preview} |"
            )
        lines.append("")

    # ----------------------------
    # Retrieval Preview
    # ----------------------------
    lines.append("## 4. Retrieval Preview")
    lines.append("")

    if not retrieval_preview:
        lines.append("> No retrieval preview rows found.")
        lines.append("")
    else:
        for idx, item in enumerate(retrieval_preview, start=1):
            claim_text = (item.get("claim_text") or "").strip()
            claim_id = item.get("claim_id")
            results = item.get("results", []) or []

            lines.append(f"### Query #{idx}")
            lines.append("")
            if claim_id:
                lines.append(f"- claim_id: `{claim_id}`")
            lines.append("")
            lines.append("```")
            lines.append(claim_text)
            lines.append("```")
            lines.append("")

            if not results:
                lines.append("_No retrieval results found._")
                lines.append("")
                continue

            lines.append("| Rank | Section | Page | BM25 | Semantic | Score |")
            lines.append("|---:|---|---:|---:|---:|---:|")
            for r_i, r in enumerate(results, start=1):
                lines.append(
                    f"| {r_i} | `{r.get('section')}` | {r.get('page')} | "
                    f"{float(r.get('bm25_score') or 0.0):.3f} | "
                    f"{float(r.get('semantic_score') or 0.0):.3f} | "
                    f"{float(r.get('combined_score') or 0.0):.3f} |"
                )
            lines.append("")

            lines.append("#### Evidence Snippets")
            lines.append("")
            for r_i, r in enumerate(results, start=1):
                lines.append(f"##### Rank #{r_i}")
                lines.append("")
                lines.append(f"- chunk_id: `{r.get('chunk_id')}`")
                lines.append(f"- section: `{r.get('section')}`")
                lines.append(f"- page: `{r.get('page')}`")
                lines.append("")
                lines.append("```")
                lines.append((r.get("text") or "").strip())
                lines.append("```")
                lines.append("")

        sec_counter = Counter()
        for item in retrieval_preview:
            for r in item.get("results", []) or []:
                sec_counter[r.get("section") or "unknown"] += 1

        if sec_counter:
            lines.append("## 5. Retrieval Coverage")
            lines.append("")
            lines.append("| Section | Retrieved Count |")
            lines.append("|---|---:|")
            for sec, cnt in sec_counter.most_common():
                lines.append(f"| `{sec}` | {cnt} |")
            lines.append("")
            lines.append("```")
            for sec, cnt in sec_counter.most_common():
                lines.append(f"{sec:<20} {'█' * min(cnt, 30)}")
            lines.append("```")
            lines.append("")

    return "\n".join(lines)