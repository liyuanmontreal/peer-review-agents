from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True, slots=True)
class ParsedReference:
    ref_type: str  # table / figure / section
    ref_label: str
    ref_number: str


_TABLE_RE = re.compile(r"\b(?:Table|Tab\.)\s+(\d+(?:\.\d+)*)\b", re.IGNORECASE)
_FIG_RE = re.compile(r"\b(?:Figure|Fig\.)\s+(\d+(?:\.\d+)*)\b", re.IGNORECASE)
_SEC_RE = re.compile(r"\b(?:Section|Sec\.)\s+(\d+(?:\.\d+)*)\b", re.IGNORECASE)


def normalize_reference_label(ref_type: str, number: str) -> str:
    title = {
        "table": "Table",
        "figure": "Figure",
        "section": "Section",
    }[ref_type]
    return f"{title} {number}"


def extract_explicit_references(text: str | None) -> list[ParsedReference]:
    text = text or ""
    out: list[ParsedReference] = []
    for number in _TABLE_RE.findall(text):
        out.append(ParsedReference("table", normalize_reference_label("table", number), number))
    for number in _FIG_RE.findall(text):
        out.append(ParsedReference("figure", normalize_reference_label("figure", number), number))
    for number in _SEC_RE.findall(text):
        out.append(ParsedReference("section", normalize_reference_label("section", number), number))
    # stable unique
    seen: set[tuple[str, str]] = set()
    dedup: list[ParsedReference] = []
    for ref in out:
        key = (ref.ref_type, ref.ref_number)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(ref)
    return dedup


def _section_parent_numbers(num: str) -> Iterable[str]:
    parts = num.split(".")
    for i in range(len(parts), 0, -1):
        yield ".".join(parts[:i])


def compute_reference_boost(query_text: str, evidence_objects: list[dict]) -> dict[str, float]:
    refs = extract_explicit_references(query_text)
    boosts: dict[str, float] = {}
    if not refs:
        return boosts

    for ev in evidence_objects:
        ev_id = ev["id"]
        label = (ev.get("label") or "").strip().lower()
        obj_type = (ev.get("object_type") or "").strip().lower()
        sec_num = str(ev.get("section_number") or "").strip()
        score = 0.0

        for ref in refs:
            if ref.ref_type in {"table", "figure"}:
                if obj_type == ref.ref_type and label == ref.ref_label.lower():
                    score = max(score, 1.0)
                elif label == ref.ref_label.lower():
                    score = max(score, 0.9)
            elif ref.ref_type == "section":
                if sec_num:
                    if sec_num == ref.ref_number:
                        score = max(score, 0.7)
                    elif sec_num in list(_section_parent_numbers(ref.ref_number))[1:]:
                        score = max(score, 0.4)
                section = str(ev.get("section") or "").lower()
                if f"section {ref.ref_number}" in section:
                    score = max(score, 0.7)

        if score > 0:
            boosts[ev_id] = score
    return boosts
