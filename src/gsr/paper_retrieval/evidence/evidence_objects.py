from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import json


@dataclass(slots=True)
class EvidenceObject:
    id: str
    paper_id: str
    object_type: str  # text_chunk / table / figure
    retrieval_text: str
    label: str | None = None
    page: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    section_number: str | None = None
    caption_text: str | None = None
    content_text: str | None = None
    bbox: list[float] | None = None
    span_ids: list[str] | None = None
    asset_path: str | None = None
    metadata: dict[str, Any] | None = None

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        d["bbox_json"] = json.dumps(d.pop("bbox"), ensure_ascii=False) if d.get("bbox") is not None else None
        d["span_ids_json"] = json.dumps(d.pop("span_ids"), ensure_ascii=False) if d.get("span_ids") is not None else None
        d["metadata_json"] = json.dumps(d.pop("metadata"), ensure_ascii=False) if d.get("metadata") is not None else None
        return d


TEXT_CHUNK = "text_chunk"
TABLE = "table"
FIGURE = "figure"
