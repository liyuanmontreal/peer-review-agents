from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import quote as urlquote


_ILLEGAL_WIN_CHARS = r'[\\/*?:"<>|]'


def safe_filename(s: str) -> str:
    """Make a string safe for Windows filename."""
    s = (s or "").strip()
    s = re.sub(_ILLEGAL_WIN_CHARS, "_", s)
    s = re.sub(r"\s+", " ", s)
    return s


def md_escape(s: str | None) -> str:
    return (s or "").replace("\r", "").strip()


def as_quote_block(text: str | None) -> str:
    t = (text or "").strip()
    if not t:
        return "> (empty)\n"
    return "> " + "\n> ".join(t.splitlines()) + "\n"


def parse_chunk_keys(raw: str | None) -> list[str]:
    """
    Parse evidence_chunk_ids from:
      - JSON array: ["id1","id2"] or [1,2]
      - comma-separated: id1,id2
    Return list[str].
    """
    if not raw:
        return []
    raw = raw.strip()
    if not raw:
        return []

    # JSON list
    try:
        arr = json.loads(raw)
        if isinstance(arr, list):
            return [str(x) for x in arr if str(x).strip()]
    except Exception:
        pass

    # CSV
    return [p.strip() for p in raw.split(",") if p.strip()]


def file_url(path_str: str) -> str:
    """
    Build a file:// URL (clickable in some markdown renderers).
    """
    p = Path(path_str).resolve()
    s = str(p).replace("\\", "/")
    parts = s.split("/")
    encoded = "/".join(urlquote(x) for x in parts)
    return f"file:///{encoded}"


def file_url_page(path_str: str, page_1_indexed: int) -> str:
    return f"{file_url(path_str)}#page={page_1_indexed}"