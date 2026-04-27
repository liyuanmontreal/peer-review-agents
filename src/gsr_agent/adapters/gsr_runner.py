"""GSR adapter — safe bridge between gsr_agent and the src/gsr pipeline.

Access tiers
------------
Tier 1 — Koala API fields only (no GSR workspace required):
    index_paper_for_koala(paper)           Build PaperIndex from Koala Paper fields.
    get_seed_evidence_candidates(index)    Yield candidates from abstract (used by
                                           seed_comment.py; no DB required).

Tier 2 — GSR workspace (requires an indexed paper in gsr.db):
    check_gsr_available()                  Probe module imports; return GSRAvailability.
    get_gsr_workspace(default)             Resolve workspace path from arg / env / default.
    ensure_paper_indexed_for_koala()       Check/report paper index state in GSR DB.
    get_paper_summary_sections()           Read section text from GSR DB chunks.
    get_seed_evidence_candidates_from_gsr() Read evidence objects from GSR DB.

Phase 5 placeholders (do NOT implement yet):
    extract_claims_from_koala_comment()    Phase 5: LLM claim extraction.
    retrieve_and_verify_claims()           Phase 5: full RAG verification pipeline.

Isolation rule: gsr.* imports only inside functions — never at module scope.
This keeps gsr_agent importable even when GSR's heavy dependencies are absent.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..koala.models import Paper

log = logging.getLogger(__name__)

_DEFAULT_WORKSPACE = Path("./workspace")

# GSR modules required for Tier 2 operations.
_REQUIRED_GSR_MODULES = (
    "gsr",
    "gsr.config",
    "gsr.paper_retrieval",
    "gsr.claim_extraction",
    "gsr.claim_verification",
    "gsr.data_collection",
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GSRAvailability:
    """Result of check_gsr_available()."""
    ok: bool
    missing_modules: List[str] = field(default_factory=list)
    message: str = ""


@dataclass
class PaperIndex:
    """Lightweight index of a paper's content for commenting decisions."""
    paper_id: str
    title: str
    abstract: str
    sections: Dict[str, str]
    domains: List[str] = field(default_factory=list)


@dataclass
class PaperSummarySections:
    """Structured text sections read from the GSR DB or API fields."""
    paper_id: str
    ok: bool
    sections: Dict[str, str] = field(default_factory=dict)
    workspace: Optional[Path] = None
    message: str = ""


@dataclass
class SeedEvidenceCandidate:
    """A candidate claim or observation that could anchor a seed comment."""
    claim: str
    location: str      # e.g. "abstract", "introduction", "section 3"
    confidence: float  # 0.0–1.0


@dataclass
class GSRAdapterResult:
    """Generic result wrapper for GSR adapter operations."""
    ok: bool
    paper_id: str = ""
    workspace: Optional[Path] = None
    message: str = ""
    details: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tier 1 — Koala API fields only
# ---------------------------------------------------------------------------

def index_paper_for_koala(paper: Paper) -> PaperIndex:
    """Build a PaperIndex from a Koala Paper.

    Phase 4A: uses API-provided fields directly. Phase 5 will add PDF parsing.
    """
    return PaperIndex(
        paper_id=paper.paper_id,
        title=paper.title,
        abstract=paper.abstract,
        sections=_parse_sections(paper.full_text),
        domains=list(paper.domains),
    )


def get_seed_evidence_candidates(index: PaperIndex) -> List[SeedEvidenceCandidate]:
    """Return evidence candidates from a PaperIndex (Tier 1, abstract-based).

    Used by seed_comment.py. Returns the abstract as a single candidate when
    available. Phase 5 will add LLM-based claim extraction from full text.
    """
    if not index.abstract:
        return []
    return [
        SeedEvidenceCandidate(
            claim=index.abstract,
            location="abstract",
            confidence=0.5,
        )
    ]


# ---------------------------------------------------------------------------
# Tier 2 — GSR workspace
# ---------------------------------------------------------------------------

def check_gsr_available() -> GSRAvailability:
    """Probe whether all required GSR modules can be imported.

    Returns GSRAvailability(ok=True) when all modules are importable.
    Lists any missing modules without raising.
    """
    import importlib
    missing = []
    for mod in _REQUIRED_GSR_MODULES:
        try:
            importlib.import_module(mod)
        except ImportError as exc:
            log.debug("[check_gsr_available] cannot import %r: %s", mod, exc)
            missing.append(mod)

    if missing:
        return GSRAvailability(
            ok=False,
            missing_modules=missing,
            message=f"Missing GSR modules: {', '.join(missing)}",
        )
    return GSRAvailability(ok=True, message="All GSR modules available.")


def get_gsr_workspace(default: Optional[Path] = None) -> Path:
    """Resolve the GSR workspace directory.

    Resolution order:
      1. ``default`` argument (when not None)
      2. ``GSR_WORKSPACE`` environment variable
      3. ``./workspace`` (repository-relative default)
    """
    if default is not None:
        return Path(default).resolve()
    env_ws = os.environ.get("GSR_WORKSPACE", "")
    if env_ws:
        return Path(env_ws).resolve()
    return _DEFAULT_WORKSPACE.resolve()


def ensure_paper_indexed_for_koala(
    paper: Paper,
    workspace: Optional[Path] = None,
    *,
    force: bool = False,
) -> GSRAdapterResult:
    """Check whether a paper is indexed in the GSR DB.

    Does NOT run indexing — that is a separate, expensive operation.
    Reports one of three states:
      - already_indexed: paper_chunks rows exist for this paper_id
      - not_indexed:     paper_chunks table exists but has no rows for paper_id
      - db_missing:      gsr.db does not exist at the resolved workspace path
      - error:           unexpected failure probing the DB

    Args:
        paper:     Koala Paper to check.
        workspace: GSR workspace path. Resolved via get_gsr_workspace() if None.
        force:     Ignored in Phase 5A-prep (reserved for future re-index trigger).

    Returns:
        GSRAdapterResult with ok=True when already indexed.
    """
    ws = get_gsr_workspace(workspace)
    db_path = ws / "gsr.db"
    paper_id = paper.paper_id

    if not db_path.exists():
        return GSRAdapterResult(
            ok=False,
            paper_id=paper_id,
            workspace=ws,
            message=f"gsr.db not found at {db_path}",
            details={"status": "db_missing", "db_path": str(db_path)},
        )

    try:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM paper_chunks WHERE paper_id=?",
                (paper_id,),
            ).fetchone()
            count = int(row[0]) if row else 0
        finally:
            conn.close()
    except sqlite3.OperationalError as exc:
        return GSRAdapterResult(
            ok=False,
            paper_id=paper_id,
            workspace=ws,
            message=f"DB error checking paper_chunks: {exc}",
            details={"status": "error", "error": str(exc)},
        )

    if count > 0:
        return GSRAdapterResult(
            ok=True,
            paper_id=paper_id,
            workspace=ws,
            message=f"Paper {paper_id!r} is indexed ({count} chunks).",
            details={"status": "already_indexed", "chunk_count": count},
        )

    return GSRAdapterResult(
        ok=False,
        paper_id=paper_id,
        workspace=ws,
        message=f"Paper {paper_id!r} is not yet indexed in GSR DB.",
        details={"status": "not_indexed", "chunk_count": 0},
    )


def get_paper_summary_sections(
    paper_id: str,
    workspace: Optional[Path] = None,
) -> PaperSummarySections:
    """Read paper section text from the GSR DB (Tier 2).

    Queries ``paper_chunks`` and groups text by section heading.
    Returns an empty PaperSummarySections (not an exception) when the DB or
    the paper is missing.
    """
    ws = get_gsr_workspace(workspace)
    db_path = ws / "gsr.db"

    if not db_path.exists():
        return PaperSummarySections(
            paper_id=paper_id,
            ok=False,
            workspace=ws,
            message=f"gsr.db not found at {db_path}",
        )

    try:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        try:
            rows = conn.execute(
                "SELECT section, text FROM paper_chunks WHERE paper_id=? ORDER BY chunk_index",
                (paper_id,),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.OperationalError as exc:
        return PaperSummarySections(
            paper_id=paper_id,
            ok=False,
            workspace=ws,
            message=f"DB error reading paper_chunks: {exc}",
        )

    if not rows:
        return PaperSummarySections(
            paper_id=paper_id,
            ok=False,
            workspace=ws,
            message=f"No chunks found for paper_id={paper_id!r}",
        )

    sections: Dict[str, str] = {}
    for section, text in rows:
        key = section or "body"
        if key in sections:
            sections[key] += " " + text
        else:
            sections[key] = text

    return PaperSummarySections(
        paper_id=paper_id,
        ok=True,
        sections=sections,
        workspace=ws,
        message=f"Loaded {len(sections)} sections ({len(rows)} chunks).",
    )


def get_seed_evidence_candidates_from_gsr(
    paper_id: str,
    workspace: Optional[Path] = None,
    *,
    max_candidates: int = 5,
) -> List[SeedEvidenceCandidate]:
    """Read evidence candidates from the GSR DB (Tier 2).

    Queries ``evidence_objects`` for text_chunk, table, and figure objects.
    Returns an empty list (not an exception) when the DB or paper is missing.

    Phase 5 will wire this into the seed comment generation path.
    """
    ws = get_gsr_workspace(workspace)
    db_path = ws / "gsr.db"

    if not db_path.exists():
        log.debug(
            "[get_seed_evidence_candidates_from_gsr] gsr.db not found at %s", db_path
        )
        return []

    try:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        try:
            rows = conn.execute(
                """SELECT object_type, section, retrieval_text, caption_text
                   FROM evidence_objects
                   WHERE paper_id=?
                   ORDER BY object_type, page
                   LIMIT ?""",
                (paper_id, max_candidates),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.OperationalError as exc:
        log.debug(
            "[get_seed_evidence_candidates_from_gsr] DB error for %r: %s", paper_id, exc
        )
        return []

    candidates = []
    for obj_type, section, retrieval_text, caption_text in rows:
        text = retrieval_text or caption_text or ""
        if not text:
            continue
        location = section or obj_type or "unknown"
        candidates.append(
            SeedEvidenceCandidate(
                claim=text,
                location=location,
                confidence=0.6 if obj_type == "text_chunk" else 0.4,
            )
        )
    return candidates


# ---------------------------------------------------------------------------
# Phase 5A — claim extraction and verification
# ---------------------------------------------------------------------------

def extract_claims_from_koala_comment(
    comment: str,
    paper_id: str,
    workspace: Optional[Path] = None,
    *,
    max_claims: int = 8,
) -> List[dict]:
    """Extract challengeable claims from a citable Koala comment via GSR.

    Uses GSR's field-mode claim extractor directly (no SQLite writes).
    Returns up to max_claims claim dicts, each with keys: id, claim_text,
    verbatim_quote, claim_type, confidence, challengeability, category,
    binary_question, why_challengeable.

    Returns an empty list when GSR modules are unavailable or the comment
    yields no verifiable claims.
    """
    try:
        from gsr.claim_extraction.extractor import _extract_claims_from_field, _config_hash
        from gsr.claim_extraction.llm import get_model_id
    except ImportError as exc:
        log.warning("[extract_claims_from_koala_comment] GSR unavailable: %s", exc)
        return []

    model_id = get_model_id(None)
    cfg_hash = _config_hash(
        model_id=model_id,
        fields=("body",),
        min_confidence=0.5,
        min_challengeability=0.5,
    )

    try:
        claims = _extract_claims_from_field(
            review_id=f"koala_comment_{paper_id}",
            paper_title="",
            field_name="body",
            review_text=comment,
            cfg_hash=cfg_hash,
            model=None,
            min_confidence=0.5,
            min_challengeability=0.5,
        )
    except Exception as exc:
        log.warning("[extract_claims_from_koala_comment] extraction failed: %s", exc)
        return []

    return claims[:max_claims]


def retrieve_and_verify_claims(
    paper_id: str,
    extracted_claims: List[dict],
    workspace: Optional[Path] = None,
    *,
    top_k: int = 5,
) -> List[dict]:
    """Verify extracted claims against paper_chunks evidence from the GSR DB.

    Loads up to top_k text chunks for paper_id from gsr.db and calls
    verify_claim() for each claim.  Returns a list of verification result
    dicts with keys: id, claim_id, paper_id, review_id, verdict, confidence,
    reasoning, supporting_quote, evidence, model_id, status, error.

    Returns an empty list when GSR modules are unavailable, the DB is missing,
    or extracted_claims is empty.
    """
    if not extracted_claims:
        return []

    try:
        from gsr.claim_verification.verifier import verify_claim
    except ImportError as exc:
        log.warning("[retrieve_and_verify_claims] GSR unavailable: %s", exc)
        return []

    ws = get_gsr_workspace(workspace)
    db_path = ws / "gsr.db"

    evidence_chunks: List[dict] = []
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            try:
                rows = conn.execute(
                    """SELECT id AS chunk_id, text, section, page
                       FROM paper_chunks
                       WHERE paper_id=?
                       ORDER BY chunk_index
                       LIMIT ?""",
                    (paper_id, top_k),
                ).fetchall()
                for row in rows:
                    evidence_chunks.append({
                        "chunk_id": row[0],
                        "text": row[1],
                        "section": row[2],
                        "page": row[3],
                        "object_type": "text_chunk",
                        "score": 1.0,
                    })
            finally:
                conn.close()
        except sqlite3.OperationalError as exc:
            log.warning("[retrieve_and_verify_claims] DB error loading chunks: %s", exc)

    results = []
    for claim in extracted_claims:
        claim_for_verify = {
            "id": claim.get("id", ""),
            "paper_id": claim.get("paper_id", paper_id),
            "review_id": claim.get("review_id", ""),
            "claim_text": claim.get("claim_text", ""),
            "verbatim_quote": claim.get("verbatim_quote", ""),
            "claim_type": claim.get("claim_type", "factual"),
            "confidence": claim.get("confidence", 0.5),
        }
        try:
            result = verify_claim(claim_for_verify, evidence_chunks)
            results.append(result)
        except Exception as exc:
            log.warning(
                "[retrieve_and_verify_claims] verify_claim failed for %r: %s",
                claim_for_verify["id"], exc,
            )
            results.append({
                "id": claim_for_verify["id"],
                "claim_id": claim_for_verify["id"],
                "paper_id": paper_id,
                "review_id": claim_for_verify["review_id"],
                "verdict": "insufficient_evidence",
                "confidence": 0.0,
                "reasoning": f"Verification error: {exc}",
                "supporting_quote": None,
                "evidence": [],
                "model_id": "",
                "status": "error",
                "error": str(exc),
            })

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_sections(full_text: str) -> Dict[str, str]:
    """Split full_text into a sections dict. Phase 4A: returns empty dict."""
    return {}
