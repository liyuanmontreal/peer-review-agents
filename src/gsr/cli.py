'''
src.gsr.cli
Command-line interface for GSR. Provides subcommands for each module and a "run" command for the full pipeline.

Key design:
- Workspace is resolved in main() (env var override) BEFORE importing gsr.config.
- Subcommand handlers should rely on init_db() / gsr.config.DB_PATH (workspace-aware),
  rather than computing DB paths themselves.

'''

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys

from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Iterable, List, Optional
import sqlite3
from gsr.utils.timing import timed

from gsr.reporting.exporter import export_retrieval_report

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_latest_experiment_id(conn: sqlite3.Connection, paper_id: str) -> str | None:
    row = conn.execute(
        """
        SELECT er.experiment_id
        FROM extraction_runs er
        WHERE er.paper_id = ?
          AND er.experiment_id IS NOT NULL
        ORDER BY er.started_at DESC, er.id DESC
        LIMIT 1
        """,
        (paper_id,),
    ).fetchone()
    return row[0] if row else None

def _read_paper_ids_file(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--paper-ids-file not found: {p}")
    ids: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        ids.append(s)
    # stable unique while preserving order
    seen = set()
    out: List[str] = []
    for pid in ids:
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
    return out


def _paper_ids_from_venue(conn: sqlite3.Connection, venue_id: str) -> List[str]:
    rows = conn.execute(
        "SELECT id FROM papers WHERE venue_id = ? ORDER BY number ASC, id ASC",
        (venue_id,),
    ).fetchall()
    return [r[0] for r in rows]


def _resolve_extract_paper_ids(
    conn: sqlite3.Connection,
    *,
    paper_id: Optional[str],
    venue: Optional[str],
    paper_ids_file: Optional[str],
) -> List[str]:
    if paper_id:
        return [paper_id]
    if paper_ids_file:
        return _read_paper_ids_file(paper_ids_file)
    if venue:
        return _paper_ids_from_venue(conn, venue)
    # default: all papers that have at least one review
    rows = conn.execute(
        "SELECT DISTINCT paper_id FROM reviews WHERE paper_id IS NOT NULL ORDER BY paper_id ASC"
    ).fetchall()
    return [r[0] for r in rows]

def _resolve_report_paper_ids(
    conn: sqlite3.Connection,
    *,
    paper_id: Optional[str],
    venue: Optional[str],
    paper_ids_file: Optional[str],
) -> List[str]:
    """Resolve paper ids for reporting scope.

    For report batch export we typically want a stable order (by paper number when possible).
    """
    if paper_id:
        return [paper_id]
    if paper_ids_file:
        return _read_paper_ids_file(paper_ids_file)
    if venue:
        return _paper_ids_from_venue(conn, venue)
    # default: all papers
    rows = conn.execute("SELECT id FROM papers ORDER BY number ASC, id ASC").fetchall()
    return [r[0] for r in rows]



def _load_env() -> None:
    """Load .env file from repo root (if present).

    IMPORTANT: do NOT import gsr.config here. Workspace overrides must be applied
    BEFORE gsr.config is imported so DB_PATH is computed correctly.
    """
    repo_root = Path(__file__).resolve().parents[2]
    env_file = repo_root / ".env"
    load_dotenv(env_file)


def _resolve_dir(repo_root: Path, value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p.resolve()
    return (repo_root / p).resolve()

def _pick_workspace_dir(args: argparse.Namespace, repo_root: Path) -> tuple[Path | None, str | None]:
    """
    Decide workspace directory and its source label.

    Priority:
      1) --workspace
      2) --run-id (under --runs-dir)
      3) --latest-run (under --runs-dir)
      4) if command == run: auto-create a new run dir under --runs-dir
      5) else: None (caller will fall back to env/default)
    """
    # 1) explicit --workspace
    ws = getattr(args, "workspace", None)
    if ws:
        return _resolve_dir(repo_root, ws), "cli-workspace"

    runs_base = _resolve_dir(repo_root, getattr(args, "runs_dir", "runs"))

    # 2) explicit --run-id
    rid = getattr(args, "run_id", None)
    if rid:
        run_dir = (runs_base / rid).resolve()
        return run_dir, "cli-run-id"

    # 3) --latest-run
    if getattr(args, "latest_run", False):
        if runs_base.exists():
            candidates = [p for p in runs_base.iterdir() if p.is_dir()]
            if candidates:
                latest = max(candidates, key=lambda p: p.stat().st_mtime)
                return latest.resolve(), "cli-latest-run"
        # no candidates → fall through to default behavior

    # 4) run command auto-create
    if getattr(args, "command", None) == "run":
        url = getattr(args, "url", None)
        run_name = getattr(args, "run_name", None)
        run_id = getattr(args, "run_id", None) or _make_run_id(url=url, run_name=run_name)
        run_dir = (runs_base / run_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir, "cli-run-auto"

    # 5) default
    return None, None

def _safe_slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    return s[:80] if s else "run"


def _extract_forum_id_from_url(url: str) -> str | None:
    m = re.search(r"[?&]id=([^&]+)", url)
    return m.group(1) if m else None


def _make_run_id(url: str | None = None, run_name: str | None = None) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts: list[str] = [ts]
    if url:
        fid = _extract_forum_id_from_url(url)
        if fid:
            parts.append(_safe_slug(fid))
    if run_name:
        parts.append(_safe_slug(run_name))
    return "_".join(parts)


def _print_workspace_banner() -> None:
    """Print workspace and DB path with an explicit provenance label."""
    from gsr.config import DB_PATH, get_workspace_info

    ws_dir, ws_src = get_workspace_info()
    print(f"Workspace: {ws_dir} (source={ws_src})")
    print(f"DB: {DB_PATH}")

def _paper_ids_from_chunks(
    conn: sqlite3.Connection,
    *,
    paper_id: str | None,
    limit: int | None,
) -> list[str]:
    if paper_id:
        return [paper_id]

    rows = conn.execute(
        """
        SELECT DISTINCT paper_id
        FROM paper_chunks
        ORDER BY paper_id
        """
    ).fetchall()

    ids = [r[0] for r in rows]
    if limit:
        ids = ids[:limit]
    return ids


def _ensure_evidence_objects_for_paper(
    conn: sqlite3.Connection,
    *,
    paper_id: str,
    embedding_model_id: str | None,
    embed: bool,
    force: bool,
) -> dict[str, int | str | bool]:
    """
    Build and persist evidence_objects + evidence_embeddings for one paper.

    Returns a summary dict like:
      {
        "paper_id": ...,
        "objects": 123,
        "embedded": 123,
        "skipped": False,
      }
    """
    from gsr.paper_retrieval.evidence.evidence_builder import build_evidence_objects_for_paper
    from gsr.paper_retrieval.storage.storage import (
        load_chunks_for_paper,
        load_pdf_spans_for_paper,
        load_evidence_objects_for_paper,
        save_evidence_objects,
        save_evidence_embeddings,
        delete_evidence_objects_for_paper,
    )

    existing_objects = load_evidence_objects_for_paper(paper_id, conn)
    if existing_objects and not force:
        if not embed:
            return {
                "paper_id": paper_id,
                "objects": len(existing_objects),
                "embedded": 0,
                "skipped": True,
            }

        resolved_model_id = embedding_model_id or "allenai/specter2_base"
        cur = conn.execute(
            """
            SELECT COUNT(*)
            FROM evidence_embeddings ee
            JOIN evidence_objects eo ON ee.evidence_object_id = eo.id
            WHERE eo.paper_id = ? AND ee.model_id = ?
            """,
            (paper_id, resolved_model_id),
        )
        embedded_count = cur.fetchone()[0]
        if embedded_count > 0:
            return {
                "paper_id": paper_id,
                "objects": len(existing_objects),
                "embedded": embedded_count,
                "skipped": True,
            }

    chunks = load_chunks_for_paper(paper_id, conn)
    spans = load_pdf_spans_for_paper(paper_id, conn)

    if not chunks:
        return {
            "paper_id": paper_id,
            "objects": 0,
            "embedded": 0,
            "skipped": True,
        }

    objects = build_evidence_objects_for_paper(
        paper_id=paper_id,
        chunks=chunks,
        spans=spans,
    )

    if force:
        delete_evidence_objects_for_paper(
            paper_id,
            conn,
            model_id=embedding_model_id or "allenai/specter2_base",
        )

    save_evidence_objects(objects, conn)

    embedded = 0
    if embed and objects:
        from gsr.paper_retrieval.storage.embeddings import load_embedding_model

        resolved_model_id = embedding_model_id or "allenai/specter2_base"
        model, resolved_model_id = load_embedding_model(resolved_model_id)

        texts = [(getattr(o, "retrieval_text", None) or "") for o in objects]
        vectors = model.encode(texts)

        vectors = [
            v.tolist() if hasattr(v, "tolist") else v
            for v in vectors
        ]

        save_evidence_embeddings(
            [{"id": getattr(o, "id")} for o in objects],
            vectors,
            resolved_model_id,
            conn,
        )
        embedded = len(vectors)

    return {
        "paper_id": paper_id,
        "objects": len(objects),
        "embedded": embedded,
        "skipped": False,
    }

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    # SAFE to import extractor constants here (doesn't depend on workspace/db)
    from gsr.claim_extraction.extractor import MIN_CHALLENGEABILITY, MIN_CONFIDENCE

    p = argparse.ArgumentParser(
        prog="gsr",
        description="Good Samaritan Review — fetch, extract, and analyse OpenReview data.",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    # Global workspace/run settings
    p.add_argument(
        "--workspace",
        default=None,
        help=(
            "Workspace directory for runtime artifacts (db, json, pdf, csv). "
            "Overrides .env and config defaults. Can be absolute or relative to repo root."
        ),
    )
    p.add_argument(
        "--runs-dir",
        default="runs",
        help=(
            "Base directory to store per-run workspaces (default: runs). "
            "Used by `gsr run` when no --workspace is provided and GSR_WORKSPACE is not set."
        ),
    )

    p.add_argument(
        "--run-id", "--run_id",
        dest="run_id",
        default=None,
        help=(
            "Use an existing run directory under --runs-dir as workspace "
            "(e.g., --run-id iclr2024_limit100_v1)."
        ),
    )

    p.add_argument(
        "--latest-run", "--latest_run",
        dest="latest_run",
        action="store_true",
        help=(
            "Use the most recent run directory under --runs-dir as workspace "
            "(only if --workspace/--run-id not provided)."
        ),
    )

    sub = p.add_subparsers(dest="command")

    # -- fetch ---------------------------------------------------------------
    fetch_p = sub.add_parser("fetch", help="Fetch papers and reviews from OpenReview")
    source_group = fetch_p.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--venue", help="OpenReview venue ID (batch mode)")
    source_group.add_argument("--url", help="OpenReview forum URL (single-paper mode)")
    fetch_p.add_argument("--limit", type=int, default=None, help="Max number of papers to fetch (default: all)")
    fetch_p.add_argument("--json-only", action="store_true", help="Save only the JSON snapshot (skip SQLite)")
    fetch_p.add_argument("--db-only", action="store_true", help="Save only to SQLite (skip JSON snapshot)")
    fetch_p.add_argument("--pdf", action="store_true", help="Download PDF attachments for each paper")
    fetch_p.add_argument("--pdf-dir", default=None, help="Directory to store downloaded PDFs (default: <workspace>/pdf)")
    fetch_p.add_argument("--run-name", default=None, help="Optional human-friendly suffix for the generated run id (e.g., demo, exp1).")

    # -- extract -------------------------------------------------------------
    extract_p = sub.add_parser("extract", help="Extract claims from stored reviews")

    extract_scope = extract_p.add_mutually_exclusive_group(required=False)
    extract_scope.add_argument("--venue", default=None, help="Process reviews for all papers in this venue_id")
    extract_scope.add_argument( "--paper-ids-file", default=None, help="Path to a text file containing one paper_id per line") 
    extract_scope.add_argument("--paper-id", default=None, help="Only process reviews for this paper ID")

    extract_p.add_argument("--limit", type=int, default=None, help="Max number of reviews to process (default: all)") 
    extract_p.add_argument("--model", default=None, help="LLM model to use (default: GSR_LLM_MODEL env var or gpt-4o-mini)")
    extract_p.add_argument("--fields", default="weaknesses,strengths,questions,summary", help="Comma-separated review fields to extract from (default: weaknesses,strengths,questions,summary)",    )
    extract_p.add_argument("--force", action="store_true", help="Re-run extraction even if a successful run exists for the review.")
    extract_p.add_argument("--delay", type=float, default=0.1, help="Delay in seconds between LLM calls (default: 0.5)")
    extract_p.add_argument("--min-challengeability", type=float, default=MIN_CHALLENGEABILITY, help=f"Minimum challengeability to keep a claim (default: {MIN_CHALLENGEABILITY}).",    )
    extract_p.add_argument("--min-confidence", type=float,default=MIN_CONFIDENCE,  help=f"Minimum confidence to keep a claim (default: {MIN_CONFIDENCE}).",    )
    extract_p.add_argument("--score-claims", action="store_true", help="Enable post-extraction claim scoring with an external scorer model.",    )
    extract_p.add_argument("--scorer-model", default="meta-llama/Llama-3.1-8B", help="Model name used for claim scoring.",    )
    extract_p.add_argument("--max-workers", type=int, default=2,  help="Max concurrent review extraction workers (default: 2)",    )
    extract_p.add_argument("--extract-mode",choices=["field", "review","grouped"],default="field", help="Extraction granularity: field (default) or review (single call per review) or grouped (multiple claims per call).",)


    # -- retrieve ------------------------------------------------------------
    retrieve_p = sub.add_parser( "retrieve",  help="Build / refresh paper retrieval index (PDF parsing, chunking, spans, embeddings)"    )
    retrieve_p.add_argument("--paper-id", default=None, help="Index only this paper ID (default: all papers with a PDF)")
    retrieve_p.add_argument("--limit", type=int, default=None, help="Max number of papers to index (default: all)")

    # Default to V2-friendly settings
    retrieve_p.add_argument("--chunk-size",type=int, default=8,  help="Target chunk size in spans/lines for layout-aware V2 indexing (default: 8)",    )
    retrieve_p.add_argument("--chunk-overlap", type=int, default=2,  help="Overlap between consecutive chunks for layout-aware V2 indexing (default: 2)",    )
    retrieve_p.add_argument("--no-embed", action="store_true", help="Skip embedding (BM25-only retrieval, no sentence-transformers needed)")
    retrieve_p.add_argument("--embedding-model", default=None, help="HuggingFace model ID for embeddings (default: GSR_EMBEDDING_MODEL env var or allenai/specter2_base)")
    retrieve_p.add_argument("--force", action="store_true", help="Re-index papers that are already indexed")
    retrieve_p.add_argument("--layout-aware",dest="layout_aware", action="store_true",  default=True, help="Use layout-aware PDF parsing (PyMuPDF V2 spans) for grounding and evidence objects. (default: ON).",    )
    retrieve_p.add_argument("--no-layout-aware", dest="layout_aware", action="store_false", help="Disable layout-aware indexing and fall back to legacy V1 chunking.",    )
    retrieve_p.add_argument("--docling",action="store_true", help="Use Docling to enrich layout-aware parsing for figures/tables/sections during indexing.",)
    retrieve_p.add_argument("--figure-ocr", dest="figure_ocr", action="store_true", help="Run LightOnOCR-2 figure text recovery during evidence object construction (Phase 1 A/B flag).")
    retrieve_p.add_argument("--bbox-refine", dest="bbox_refine", action="store_true", default=False, help="Run GroundingDINO bbox refinement after evidence object construction (Phase P1.5). Refines figure/table bboxes for better PDF overlays. Requires transformers+torch. Default: off.")
    retrieve_p.add_argument("--bbox-detector", dest="bbox_detector", default="groundingdino", choices=["groundingdino", "rtdetr", "paddlex_layout"], help="Detector backend for --bbox-refine: 'groundingdino' (default), 'rtdetr', or 'paddlex_layout' (experimental, requires paddlex). Default: groundingdino.")
    retrieve_p.add_argument("--top-k", type=int, default=5, help="Number of evidence chunks to cache per claim")
    retrieve_p.add_argument("--experiment-id", default=None, help="Retrieve evidence for claims from this extraction experiment")
    retrieve_p.add_argument("--claims-only", action="store_true", help="Only build claim-level retrieval_results using existing paper index")
    retrieve_p.add_argument("--index-only", action="store_true", help="Only build paper index/chunks/embeddings, do not cache claim-level retrieval")
    # -- verify --------------------------------------------------------------
    verify_p = sub.add_parser("verify", help="Verify extracted claims against paper evidence (Module 4)")
    verify_p.add_argument("--paper-id", default=None, help="Only verify claims from this paper ID (default: all papers)")
    verify_p.add_argument("--review-id", default=None, help="Only verify claims from this review ID")
    verify_p.add_argument("--limit", type=int, default=None, help="Max number of claims to verify (default: all)")
    verify_p.add_argument("--model", default=None, help="LLM model to use (default: GSR_LLM_MODEL env var or gpt-4o-mini)")
    verify_p.add_argument("--top-k", type=int, default=5, help="Number of evidence chunks per claim (default: 5)")
    verify_p.add_argument("--embedding-model", default=None, help="Embedding model ID associated with retrieval cache / optional fallback live retrieval.")
    verify_p.add_argument("--force", action="store_true", help="Re-verify already-verified claims")
    verify_p.add_argument("--delay", type=float, default=0.1, help="Delay in seconds between LLM calls (default: 0.5)")
    verify_p.add_argument("--all-claims", action="store_true", help="Also verify non-verifiable (subjective) claims")
    verify_p.add_argument("--experiment-id", default=None, help="Verify only claims from this extraction experiment (default: latest experiment for the paper)")
    verify_p.add_argument( "--allow-live-retrieval",action="store_true",help="If cached evidence is missing, retrieve evidence on the fly (default: off)."    )
    verify_p.add_argument("--require-cached-evidence",action="store_true",default=True, help="Only verify claims with cached retrieval results (default: on)."    )
    verify_p.add_argument( "--max-workers", type=int, default=2, help="Number of parallel LLM verification workers (default: 2)",)
    verify_p.add_argument("--selective-figure-ocr", dest="selective_figure_ocr", action="store_true", default=False, help="Enable verify-time selective figure OCR escalation (Phase 2.0). OCRs only the top-ranked figure for claims that trigger escalation. Independent of --figure-ocr on retrieve.")
    verify_p.add_argument("--figure-ocr-heuristic", dest="figure_ocr_heuristic", action="store_true", default=False, help="Enable heuristic escalation paths (insufficient_evidence / low-confidence + rank + lexical gate) in addition to explicit figure references. Only relevant with --selective-figure-ocr. Experimental — off by default.")
    verify_p.add_argument("--figure-semantic", dest="figure_semantic", action="store_true", default=False, help="Enable verify-time selective Qwen2.5-VL figure semantic enrichment . Only claims with explicit figure references (Figure N / Fig. N) trigger enrichment by default.")
    verify_p.add_argument("--figure-semantic-top", dest="figure_semantic_top", action="store_true", default=False, help="Also enrich the top-ranked figure even without an explicit reference. Only relevant with --figure-semantic. Experimental — off by default.")
    # -- report --------------------------------------------------------------
    report_p = sub.add_parser("report", help="Generate reports")
    report_sub = report_p.add_subparsers(dest="report_type", required=True)

    # verification report
    r_ver = report_sub.add_parser("verification", help="Claim → Evidence → Verification report")
    r_ver.add_argument("--paper-id", required=True)
    r_ver.add_argument("--out", default=None)
    r_ver.add_argument("--limit", type=int, default=None)

    # annotate-claim report
    r_ann = report_sub.add_parser(
        "annotate-claim",
        help="Render an annotated PDF for one verified claim",
    )
    r_ann.add_argument("--claim-id", required=True)
    r_ann.add_argument("--output-dir", default=None)


    # extraction report
    r_ext = report_sub.add_parser("extraction", help="Review → extracted claims report")

    ext_scope = r_ext.add_mutually_exclusive_group(required=True)
    ext_scope.add_argument("--paper-id", help="Export report for a single paper_id")
    ext_scope.add_argument("--venue", help="Export reports for all papers in this venue_id")
    ext_scope.add_argument("--paper-ids-file", help="Text file with one paper_id per line")


    # retrieval report
    r_ret = report_sub.add_parser(        "retrieval",        help="Export retrieval report for one paper"    )
    r_ret.add_argument("--paper-id", required=True)
    r_ret .add_argument("--out", default=None)
    r_ret .add_argument("--preview-limit", type=int, default=3)
    r_ret .add_argument("--top-k", type=int, default=5)

    # single-paper output path
    r_ext.add_argument("--out", default=None, help="Output path for single-paper mode (default: <workspace>/output/reports/...)")
    # batch output directory (optional)
    r_ext.add_argument("--out-dir", default=None, help="Output directory for batch mode (default: <workspace>/output/reports/)")
    r_ext.add_argument("--limit", type=int, default=None, help="Limit number of papers to export (batch mode only)")

    # compare
    r_cmp = report_sub.add_parser("compare", help="Compare extraction variants")
    r_cmp.add_argument("--paper-id", required=True)
    r_cmp.add_argument("--out", default=None)

    # ocr-audit
    r_ocr = report_sub.add_parser("ocr-audit", help="Figure OCR audit report (Phase 1)")
    r_ocr.add_argument("--paper-id", default=None, help="Scope to a single paper (default: all papers)")
    r_ocr.add_argument("--out-dir", default=None, help="Output directory for ocr_audit.csv and ocr_audit.md (default: <workspace>/reports/)")

    # bbox-audit
    r_bbox_audit = report_sub.add_parser("bbox-audit", help="BBox refinement audit: summarise figure/table bbox sources after --bbox-refine")
    r_bbox_audit.add_argument("--paper-id", required=True, help="Paper to audit")
    r_bbox_audit.add_argument("--out-dir", default=None, help="Output directory (default: <workspace>/reports/)")
    r_bbox_audit.add_argument("--include-text", dest="include_text", action="store_true", default=False, help="Also include text_chunk objects (default: figures/tables only)")

    # object-audit
    r_obj_audit = report_sub.add_parser("object-audit", help="Object-level debug audit for figure/table evidence objects (false identity + mixed bbox diagnostics)")
    r_obj_audit.add_argument("--paper-id", required=True, help="Paper to audit")
    r_obj_audit.add_argument("--out-dir", default=None, help="Output directory (default: <workspace>/reports/)")

    # ocr-smoke
    r_smoke = report_sub.add_parser("ocr-smoke", help="Single-image OCR smoke test for debugging inference failures")
    r_smoke_scope = r_smoke.add_mutually_exclusive_group()
    r_smoke_scope.add_argument("--paper-id", default=None, help="Look up figure by paper-id + label")
    r_smoke_scope.add_argument("--pdf-path", default=None, help="Direct path to PDF (use with --page and --bbox)")
    r_smoke.add_argument("--label", default=None, help="Figure label to look up, e.g. 'Figure 3' (requires --paper-id)")
    r_smoke.add_argument("--page", type=int, default=None, help="1-indexed page number (use with --pdf-path and --bbox)")
    r_smoke.add_argument("--bbox", default=None, help="Bounding box as 'x0,y0,x1,y1' (use with --pdf-path and --page)")
    r_smoke.add_argument("--force-cpu", action="store_true", help="Force CPU inference (for isolating device/dtype failures)")
    r_smoke.add_argument("--caption", default=None, help="Optional caption text for duplicate-filter testing")

    # dino-smoke
    r_dino = report_sub.add_parser("dino-smoke", help="Single-page GroundingDINO smoke test for bbox refinement debugging")
    r_dino.add_argument("--paper-id", default=None, help="Resolve PDF path from paper_id in DB")
    r_dino.add_argument("--page", type=int, default=1, help="1-indexed page to test (default: 1)")
    r_dino.add_argument("--neighbor-pages", type=int, default=0, metavar="N", help="Also test N pages before and after --page (default: 0)")

    # bbox-smoke (Phase P1.6) — generalized single-page smoke test for any detector backend
    r_bbox_smoke = report_sub.add_parser("bbox-smoke", help="Single-page detector smoke test (supports --detector groundingdino|rtdetr)")
    r_bbox_smoke.add_argument("--paper-id", default=None, help="Resolve PDF path from paper_id in DB")
    r_bbox_smoke.add_argument("--page", type=int, default=1, help="1-indexed page to test (default: 1)")
    r_bbox_smoke.add_argument("--neighbor-pages", type=int, default=0, metavar="N", help="Also test N pages before and after --page (default: 0)")
    r_bbox_smoke.add_argument("--detector", default="groundingdino", choices=["groundingdino", "rtdetr", "paddlex_layout"], help="Detector backend: 'groundingdino', 'rtdetr', or 'paddlex_layout' (experimental, requires paddlex). Default: groundingdino.")

    # -- analyze -------------------------------------------------------------
    analyze_p = sub.add_parser("analyze", help="Run data analysis on stored papers/reviews")
    analyze_p.add_argument(
        "--type",
        choices=["venue_stats", "cross_venue", "review_quality", "rebuttal_effect", "all"],
        default="all",
        help="Analysis type to run (default: all)",
    )
    analyze_p.add_argument("--venue", default=None, help="Limit analysis to a specific venue ID (default: all venues)")
    analyze_p.add_argument("--output", default=None, help="Output directory for JSON/CSV files (default: <workspace>/analytics)")
    analyze_p.add_argument(
        "--latest-run",
        action="store_true",
        help="Use the most recent run directory under --runs-dir as workspace (only if workspace not otherwise specified).",
    )

    # -- run ----------------------------------------------------------------
    run_p = sub.add_parser("run", help="Run full pipeline for a single OpenReview paper")
    run_p.add_argument("--url", required=True, help="OpenReview forum URL")
    run_p.add_argument("--model", default=None, help="LLM model override")
    run_p.add_argument("--limit", type=int, default=None, help="Limit reviews")
    run_p.add_argument("--force", action="store_true", help="Force re-run")
    run_p.add_argument("--delay", type=float, default=0.1, help="Delay between LLM calls")
    run_p.add_argument("--top-k", type=int, default=5, help="Evidence chunks per claim")    
    run_p.add_argument("--no-report", action="store_true", help="Do not export report at end") # report ON by default; allow turning it off
    run_p.add_argument("--run-name", default=None, help="Optional suffix for generated run id (e.g., demo, exp1).")

    # -- benchmark-parse -----------------------------------------------------
    bench_p = sub.add_parser(
        "benchmark-parse",
        help="Benchmark PDF parsing pipelines (PyMuPDF, Docling, Marker, LlamaParse)",
    )
    bench_src = bench_p.add_mutually_exclusive_group(required=True)
    bench_src.add_argument("--pdf", default=None, help="Path to a single PDF to benchmark")
    bench_src.add_argument("--pdf-dir", default=None, help="Directory of PDFs to benchmark")
    bench_p.add_argument(
        "--paper-id",
        default=None,
        help="Paper ID to assign (defaults to PDF stem when --pdf is used)",
    )
    bench_p.add_argument(
        "--parsers",
        default=None,
        help=(
            "Comma-separated list of parsers to run "
            "(default: all). Choices: pymupdf,docling,marker,llamaparse"
        ),
    )
    bench_p.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for benchmark artefacts "
            "(default: <workspace>/benchmarks/pdf_parsing)"
        ),
    )

    return p


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _handle_fetch(args: argparse.Namespace) -> None:
    from gsr.config import PDF_DIR
    from gsr.data_collection import build_client, fetch_venue_data, init_db, save_json, save_to_db

    with timed("fetch_total", url=getattr(args, "url", None), venue=getattr(args, "venue", None)):
        print("Authenticating with OpenReview...")
        with timed("fetch_auth"):
            client = build_client()

        if args.url:
            from gsr.link_ingest.openreview.resolve import resolve_openreview_url
            from gsr.link_ingest.openreview.fetch_forum import fetch_forum_data
            from gsr.link_ingest.openreview.storage_adapter import save_forum_bundle

            with timed("fetch_forum_resolve_url", url=args.url):
                forum_id = resolve_openreview_url(args.url)

            log.info("[fetch] START url=%s forum_id=%s", args.url, forum_id)
            with timed("fetch_forum_data", forum_id=forum_id):
                bundle = fetch_forum_data(forum_id)

            conn = init_db()
            try:
                with timed("fetch_save_forum_bundle", paper_id=bundle.get("id")):
                    save_forum_bundle(bundle, conn)
                n_reviews = len(bundle.get("reviews") or [])
                log.info("[fetch] DONE paper_id=%s reviews=%d", bundle.get("id"), n_reviews)
                print(f"Fetched single paper: {bundle['id']}")
            finally:
                conn.close()
            return

        log.info("[fetch] START venue=%s limit=%s pdf=%s", args.venue, args.limit, args.pdf)
        print(
            f"Fetching papers from {args.venue}"
            + (f" (limit {args.limit})" if args.limit else "")
            + "..."
        )

        pdf_dir = args.pdf_dir or PDF_DIR
        with timed("fetch_venue_data", venue=args.venue, limit=args.limit, pdf=args.pdf):
            venue_data = fetch_venue_data(
                client,
                args.venue,
                limit=args.limit,
                download_pdfs=args.pdf,
                pdf_dir=pdf_dir,
            )

        n_papers = len(venue_data["papers"])
        n_reviews = sum(len(p["reviews"]) for p in venue_data["papers"])
        print(f"Fetched {n_papers} papers with {n_reviews} reviews.")

        if not args.db_only:
            with timed("fetch_save_json", papers=n_papers, reviews=n_reviews):
                path = save_json(venue_data)
            print(f"JSON snapshot: {path}")

        if not args.json_only:
            conn = init_db()
            try:
                with timed("fetch_save_to_db", papers=n_papers, reviews=n_reviews):
                    save_to_db(venue_data, conn)
                print("SQLite database updated.")
            finally:
                conn.close()

        log.info("[fetch] DONE venue=%s papers=%d reviews=%d", args.venue, n_papers, n_reviews)
        print("Done.")

def _handle_extract(args: argparse.Namespace) -> None:
    from gsr.claim_extraction import (
        extract_all_claims,
        init_claims_db,
        save_extraction_results,
    )

    from gsr.data_collection.storage import init_db
    import gsr.claim_extraction.extractor as extractor_mod

    with timed(
        "extract_total",
        paper_id=getattr(args, "paper_id", None),
        limit=getattr(args, "limit", None),
        model=getattr(args, "model", None),
    ):
        # Robust defaults: if args doesn't carry these (e.g., called from _handle_run),
        # fall back to extractor module defaults.
        mc = getattr(args, "min_challengeability", None)
        mf = getattr(args, "min_confidence", None)

        if mc is None:
            mc = extractor_mod.MIN_CHALLENGEABILITY
        if mf is None:
            mf = extractor_mod.MIN_CONFIDENCE

        extractor_mod.MIN_CHALLENGEABILITY = float(mc)
        extractor_mod.MIN_CONFIDENCE = float(mf)

        logging.getLogger(__name__).debug(
            "extract args: paper_id=%s fields=%s min_confidence=%s min_challengeability=%s force=%s",
            getattr(args, "paper_id", None),
            getattr(args, "fields", None),
            mf,
            mc,
            getattr(args, "force", None),
        )

        raw_fields = getattr(args, "fields", None) or "summary,strengths,weaknesses,questions"
        fields = tuple(f.strip() for f in raw_fields.split(",") if f.strip())
        if not fields:
            fields = ("summary", "strengths", "weaknesses", "questions")

        conn = init_db()
        try:
            with timed("extract_init_claims_db", paper_id=getattr(args, "paper_id", None)):
                init_claims_db(conn)

            print(
                "Extracting claims"
                + (f" (limit {args.limit} reviews)" if args.limit else "")
                + (f" for paper {args.paper_id}" if args.paper_id else "")
                + f" using model {args.model or 'default'}"
                + f" | thresholds: challengeability>={extractor_mod.MIN_CHALLENGEABILITY},"
                + f" confidence>={extractor_mod.MIN_CONFIDENCE}"
                + "..."
            )

            with timed(
                "extract_all_claims_call",
                paper_id=getattr(args, "paper_id", None),
                fields=",".join(fields),
                score_claims=getattr(args, "score_claims", False),
            ):
                summary = extract_all_claims(
                    conn,
                    limit=args.limit,
                    paper_id=args.paper_id,
                    model=args.model,
                    fields=fields,
                    min_confidence=float(mf),  # IMPORTANT: args.min_* may be None in `run` pipeline; use resolved floats
                    min_challengeability=float(mc),
                    force=bool(getattr(args, "force", False)),
                    delay=float(getattr(args, "delay", 0.1)),
                    score_claims=getattr(args, "score_claims", False),
                    scorer_model=getattr(args, "scorer_model", "meta-llama/Llama-3.1-8B"),
                    max_workers=int(getattr(args, "max_workers", 2)),
                    extract_mode=getattr(args, "extract_mode", "field"),
                )
                

            if summary.get("results"):
                with timed(
                    "save_extraction_results",
                    paper_id=getattr(args, "paper_id", None),
                    result_count=len(summary["results"]),
                ):
                    save_extraction_results(summary["results"], conn)

            print(
                f"Done. Experiment: {summary['experiment_id']},"
                f"Processed {summary['reviews_processed']} reviews, "
                f"extracted {summary['claims_extracted']} claims, "
                f"{summary['errors']} errors."
            )

        finally:
            conn.close()
def is_paper_index_up_to_date(
    conn,
    *,
    paper_id: str,
    embedding_model_id: str | None,
    no_embed: bool,
) -> bool:
    """First-pass check whether a paper index can be reused.

    Current heuristic:
    - paper must already exist in paper_chunks
    - if embeddings are required, paper must also exist in chunk_embeddings for the requested model

    Notes:
    - This is intentionally a simple v1 check.
    - It does NOT yet validate chunk_size / chunk_overlap / layout_aware / pdf hash.
    - Those can be added later via a paper_index_meta table.
    """
    from gsr.paper_retrieval.storage.storage import (
        get_embedded_paper_ids,
        get_indexed_paper_ids,
    )

    indexed_ids = set(get_indexed_paper_ids(conn))
    if paper_id not in indexed_ids:
        return False

    if no_embed:
        return True

    resolved_model_id = embedding_model_id or "allenai/specter2_base"
    embedded_ids = set(get_embedded_paper_ids(conn, resolved_model_id))    

    logging.getLogger(__name__).debug(
        "index check: paper_id=%s model=%s indexed=%s embedded=%s",
        paper_id, resolved_model_id,
        paper_id in indexed_ids, paper_id in embedded_ids,
    )
    return paper_id in embedded_ids

def _handle_retrieve(args: argparse.Namespace) -> None:
    from gsr.paper_retrieval import index_all_papers, init_retrieval_db
    from gsr.paper_retrieval.retrieval import retrieve_all_claims_for_paper
    from gsr.data_collection.storage import init_db

    def _paper_ids_from_chunks(
        conn: sqlite3.Connection,
        *,
        paper_id: str | None,
        limit: int | None,
    ) -> list[str]:
        if paper_id:
            return [paper_id]

        rows = conn.execute(
            """
            SELECT DISTINCT paper_id
            FROM paper_chunks
            ORDER BY paper_id
            """
        ).fetchall()

        ids = [r[0] for r in rows]
        if limit:
            ids = ids[:limit]
        return ids

    def _ensure_evidence_objects_for_paper(
        conn: sqlite3.Connection,
        *,
        paper_id: str,
        embedding_model_id: str | None,
        embed: bool,
        force: bool,
        use_figure_ocr: bool = False,
    ) -> dict[str, int | str | bool]:
        """
        Build and persist evidence_objects + evidence_embeddings for one paper.
        """
        from gsr.paper_retrieval.evidence.evidence_builder import build_evidence_objects_for_paper
        from gsr.paper_retrieval.storage.storage import (
            load_chunks_for_paper,
            load_pdf_spans_for_paper,
            load_evidence_objects_for_paper,
            save_evidence_objects,
            save_evidence_embeddings,
            delete_evidence_objects_for_paper,
        )

        existing_objects = load_evidence_objects_for_paper(paper_id, conn)
        resolved_model_id = embedding_model_id or "allenai/specter2_base"

        if existing_objects and not force:
            if not embed:
                return {
                    "paper_id": paper_id,
                    "objects": len(existing_objects),
                    "embedded": 0,
                    "skipped": True,
                }

            cur = conn.execute(
                """
                SELECT COUNT(*)
                FROM evidence_embeddings ee
                JOIN evidence_objects eo ON ee.evidence_object_id = eo.id
                WHERE eo.paper_id = ? AND ee.model_id = ?
                """,
                (paper_id, resolved_model_id),
            )
            embedded_count = cur.fetchone()[0]
            if embedded_count > 0:
                return {
                    "paper_id": paper_id,
                    "objects": len(existing_objects),
                    "embedded": embedded_count,
                    "skipped": True,
                }

        chunks = load_chunks_for_paper(paper_id, conn)
        spans = load_pdf_spans_for_paper(paper_id, conn)

        if not chunks:
            return {
                "paper_id": paper_id,
                "objects": 0,
                "embedded": 0,
                "skipped": True,
            }

        pdf_path: str | None = None
        if use_figure_ocr:
            row = conn.execute("SELECT pdf_path FROM papers WHERE id = ?", (paper_id,)).fetchone()
            pdf_path = row[0] if row else None

        objects = build_evidence_objects_for_paper(
            paper_id=paper_id,
            chunks=chunks,
            spans=spans,
            pdf_path=pdf_path,
            use_figure_ocr=use_figure_ocr,
        )

        if force:
            delete_evidence_objects_for_paper(
                paper_id,
                conn,
                model_id=resolved_model_id,
            )

        save_evidence_objects(objects, conn)

        # Full-paper bbox refinement is no longer run here.
        # Selective post-retrieval refinement runs inside retrieve_all_claims_for_paper()
        # when --bbox-refine is passed.

        embedded = 0
        if embed and objects:
            from gsr.paper_retrieval.storage.embeddings import load_embedding_model

            model, resolved_model_id = load_embedding_model(resolved_model_id)

            texts = [(getattr(o, "retrieval_text", None) or "") for o in objects]
            vectors = model.encode(texts)
            vectors = [v.tolist() if hasattr(v, "tolist") else v for v in vectors]

            save_evidence_embeddings(
                [{"id": getattr(o, "id")} for o in objects],
                vectors,
                resolved_model_id,
                conn,
            )
            embedded = len(vectors)

        return {
            "paper_id": paper_id,
            "objects": len(objects),
            "embedded": embedded,
            "skipped": False,
        }

    with timed(
        "retrieve_total",
        paper_id=getattr(args, "paper_id", None),
        limit=getattr(args, "limit", None),
        layout_aware=getattr(args, "layout_aware", True),
        no_embed=getattr(args, "no_embed", False),
        index_only=getattr(args, "index_only", False),
        claims_only=getattr(args, "claims_only", False),
        docling=getattr(args, "docling", False),
    ):
        conn = init_db()
        try:
            with timed("retrieve_init_db", paper_id=getattr(args, "paper_id", None)):
                init_retrieval_db(conn)

            layout_aware = getattr(args, "layout_aware", True)
            chunk_size = args.chunk_size
            chunk_overlap = args.chunk_overlap

            if layout_aware and chunk_size == 512 and chunk_overlap == 64:
                chunk_size = 8
                chunk_overlap = 2

            exp_id = getattr(args, "experiment_id", None)
            if exp_id is None and getattr(args, "paper_id", None):
                exp_id = _resolve_latest_experiment_id(conn, args.paper_id)

            index_summary = None
            claim_retrieval_summary = None
            evidence_summary_rows: list[dict] = []

            claims_only = getattr(args, "claims_only", False)
            index_only = getattr(args, "index_only", False)

            if claims_only and index_only:
                raise SystemExit("Cannot use --claims-only and --index-only together.")

            def _build_objects_for_targets() -> None:
                target_paper_ids = _paper_ids_from_chunks(
                    conn,
                    paper_id=args.paper_id,
                    limit=args.limit,
                )

                if not target_paper_ids:
                    print("No indexed papers/chunks available for evidence object building.")
                    return

                print(
                    "Building evidence objects"
                    + (f" for paper {args.paper_id}" if args.paper_id else "")
                    + (" [with embeddings]" if not args.no_embed else " [no embeddings]")
                    + "..."
                )

                for pid in target_paper_ids:
                    summary = _ensure_evidence_objects_for_paper(
                        conn,
                        paper_id=pid,
                        embedding_model_id=args.embedding_model,
                        embed=not args.no_embed,
                        force=args.force,
                        use_figure_ocr=getattr(args, "figure_ocr", False),
                    )
                    evidence_summary_rows.append(summary)

                    status = "skipped" if summary["skipped"] else "built"
                    print(
                        f"  [{status}] paper={pid} "
                        f"objects={summary['objects']} embedded={summary['embedded']}"
                    )

            # ------------------------------------------------------------------
            # Mode A: claims-only -> skip paper indexing entirely
            # ------------------------------------------------------------------
            if claims_only:
                if not args.paper_id:
                    raise SystemExit("--claims-only currently requires --paper-id.")

                _build_objects_for_targets()

                print(
                    "Caching retrieval results for claims only"
                    + (f" for paper {args.paper_id}" if args.paper_id else "")
                    + (f" using extraction run {exp_id}" if exp_id is not None else "")
                    + f" | top_k={getattr(args, 'top_k', 5)}"
                    + (" [BM25-only]" if args.no_embed else "")
                    + "..."
                )

                with timed(
                    "retrieve_all_claims_call",
                    paper_id=getattr(args, "paper_id", None),
                    experiment_id=exp_id,
                    top_k=getattr(args, "top_k", 5),
                ):
                    claim_retrieval_summary = retrieve_all_claims_for_paper(
                        conn,
                        paper_id=args.paper_id,
                        experiment_id=exp_id,
                        top_k=getattr(args, "top_k", 5),
                        embedding_model_id=args.embedding_model,
                        force=args.force,
                        use_bbox_refine=getattr(args, "bbox_refine", False),
                        bbox_detector=getattr(args, "bbox_detector", "groundingdino"),
                    )

            # ------------------------------------------------------------------
            # Mode B / C: do paper indexing unless reuse skip applies
            # ------------------------------------------------------------------
            else:
                should_run_index = True

                if args.paper_id and not args.force and not index_only:
                    up_to_date = is_paper_index_up_to_date(
                        conn,
                        paper_id=args.paper_id,
                        embedding_model_id=args.embedding_model,
                        no_embed=args.no_embed,
                    )
                    logging.getLogger(__name__).debug(
                        "index check: paper_id=%s embedding_model=%s no_embed=%s",
                        args.paper_id,
                        args.embedding_model,
                        args.no_embed,
                    )
                    if up_to_date:
                        should_run_index = False
                        print(
                            "Paper index is up-to-date"
                            + (f" for paper {args.paper_id}" if args.paper_id else "")
                            + "; skipping parse/chunk/embed."
                        )

                if should_run_index:
                    print(
                        "Indexing papers"
                        + (f" (limit {args.limit})" if args.limit else "")
                        + (f" for paper {args.paper_id}" if args.paper_id else "")
                        + (" [layout-aware V2]" if layout_aware else " [legacy V1]")
                        + (" [docling]" if args.docling else "")
                        + (" [bbox-refine:%s]" % getattr(args, "bbox_detector", "groundingdino") if getattr(args, "bbox_refine", False) else "")
                        + (" [BM25-only, no embeddings]" if args.no_embed else "")
                        + f" | chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
                        + "..."
                    )

                    with timed(
                        "index_all_papers_call",
                        paper_id=getattr(args, "paper_id", None),
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        embed=(not args.no_embed),
                    ):
                        index_summary = index_all_papers(
                            conn,
                            limit=args.limit,
                            paper_id=args.paper_id,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            embed=not args.no_embed,
                            model_id=args.embedding_model,
                            force=args.force,
                            layout_aware=layout_aware,
                            use_docling=args.docling,
                            use_figure_ocr=getattr(args, "figure_ocr", False),
                            # bbox refinement is now selective post-retrieval only;
                            # never run full-paper refinement at index time.
                            use_bbox_refine=False,
                            bbox_detector=getattr(args, "bbox_detector", "groundingdino"),
                        )

                # Build evidence objects separately only when index_paper() did not
                # already do it.  index_paper() builds + saves evidence objects when
                # layout_aware=True, including running figure OCR.  Calling
                # _build_objects_for_targets() again in that case causes a second full
                # OCR pass for every figure (very expensive on CPU).
                #
                # Skip conditions (either is sufficient):
                #   - should_run_index=True AND layout_aware=True:
                #       index_paper() already built, saved, and embedded everything.
                #
                # Run conditions (both must hold):
                #   - should_run_index=False (index was skipped / reused): evidence
                #     objects may be absent or stale, so ensure they exist here.
                #   - OR layout_aware=False: index_paper() V1 path has no spans and
                #     never calls build_evidence_objects_for_paper(), so we must.
                if not should_run_index or not layout_aware:
                    _build_objects_for_targets()

                # Mode B: default full retrieve = index + claim retrieval caching
                if not index_only:
                    if not args.paper_id:
                        raise SystemExit(
                            "Claim-level retrieval caching currently requires --paper-id."
                        )

                    print(
                        "Caching retrieval results for claims"
                        + (f" for paper {args.paper_id}" if args.paper_id else "")
                        + (f" using extraction run {exp_id}" if exp_id is not None else "")
                        + f" | top_k={getattr(args, 'top_k', 5)}"
                        + "..."
                    )

                    with timed(
                        "retrieve_all_claims_call",
                        paper_id=getattr(args, "paper_id", None),
                        experiment_id=exp_id,
                        top_k=getattr(args, "top_k", 5),
                    ):
                        claim_retrieval_summary = retrieve_all_claims_for_paper(
                            conn,
                            paper_id=args.paper_id,
                            experiment_id=exp_id,
                            top_k=getattr(args, "top_k", 5),
                            embedding_model_id=args.embedding_model,
                            force=args.force,
                            use_bbox_refine=getattr(args, "bbox_refine", False),
                            bbox_detector=getattr(args, "bbox_detector", "groundingdino"),
                        )

            # ------------------------------------------------------------------
            # Final output
            # ------------------------------------------------------------------
            if index_summary is not None:
                print(
                    f"Done indexing. Indexed {index_summary['papers_processed']} papers, "
                    f"{index_summary['total_chunks']} chunks total, "
                    f"{index_summary['errors']} errors."
                )

            if evidence_summary_rows:
                built = sum(1 for r in evidence_summary_rows if not r["skipped"])
                skipped = sum(1 for r in evidence_summary_rows if r["skipped"])
                total_objects = sum(int(r["objects"]) for r in evidence_summary_rows)
                total_embedded = sum(int(r["embedded"]) for r in evidence_summary_rows)

                print(
                    f"Evidence objects summary: papers={len(evidence_summary_rows)}, "
                    f"built={built}, skipped={skipped}, "
                    f"objects={total_objects}, embedded={total_embedded}"
                )

            if claim_retrieval_summary is not None:
                print(
                    f"Done caching retrieval. Processed {claim_retrieval_summary['claims_processed']} claims, "
                    f"skipped {claim_retrieval_summary['claims_skipped']}, "
                    f"cache hits {claim_retrieval_summary['cache_hits']}, "
                    f"{claim_retrieval_summary['errors']} errors, "
                    f"{claim_retrieval_summary['results_written']} rows written."
                )

            if index_summary is None and claim_retrieval_summary is None and not evidence_summary_rows:
                print("Nothing to do.")

        finally:
            conn.close()
            

def _handle_verify(args: argparse.Namespace) -> None:
    from gsr.claim_verification import (
        init_verification_db,
        save_verification_results,
        verify_all_claims,
    )
    from gsr.data_collection.storage import init_db
    from gsr.paper_retrieval.storage.storage import init_retrieval_db

    with timed(
        "verify_total",
        paper_id=getattr(args, "paper_id", None),
        review_id=getattr(args, "review_id", None),
        limit=getattr(args, "limit", None),
        top_k=getattr(args, "top_k", None),
        model=getattr(args, "model", None),
    ):
        conn = init_db()
        try:
            with timed("verify_init_retrieval_db", paper_id=getattr(args, "paper_id", None)):
                init_retrieval_db(conn)

            with timed("verify_init_verification_db", paper_id=getattr(args, "paper_id", None)):
                init_verification_db(conn)

            print(
                "Verifying claims"
                + (f" (limit {args.limit})" if args.limit else "")
                + (f" for paper {args.paper_id}" if args.paper_id else "")
                + (f" for review {args.review_id}" if args.review_id else "")
                + f" using model {args.model or 'default'}"              
                + "..."
            )
            exp_id = getattr(args, "experiment_id", None)
          
            if exp_id is None and getattr(args, "paper_id", None):
                exp_id = _resolve_latest_experiment_id(conn, args.paper_id)

            with timed(
                "verify_all_claims_call",
                paper_id=getattr(args, "paper_id", None),
                experiment_id=exp_id,
                top_k=getattr(args, "top_k", None),
                max_workers=getattr(args, "max_workers", 2),

            ):
                summary = verify_all_claims(
                    conn,
                    paper_id=args.paper_id,
                    review_id=args.review_id,
                    limit=args.limit,
                    model=args.model,
                    top_k=args.top_k,
                    experiment_id=exp_id,
                    embedding_model_id=args.embedding_model,
                    force=args.force,
                    delay=args.delay,
                    allow_live_retrieval=getattr(args, "allow_live_retrieval", False),
                    require_cached_evidence=getattr(args, "require_cached_evidence", True),
                    selective_figure_ocr=getattr(args, "selective_figure_ocr", False),
                    enable_heuristic_figure_escalation=getattr(args, "figure_ocr_heuristic", False),
                    selective_figure_semantic=getattr(args, "figure_semantic", False),
                    enable_figure_semantic_top=getattr(args, "figure_semantic_top", False),
                )

                if summary["results"]:
                    with timed(
                        "save_verification_results",
                        paper_id=getattr(args, "paper_id", None),
                        result_count=len(summary["results"]),
                    ):
                        save_verification_results(summary["results"], conn)

                verdicts_str = ", ".join(
                    f"{v}: {n}" for v, n in sorted(summary["verdicts"].items())
                )
               
                sel_stats = summary.get("selective_figure_ocr_stats")
                sem_stats = summary.get("figure_semantics_stats")
                print(
                    f"Done. Verified {summary['claims_processed']} claims, "
                    f"{summary['errors']} errors."
                    + (
                        f"\nMissing cached evidence — {summary['missing_evidence']}"
                        if summary.get("missing_evidence")
                        else ""
                    )
                    + (f"\nVerdicts — {verdicts_str}" if verdicts_str else "")
                    + (
                        f"\nSelective figure OCR —"
                        f"\n  escalated={sel_stats['escalated']}"
                        f"\n  cache_hit_pos={sel_stats['cache_hit_positive']}"
                        f"\n  cache_hit_neg={sel_stats['cache_hit_negative']}"
                        f"\n  fresh={sel_stats['fresh_ocr']}"
                        f"\n  ocr_unavailable={sel_stats.get('ocr_unavailable', 0)}"
                        f"\n  skipped_rank={sel_stats.get('skipped_rank_policy', 0)}"
                        f"\n  skipped_lexical={sel_stats.get('skipped_lexical', 0)}"
                        f"\n  verdict_changed={sel_stats['verdict_changed']}"
                        f"\n  confidence_only={sel_stats.get('confidence_improved_only', 0)}"
                        f"\n  no_change={sel_stats.get('no_change', 0)}"
                        f"\n  explicit_ref_escalations={sel_stats.get('explicit_ref_escalations', 0)}"
                        f"\n  heuristic_escalations={sel_stats.get('heuristic_escalations', 0)}"
                        f"\n  explicit_ref_verdict_changed={sel_stats.get('explicit_ref_verdict_changed', 0)}"
                        f"\n  heuristic_verdict_changed={sel_stats.get('heuristic_verdict_changed', 0)}"
                        f"\n  avg_rank={sel_stats.get('avg_escalated_figure_rank') or 'N/A'}"
                        if sel_stats and sel_stats.get("enabled")
                        else ""
                    )
                    + (
                        f"\nFigure semantics —"
                        f"\n  attempted={sem_stats['attempted']}"
                        f"\n  cache_hit={sem_stats['cache_hit']}"
                        f"\n  fresh={sem_stats['fresh_success']}"
                        f"\n  failed={sem_stats['fresh_failed']}"
                        f"\n  explicit_ref={sem_stats['explicit_ref']}"
                        f"\n  top_fallback={sem_stats['top_fallback']}"
                        if sem_stats and sem_stats.get("enabled")
                        else ""
                    )
                )
        finally:
            conn.close()

def _handle_report(args: argparse.Namespace) -> None:
    from gsr.config import DB_PATH
    from gsr.config import REPORT_DIR
    from gsr.reporting import (
        export_verification_report,
        export_extraction_report,
        export_extraction_comparison_report,
    )

    # Extraction report now supports batch scopes (paper-id / venue / paper-ids-file).
    if args.report_type == "extraction":
        # single-paper mode
        if getattr(args, "paper_id", None):
            print(f"Generating extraction report for {args.paper_id}...")
            out = export_extraction_report(
                db_path=DB_PATH,
                paper_id=args.paper_id,
                out_path=getattr(args, "out", None),
            )
            print("Report written to:", out)
            return

        # batch mode
        if getattr(args, "out", None):
            # avoid silent misuse; keep behavior deterministic
            print("WARNING: --out is ignored in batch mode. Use --out-dir to control output directory.")

        
        from pathlib import Path
        from gsr.reporting.utils import safe_filename

        conn = sqlite3.connect(str(DB_PATH))
        try:
            paper_ids = _resolve_report_paper_ids(
                conn,
                paper_id=None,
                venue=getattr(args, "venue", None),
                paper_ids_file=getattr(args, "paper_ids_file", None),
            )
        finally:
            conn.close()

        if getattr(args, "limit", None):
            try:
                lim = int(args.limit)
                if lim > 0:
                    paper_ids = paper_ids[:lim]
            except Exception:
                pass

        if not paper_ids:
            print("No papers found for extraction report scope.")
            return

        out_dir = args.out_dir        
        # Default report path under workspace
        if out_dir is None or out_dir == "":
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            out_dir = REPORT_DIR 
        else:              
            out_dir.mkdir(parents=True, exist_ok=True)
        

        scope_desc = (
            f"venue {args.venue}" if getattr(args, "venue", None)
            else f"paper-ids-file {args.paper_ids_file}" if getattr(args, "paper_ids_file", None)
            else "all papers"
        )
        print(
            f"Generating extraction reports for {scope_desc}"
            + (f" (limit {len(paper_ids)} papers)" if getattr(args, "limit", None) else "")
            + "..."
        )

        total = len(paper_ids)
        for i, pid in enumerate(paper_ids, start=1):
            print(f"[{i}/{total}] paper_id={pid}")
            out_path = None
            if out_dir:
                out_path = out_dir / f"{safe_filename(pid)}_extraction.md"
              
                

            out = export_extraction_report(
                db_path=DB_PATH,
                paper_id=pid,
                out_path=str(out_path) if out_path else None,
            )
            print("  ->", out)

        print("Done.")
        return

    # Non-batch report types remain single-paper.   
    #print(f"Generating report for {args.paper_id}...")

    if args.report_type == "verification":
        out = export_verification_report(
            db_path=DB_PATH,
            paper_id=args.paper_id,
            out_path=args.out,
            limit=args.limit,
        )

    elif args.report_type == "compare":
        out = export_extraction_comparison_report(
            db_path=DB_PATH,
            paper_id=args.paper_id,
            out_path=args.out,
        )    
    elif args.report_type == "annotate-claim":
        out = _handle_annotate_claim(args)
    
    elif args.report_type == "retrieval":
        out = export_retrieval_report(
            db_path=DB_PATH,
            paper_id=args.paper_id,
            out_path=args.out,
            preview_limit=args.preview_limit,
            retrieval_top_k=args.top_k,
        )
        print(out)

    elif args.report_type == "ocr-audit":
        from gsr.reporting import export_ocr_audit
        paper_id = getattr(args, "paper_id", None)
        out_dir = getattr(args, "out_dir", None)
        scope = f"paper {paper_id}" if paper_id else "all papers"
        print(f"Generating figure OCR audit for {scope}...")
        csv_path, md_path = export_ocr_audit(
            db_path=DB_PATH,
            paper_id=paper_id,
            out_dir=out_dir,
        )
        print(f"  CSV: {csv_path}")
        print(f"  Markdown: {md_path}")
        return

    elif args.report_type == "bbox-audit":
        from gsr.reporting import export_bbox_audit
        from collections import Counter as _Counter
        paper_id = args.paper_id
        out_dir = getattr(args, "out_dir", None)
        include_text = getattr(args, "include_text", False)

        csv_path, md_path = export_bbox_audit(
            db_path=DB_PATH,
            paper_id=paper_id,
            out_dir=out_dir,
            include_text=include_text,
        )

        # Compact console summary
        import sqlite3 as _sqlite3
        import json as _json
        from gsr.reporting.bbox_audit import compute_bbox_audit_stats
        _conn = _sqlite3.connect(str(DB_PATH))
        try:
            _stats = compute_bbox_audit_stats(_conn, paper_id, include_text=include_text)
        finally:
            _conn.close()

        print(f"\n--- BBox Audit ---")
        print(f"  paper_id           : {paper_id}")
        print(f"  included_objects   : {_stats['total']}")
        for _ot, _c in sorted(_stats["by_type"].items()):
            print(f"  {_ot}s{' ' * max(0, 12 - len(_ot))}     : {_c['total']} (refined {_c['refined']})")
        print(f"  bbox_sources       : {_stats['source_dist']}")
        print(f"  markdown_report    : {md_path}")
        print(f"  csv_report         : {csv_path}")
        return

    elif args.report_type == "object-audit":
        from gsr.reporting import export_object_audit
        paper_id = args.paper_id
        out_dir = getattr(args, "out_dir", None)

        csv_path, md_path = export_object_audit(
            db_path=DB_PATH,
            paper_id=paper_id,
            out_dir=out_dir,
        )
        print(f"\n--- Object Audit ---")
        print(f"  paper_id        : {paper_id}")
        import csv as _csv
        with open(csv_path, newline="", encoding="utf-8") as _f:
            _rows = list(_csv.DictReader(_f))
        n_fig = sum(1 for r in _rows if r.get("object_type") == "figure")
        n_tbl = sum(1 for r in _rows if r.get("object_type") == "table")
        n_false = sum(1 for r in _rows if r.get("suspicious_false_table_identity") == "1")
        n_mixed = sum(1 for r in _rows if r.get("suspicious_mixed_visual_bbox") == "1")
        n_cap = sum(1 for r in _rows if r.get("likely_caption_only") == "1")
        print(f"  figures         : {n_fig}")
        print(f"  tables          : {n_tbl}")
        print(f"  suspicious_false_table_identity : {n_false}")
        print(f"  suspicious_mixed_visual_bbox    : {n_mixed}")
        print(f"  likely_caption_only             : {n_cap}")
        print(f"  markdown_report : {md_path}")
        print(f"  csv_report      : {csv_path}")
        return

    elif args.report_type == "ocr-smoke":
        from gsr.paper_retrieval.vision.ocr_lighton import ocr_smoke_test
        import json as _json

        pdf_path = getattr(args, "pdf_path", None)
        paper_id = getattr(args, "paper_id", None)
        label = getattr(args, "label", None)
        page = getattr(args, "page", None)
        bbox_str = getattr(args, "bbox", None)
        force_cpu = getattr(args, "force_cpu", False)
        caption = getattr(args, "caption", None)

        resolved_bbox: list[float] | None = None

        if paper_id:
            # Look up figure from evidence_objects by paper_id + label
            conn = sqlite3.connect(str(DB_PATH))
            try:
                if label:
                    row = conn.execute(
                        """
                        SELECT page, bbox_json, metadata_json, pdf_path
                        FROM evidence_objects
                        LEFT JOIN papers ON papers.id = evidence_objects.paper_id
                        WHERE evidence_objects.paper_id = ?
                          AND evidence_objects.object_type = 'figure'
                          AND evidence_objects.label = ?
                        LIMIT 1
                        """,
                        (paper_id, label),
                    ).fetchone()
                else:
                    row = conn.execute(
                        """
                        SELECT page, bbox_json, metadata_json, pdf_path
                        FROM evidence_objects
                        LEFT JOIN papers ON papers.id = evidence_objects.paper_id
                        WHERE evidence_objects.paper_id = ?
                          AND evidence_objects.object_type = 'figure'
                        ORDER BY page ASC
                        LIMIT 1
                        """,
                        (paper_id,),
                    ).fetchone()
            finally:
                conn.close()

            if not row:
                target = f"label='{label}'" if label else "any figure"
                print(f"No figure evidence object found for paper_id='{paper_id}' {target}.")
                return

            ev_page, bbox_json_str, meta_json_str, db_pdf_path = row
            if not pdf_path:
                pdf_path = db_pdf_path
            if not page:
                page = ev_page
            if bbox_json_str:
                try:
                    resolved_bbox = _json.loads(bbox_json_str)
                except Exception:
                    resolved_bbox = None
            if not caption and meta_json_str:
                try:
                    meta = _json.loads(meta_json_str)
                    caption = meta.get("caption_text") or caption
                except Exception:
                    pass

        # Fallback: parse bbox from --bbox arg
        if resolved_bbox is None and bbox_str:
            try:
                resolved_bbox = [float(v) for v in bbox_str.split(",")]
            except ValueError:
                print(f"Invalid --bbox format: '{bbox_str}'. Expected 'x0,y0,x1,y1'.")
                return

        if not pdf_path:
            print("Error: --pdf-path or --paper-id is required.")
            return
        if not page:
            print("Error: --page is required when using --pdf-path directly.")
            return
        if not resolved_bbox:
            print("Error: --bbox is required when using --pdf-path directly.")
            return

        print(f"Running OCR smoke test...")
        print(f"  PDF: {pdf_path}")
        print(f"  Page: {page}  BBox: {resolved_bbox}")
        print(f"  Force CPU: {force_cpu}")
        if label:
            print(f"  Label: {label}")

        result = ocr_smoke_test(
            pdf_path=pdf_path,
            page=page,
            bbox=resolved_bbox,
            caption_text=caption,
            force_cpu=force_cpu,
        )

        print("\n--- Environment ---")
        env = result.get("env", {})
        for k, v in env.items():
            if k != "detail":
                print(f"  {k}: {v}")
        print(f"  detail: {env.get('detail', '')}")

        print(f"\n--- Crop Render ---")
        print(f"  rendered: {result['crop_rendered']}")
        if result.get("image_size"):
            print(f"  image_size: {result['image_size'][0]}x{result['image_size'][1]} px")

        print(f"\n--- Inference ---")
        print(f"  device_selected: {result.get('device_selected')}")
        print(f"  ocr_attempted: {result['ocr_attempted']}")
        print(f"  ocr_status: {result['ocr_status']}")

        if result.get("error_type"):
            print(f"  error_type: {result['error_type']}")
            print(f"  error_message: {result.get('error_message', '')}")

        if result.get("ocr_text_preview"):
            print(f"\n--- OCR Text Preview ---")
            print(f"  {result['ocr_text_preview']}")

        return

    elif args.report_type == "bbox-smoke":
        # Phase P1.6 — generalized detector smoke test (groundingdino or rtdetr)
        import fitz
        from gsr.config import DB_PATH

        _det = getattr(args, "detector", "groundingdino")
        paper_id = getattr(args, "paper_id", None)
        page_1idx = getattr(args, "page", 1) or 1
        neighbor_pages = getattr(args, "neighbor_pages", 0) or 0

        if _det == "rtdetr":
            from gsr.paper_retrieval.vision.rtdetr_detector import (
                rtdetr_available as _available_fn,
                detect_on_page as _detect_fn,
                filter_detections as _filter_fn,
                _MODEL_ID as _mid,
            )
            _raw_label_key = "rtdetr_label"
        elif _det == "paddlex_layout":
            from gsr.paper_retrieval.vision.paddlex_layout_detector import (
                paddlex_layout_available as _available_fn,
                detect_on_page as _detect_fn,
                filter_detections as _filter_fn,
                _MODEL_ID as _mid,
            )
            _raw_label_key = "paddlex_layout_label"
        else:
            from gsr.paper_retrieval.vision.grounding_dino import (
                grounding_dino_available as _available_fn,
                detect_on_page as _detect_fn,
                filter_detections as _filter_fn,
                _MODEL_ID as _mid,
            )
            _raw_label_key = "dino_label"

        # Resolve PDF path
        _pdf_path: str | None = None
        if paper_id:
            _conn_s = sqlite3.connect(str(DB_PATH))
            try:
                _row = _conn_s.execute(
                    "SELECT pdf_path FROM papers WHERE id = ? LIMIT 1", (paper_id,)
                ).fetchone()
                _pdf_path = _row[0] if _row else None
            finally:
                _conn_s.close()

        if not _pdf_path:
            print("Error: could not resolve PDF path. Provide --paper-id with a known paper.")
            return

        try:
            _fdoc = fitz.open(_pdf_path)
            _num_pages = len(_fdoc)
            _fdoc.close()
        except Exception as exc:
            print(f"Error: could not open PDF: {exc}")
            return

        print(f"\n--- Detector Smoke Test ---")
        print(f"  detector     : {_det}")
        print(f"  model_id     : {_mid}")
        print(f"  paper_id     : {paper_id}")
        print(f"  pdf_path     : {_pdf_path}")
        print(f"  total_pages  : {_num_pages}")

        _avail = _available_fn()
        print(f"  available    : {_avail}")
        if not _avail:
            print(f"  Detector '{_det}' unavailable (missing dependencies). Stopping.")
            return

        pages_to_test = sorted(set(range(
            max(1, page_1idx - neighbor_pages),
            min(_num_pages, page_1idx + neighbor_pages) + 1,
        )))

        _any_error = False
        for _pg1 in pages_to_test:
            _pg0 = _pg1 - 1
            print(f"\n  --- page {_pg1} ---")
            try:
                _pdoc2 = fitz.open(_pdf_path)
                _pg = _pdoc2[_pg0]
                _pg_area = _pg.rect.width * _pg.rect.height
                _pdoc2.close()
            except Exception:
                _pg_area = 0.0

            try:
                _raw = _detect_fn(_pdf_path, _pg0)
            except Exception as exc:
                print(f"  detection failed: {exc}")
                _any_error = True
                continue

            _usable, _rejected = _filter_fn(_raw, _pg_area)
            print(f"  raw_detections    : {len(_raw)}")
            print(f"  usable_detections : {len(_usable)}")
            if _rejected:
                from collections import Counter
                _rc = Counter(r.get("reject_reason", "unknown") for r in _rejected)
                print(f"  rejected          : {dict(_rc)}")
            if _det == "paddlex_layout" and _raw:
                from collections import Counter as _Counter
                _lc = _Counter(d.get("label", "?") for d in _raw)
                print(f"  by_norm_label     : {dict(_lc)}")

            _all = (
                [{**d, "_usable": True, "_reject_reason": None} for d in _usable]
                + [{**r, "_usable": False, "_reject_reason": r.get("reject_reason")} for r in _rejected]
            )
            _all.sort(key=lambda d: d.get("score", 0.0), reverse=True)
            for _i, _d in enumerate(_all[:10]):
                _bbox = _d.get("bbox", [])
                _area = max(0.0, _bbox[2] - _bbox[0]) * max(0.0, _bbox[3] - _bbox[1]) if len(_bbox) == 4 else 0.0
                _cov = (_area / _pg_area) if _pg_area > 0 else 0.0
                _status = "OK" if _d["_usable"] else f"SKIP:{_d['_reject_reason']}"
                print(
                    f"    [{_i}] {_status:35s}  "
                    f"type={_d.get('label','?'):6s}  "
                    f"raw={_d.get(_raw_label_key, _d.get('label','?')):20s}  "
                    f"score={_d.get('score', 0):.3f}  "
                    f"cov={_cov:.2f}  "
                    f"bbox={[round(v, 1) for v in _bbox]}"
                )

        if not _any_error:
            print("\n  [OK] Smoke test completed successfully.")
        return

    elif args.report_type == "dino-smoke":
        import inspect
        import fitz  # PyMuPDF — for page dimensions
        from gsr.config import DB_PATH
        from gsr.paper_retrieval.vision.grounding_dino import (
            grounding_dino_available,
            _ensure_model,
            detect_on_page,
            filter_detections,
            _MODEL_ID,
        )

        paper_id = getattr(args, "paper_id", None)
        page_1idx = getattr(args, "page", 1) or 1
        neighbor_pages = getattr(args, "neighbor_pages", 0) or 0

        # Resolve PDF path
        dino_pdf_path: str | None = None
        dino_num_pages: int = 0
        if paper_id:
            conn_d = sqlite3.connect(str(DB_PATH))
            try:
                row = conn_d.execute(
                    "SELECT pdf_path FROM papers WHERE id = ? LIMIT 1", (paper_id,)
                ).fetchone()
                dino_pdf_path = row[0] if row else None
            finally:
                conn_d.close()

        if not dino_pdf_path:
            print("Error: could not resolve PDF path. Provide --paper-id with a known paper.")
            return

        try:
            _doc = fitz.open(dino_pdf_path)
            dino_num_pages = len(_doc)
            _doc.close()
        except Exception as exc:
            print(f"Error: could not open PDF: {exc}")
            return

        print(f"\n--- GroundingDINO Smoke Test ---")
        print(f"  paper_id     : {paper_id}")
        print(f"  pdf_path     : {dino_pdf_path}")
        print(f"  total_pages  : {dino_num_pages}")

        available = grounding_dino_available()
        print(f"  model_available : {available}")
        if not available:
            print("  GroundingDINO unavailable (missing torch/transformers). Stopping.")
            return

        try:
            processor, _ = _ensure_model(_MODEL_ID)
            method = getattr(processor, "post_process_grounded_object_detection", None)
            postprocess_params = list(inspect.signature(method).parameters.keys()) if method else []
            print(f"  processor_class    : {type(processor).__name__}")
            print(f"  postprocess_params : {postprocess_params}")
        except Exception as exc:
            print(f"  model load failed: {exc}")
            return

        # Build the list of 1-indexed pages to test
        center = page_1idx
        pages_to_test = sorted(set(range(
            max(1, center - neighbor_pages),
            min(dino_num_pages, center + neighbor_pages) + 1,
        )))

        any_error = False
        for pg1 in pages_to_test:
            pg0 = pg1 - 1
            print(f"\n  --- page {pg1} ---")
            try:
                _pdoc = fitz.open(dino_pdf_path)
                _page = _pdoc[pg0]
                pg_w = _page.rect.width
                pg_h = _page.rect.height
                pg_area = pg_w * pg_h
                _pdoc.close()
            except Exception:
                pg_w = pg_h = pg_area = 0.0

            try:
                raw_dets = detect_on_page(dino_pdf_path, pg0)
            except Exception as exc:
                print(f"  detection failed: {exc}")
                any_error = True
                continue

            usable_dets, rejected_dets = filter_detections(raw_dets, pg_area)
            print(f"  raw_detections    : {len(raw_dets)}")
            print(f"  usable_detections : {len(usable_dets)}")
            if rejected_dets:
                from collections import Counter
                rc = Counter(r.get("reject_reason", "unknown") for r in rejected_dets)
                print(f"  rejected          : {dict(rc)}")

            all_dets = [
                {**d, "_usable": True, "_reject_reason": None} for d in usable_dets
            ] + [
                {**r, "_usable": False, "_reject_reason": r.get("reject_reason")} for r in rejected_dets
            ]
            # Sort by score descending for readability
            all_dets.sort(key=lambda d: d.get("score", 0.0), reverse=True)

            for i, d in enumerate(all_dets[:10]):
                bbox = d.get("bbox", [])
                area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]) if len(bbox) == 4 else 0.0
                coverage = (area / pg_area) if pg_area > 0 else 0.0
                status = "OK" if d["_usable"] else f"SKIP:{d['_reject_reason']}"
                print(
                    f"    [{i}] {status:35s}  "
                    f"type={d.get('label','?'):6s}  "
                    f"raw={d.get('dino_label','?'):12s}  "
                    f"score={d.get('score', 0):.3f}  "
                    f"cov={coverage:.2f}  "
                    f"bbox={[round(v, 1) for v in bbox]}"
                )

        if not any_error:
            print("\n  [OK] Smoke test completed successfully.")
        return

    else:
        raise ValueError(f"Unknown report type: {args.report_type}")

    print("Report written to:", out)


def _handle_analyze(args: argparse.Namespace) -> None:
    from gsr.config import WORKSPACE_DIR
    from gsr.data_analysis import (
        compute_cross_venue,
        compute_rebuttal_effect,
        compute_review_quality,
        compute_venue_stats,
        export_csv,
        export_json,
    )
    from gsr.data_collection.storage import init_db

    output_dir = Path(args.output) if args.output else (WORKSPACE_DIR / "analytics")

    conn = init_db()
    try:
        run_all = args.type == "all"

        if run_all or args.type == "venue_stats":
            print("Running venue statistics...")
            result = compute_venue_stats(conn, venue_id=args.venue)
            export_json(result, "venue_stats.json", output_dir)
            print("  Exported venue_stats.json")

        if run_all or args.type == "cross_venue":
            print("Running cross-venue comparison...")
            result = compute_cross_venue(conn)
            export_json(result, "cross_venue.json", output_dir)
            export_csv(result["comparison"], "cross_venue.csv", output_dir)
            print("  Exported cross_venue.json, cross_venue.csv")

        if run_all or args.type == "review_quality":
            print("Running review quality analysis...")
            result = compute_review_quality(conn, venue_id=args.venue)
            export_json(result, "review_quality.json", output_dir)
            print("  Exported review_quality.json")

        if run_all or args.type == "rebuttal_effect":
            print("Running rebuttal effect analysis...")
            result = compute_rebuttal_effect(conn, venue_id=args.venue)
            export_json(result, "rebuttal_effect.json", output_dir)
            print("  Exported rebuttal_effect.json")

        print("Done.")
    finally:
        conn.close()

def _handle_annotate_claim(args: argparse.Namespace) -> None:
    from gsr.data_collection.storage import init_db
    from gsr.reporting.annotate_pdf import annotate_pdf_by_claim

    conn = init_db()
    try:
        out = annotate_pdf_by_claim(
            conn,
            claim_id=args.claim_id,
            output_dir=args.output_dir,
        )
        return out
        #print(f"Annotated PDF written to: {out}")
    finally:
        conn.close()


def _handle_run(args: argparse.Namespace) -> None:
    print("=== Running full pipeline (single paper) ===")

    with timed("run_pipeline_total", url=args.url):
        with timed("run_fetch", url=args.url):
            _handle_fetch(
                argparse.Namespace(
                    url=args.url,
                    venue=None,
                    limit=None,
                    json_only=False,
                    db_only=False,
                    pdf=True,
                    pdf_dir=None,
                    run_name=getattr(args, "run_name", None),
                )
            )

        from gsr.link_ingest.openreview.resolve import resolve_openreview_url
        forum_id = resolve_openreview_url(args.url)
        paper_id = f"openreview::{forum_id}"

        with timed("run_extract", paper_id=paper_id):
            _handle_extract(
                argparse.Namespace(
                    limit=args.limit,
                    paper_id=paper_id,
                    model=args.model,
                    fields="weaknesses,strengths,questions,summary",
                    force=args.force,
                    delay=args.delay,
                    min_challengeability=getattr(args, "min_challengeability", None),
                    min_confidence=getattr(args, "min_confidence", None),
                    score_claims=False,
                    scorer_model="meta-llama/Llama-3.1-8B",
                )
            )

        with timed("run_retrieve", paper_id=paper_id):
            _handle_retrieve(
                argparse.Namespace(
                    paper_id=paper_id,
                    limit=None,
                    chunk_size=8,
                    chunk_overlap=2,
                    no_embed=False,
                    embedding_model=None,
                    force=args.force,
                    layout_aware=True,
                    docling=args.docling,
                    figure_ocr=getattr(args, "figure_ocr", False),
                )
            )

        with timed("run_verify", paper_id=paper_id, top_k=args.top_k):
            _handle_verify(
                argparse.Namespace(
                    paper_id=paper_id,
                    review_id=None,
                    limit=None,
                    model=args.model,
                    top_k=args.top_k,
                    embedding_model=None,
                    force=args.force,
                    delay=args.delay,
                    all_claims=False,
                    experiment_id=None,
                )
            )

        from gsr.config import DB_PATH
        from gsr.reporting.exporter import export_verification_report

        if not getattr(args, "no_report", False):
            with timed("run_report", paper_id=paper_id):
                out = export_verification_report(
                    db_path=DB_PATH,
                    paper_id=paper_id,
                    out_path=getattr(args, "out", None),
                    limit=getattr(args, "limit", None),
                )
                print("Report written to:", out)

    print("=== Pipeline complete ===")


# ---------------------------------------------------------------------------
# benchmark-parse handler
# ---------------------------------------------------------------------------

def _handle_benchmark_parse(args: argparse.Namespace) -> None:
    from gsr.config import WORKSPACE_DIR
    from gsr.paper_retrieval.parser_benchmark.runner import run_benchmark

    output_dir = getattr(args, "output_dir", None)
    if output_dir is None:
        output_dir = WORKSPACE_DIR / "benchmarks" / "pdf_parsing"

    parsers_raw = getattr(args, "parsers", None)
    parsers: list[str] | None = None
    if parsers_raw:
        parsers = [p.strip() for p in parsers_raw.split(",") if p.strip()]

    pdf_path = getattr(args, "pdf", None)
    pdf_dir = getattr(args, "pdf_dir", None)
    paper_id = getattr(args, "paper_id", None)

    if pdf_path and paper_id is None:
        paper_id = Path(pdf_path).stem

    print(f"Benchmark output directory: {output_dir}")

    report_paths = run_benchmark(
        pdf_path=pdf_path,
        pdf_dir=pdf_dir,
        output_dir=str(output_dir),
        parsers=parsers,
        verbose=args.verbose,
    )

    print("\nBenchmark complete.")
    for label, path in report_paths.items():
        print(f"  {label}: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    print("DEBUG parsed args:", vars(args))



    repo_root = Path(__file__).resolve().parents[2]

    # 1) Load .env
    _load_env()

    # 2) Decide workspace (one unified policy for all commands)
    ws_dir, ws_src = _pick_workspace_dir(args, repo_root)

    if ws_dir is not None:
        os.environ["GSR_WORKSPACE"] = str(ws_dir)
        os.environ["GSR_WORKSPACE_SOURCE"] = ws_src or "cli"
    else:
        # If env already provided (from .env or user), keep it; otherwise mark default.
        if os.environ.get("GSR_WORKSPACE"):
            os.environ.setdefault("GSR_WORKSPACE_SOURCE", "env")
        else:
            os.environ.setdefault("GSR_WORKSPACE_SOURCE", "default")

    # 3) Reload config AFTER env override so DB_PATH/WORKSPACE_DIR reflect the chosen workspace
    import importlib
    import gsr.config as _config
    importlib.reload(_config)

    # 3.1) Configure logging AFTER workspace is resolved, so logs go into the active workspace
    from gsr.config import WORKSPACE_DIR

    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_dir = WORKSPACE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "gsr.log"

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,  # reconfigure in case logging was already initialized
    )

    # Suppress verbose INFO noise from third-party libraries.
    # These loggers default to INFO and produce many irrelevant lines
    # (model-loading status, HTTP traces, dataset caching, etc.).
    for _noisy_logger in (
        "litellm",
        "sentence_transformers",
        "transformers",
        "datasets",
        "httpx",
        "urllib3",
    ):
        logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

    print(f"Log file: {log_file}")

    # 4) Ensure workspace dirs exist (safe)
    try:
        from gsr.config import ensure_workspace_dirs
        ensure_workspace_dirs()
    except Exception:
        pass

    if args.command:
        _print_workspace_banner()
    
    # Guard rail: bulk venue fetch must have an explicit destination workspace
    if args.command == "fetch" and getattr(args, "venue", None):
        if not getattr(args, "workspace", None) and not getattr(args, "run_id", None) and not getattr(args, "latest_run", False) and not os.environ.get("GSR_WORKSPACE"):
            raise SystemExit(
                "For `fetch --venue`, please specify a destination with "
                "`--workspace <dir>` (recommended for long-lived datasets) or "
                "`--run-id <id>` / `--latest-run` (for experiment snapshots)."
            )

    # analyze: optionally select latest run if workspace not specified anywhere
    if (
        args.command == "analyze"
        and getattr(args, "latest_run", False)
        and "GSR_WORKSPACE" not in os.environ
        and not getattr(args, "workspace", None)
    ):
        runs_base = Path(args.runs_dir)
        runs_base = runs_base if runs_base.is_absolute() else (repo_root / runs_base)
        if runs_base.exists():
            candidates = [p for p in runs_base.iterdir() if p.is_dir()]
            if candidates:
                latest = max(candidates, key=lambda p: p.stat().st_mtime)
                os.environ["GSR_WORKSPACE"] = str(latest.resolve())
                os.environ["GSR_WORKSPACE_SOURCE"] = "cli-latest-run"

    # Ensure workspace dirs exist
    try:
        from gsr.config import ensure_workspace_dirs
        ensure_workspace_dirs()
    except Exception:
        pass

    if args.command:
        _print_workspace_banner()

    if args.command == "fetch":
        _handle_fetch(args)
    elif args.command == "extract":
        _handle_extract(args)
    elif args.command == "retrieve":
        _handle_retrieve(args)
    elif args.command == "verify":
        _handle_verify(args)
    elif args.command == "report":
        _handle_report(args)
    elif args.command == "analyze":    
        _handle_analyze(args)
    elif args.command == "run":
        _handle_run(args)
    elif args.command == "benchmark-parse":
        _handle_benchmark_parse(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
