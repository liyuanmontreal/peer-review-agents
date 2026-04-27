"""Tests for Phase 5A-prep: GSR integration health and gsr_runner bridge.

Covers:
1. GSR module import smoke tests
2. check_gsr_available()
3. get_gsr_workspace() — explicit arg and env var
4. ensure_paper_indexed_for_koala() — db missing, not indexed, already indexed
5. get_paper_summary_sections() — graceful failure paths
6. get_seed_evidence_candidates_from_gsr() — graceful failure paths
7. Tier 1 API backward compat (PaperIndex, index_paper_for_koala, get_seed_evidence_candidates)
8. gsr_agent modules do not import gsr.* at module scope
"""

from __future__ import annotations

import importlib
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from gsr_agent.adapters.gsr_runner import (
    GSRAdapterResult,
    GSRAvailability,
    PaperIndex,
    PaperSummarySections,
    SeedEvidenceCandidate,
    check_gsr_available,
    ensure_paper_indexed_for_koala,
    get_gsr_workspace,
    get_paper_summary_sections,
    get_seed_evidence_candidates,
    get_seed_evidence_candidates_from_gsr,
    index_paper_for_koala,
)
from gsr_agent.koala.models import Paper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_paper(paper_id: str = "p-gsr-001") -> Paper:
    now = datetime.now(timezone.utc)
    return Paper(
        paper_id=paper_id,
        title="GSR Test Paper",
        open_time=now,
        review_end_time=now + timedelta(hours=48),
        verdict_end_time=now + timedelta(hours=72),
        state="REVIEW_ACTIVE",
        abstract="This paper proposes a novel approach to testing GSR integration.",
        full_text="",
        domains=["ML"],
    )


def _init_gsr_db(db_path: Path, paper_id: str, *, with_chunks: bool = True) -> None:
    """Create a minimal GSR DB with the tables gsr_runner queries."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS paper_chunks (
            id TEXT PRIMARY KEY, paper_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL, section TEXT NOT NULL,
            page INTEGER NOT NULL, text TEXT NOT NULL,
            char_start INTEGER NOT NULL, char_end INTEGER NOT NULL,
            chunk_size INTEGER NOT NULL, chunked_at TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS evidence_objects (
            id TEXT PRIMARY KEY, paper_id TEXT NOT NULL,
            object_type TEXT NOT NULL, label TEXT, page INTEGER,
            section TEXT, retrieval_text TEXT NOT NULL,
            caption_text TEXT, content_text TEXT
        )"""
    )
    if with_chunks:
        conn.execute(
            "INSERT INTO paper_chunks VALUES (?,?,?,?,?,?,?,?,?,?)",
            (f"{paper_id}_c0", paper_id, 0, "Introduction", 1,
             "This is the introduction text.", 0, 30, 512, "2026-04-25T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO paper_chunks VALUES (?,?,?,?,?,?,?,?,?,?)",
            (f"{paper_id}_c1", paper_id, 1, "Methods", 2,
             "The proposed method uses a neural network.", 0, 42, 512, "2026-04-25T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO evidence_objects VALUES (?,?,?,?,?,?,?,?,?)",
            (f"{paper_id}_eo0", paper_id, "text_chunk", None, 1,
             "Introduction", "Introduction text chunk retrieval text.", None, None),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# 1. GSR module import smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module_name", [
    "gsr",
    "gsr.config",
    "gsr.paper_retrieval",
    "gsr.claim_extraction",
    "gsr.claim_verification",
    "gsr.data_collection",
])
def test_gsr_module_imports(module_name):
    """Each required GSR module must import without error."""
    mod = importlib.import_module(module_name)
    assert mod is not None


# ---------------------------------------------------------------------------
# 2. check_gsr_available()
# ---------------------------------------------------------------------------

def test_check_gsr_available_returns_ok():
    result = check_gsr_available()
    assert isinstance(result, GSRAvailability)
    assert result.ok is True
    assert result.missing_modules == []
    assert result.message


def test_check_gsr_available_structure():
    result = check_gsr_available()
    assert hasattr(result, "ok")
    assert hasattr(result, "missing_modules")
    assert hasattr(result, "message")


# ---------------------------------------------------------------------------
# 3. get_gsr_workspace()
# ---------------------------------------------------------------------------

def test_get_gsr_workspace_explicit_arg(tmp_path):
    result = get_gsr_workspace(default=tmp_path)
    assert result == tmp_path.resolve()


def test_get_gsr_workspace_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("GSR_WORKSPACE", str(tmp_path))
    result = get_gsr_workspace()
    assert result == tmp_path.resolve()


def test_get_gsr_workspace_explicit_overrides_env(monkeypatch, tmp_path):
    env_path = tmp_path / "from_env"
    env_path.mkdir()
    explicit_path = tmp_path / "explicit"
    explicit_path.mkdir()
    monkeypatch.setenv("GSR_WORKSPACE", str(env_path))
    result = get_gsr_workspace(default=explicit_path)
    assert result == explicit_path.resolve()


def test_get_gsr_workspace_default_when_no_env(monkeypatch):
    monkeypatch.delenv("GSR_WORKSPACE", raising=False)
    result = get_gsr_workspace()
    assert isinstance(result, Path)
    assert "workspace" in str(result)


# ---------------------------------------------------------------------------
# 4. ensure_paper_indexed_for_koala()
# ---------------------------------------------------------------------------

def test_ensure_paper_indexed_db_missing(tmp_path):
    paper = _make_paper()
    result = ensure_paper_indexed_for_koala(paper, workspace=tmp_path)
    assert isinstance(result, GSRAdapterResult)
    assert result.ok is False
    assert result.details["status"] == "db_missing"
    assert "gsr.db" in result.message


def test_ensure_paper_indexed_not_indexed(tmp_path):
    paper = _make_paper("p-new")
    _init_gsr_db(tmp_path / "gsr.db", "p-other", with_chunks=True)
    result = ensure_paper_indexed_for_koala(paper, workspace=tmp_path)
    assert result.ok is False
    assert result.details["status"] == "not_indexed"
    assert result.details["chunk_count"] == 0


def test_ensure_paper_indexed_already_indexed(tmp_path):
    paper = _make_paper("p-indexed")
    _init_gsr_db(tmp_path / "gsr.db", "p-indexed", with_chunks=True)
    result = ensure_paper_indexed_for_koala(paper, workspace=tmp_path)
    assert result.ok is True
    assert result.details["status"] == "already_indexed"
    assert result.details["chunk_count"] == 2


def test_ensure_paper_indexed_has_paper_id(tmp_path):
    paper = _make_paper("p-check")
    result = ensure_paper_indexed_for_koala(paper, workspace=tmp_path)
    assert result.paper_id == "p-check"
    assert result.workspace == tmp_path.resolve()


def test_ensure_paper_indexed_empty_db(tmp_path):
    paper = _make_paper()
    _init_gsr_db(tmp_path / "gsr.db", "other", with_chunks=False)
    result = ensure_paper_indexed_for_koala(paper, workspace=tmp_path)
    assert result.ok is False
    assert result.details["status"] == "not_indexed"


# ---------------------------------------------------------------------------
# 5. get_paper_summary_sections()
# ---------------------------------------------------------------------------

def test_get_paper_summary_sections_db_missing(tmp_path):
    result = get_paper_summary_sections("p-001", workspace=tmp_path)
    assert isinstance(result, PaperSummarySections)
    assert result.ok is False
    assert result.sections == {}
    assert "gsr.db" in result.message


def test_get_paper_summary_sections_paper_not_indexed(tmp_path):
    _init_gsr_db(tmp_path / "gsr.db", "p-other", with_chunks=True)
    result = get_paper_summary_sections("p-unknown", workspace=tmp_path)
    assert result.ok is False
    assert result.sections == {}


def test_get_paper_summary_sections_returns_sections(tmp_path):
    _init_gsr_db(tmp_path / "gsr.db", "p-indexed", with_chunks=True)
    result = get_paper_summary_sections("p-indexed", workspace=tmp_path)
    assert result.ok is True
    assert "Introduction" in result.sections
    assert "Methods" in result.sections
    assert result.paper_id == "p-indexed"
    assert result.workspace == tmp_path.resolve()


def test_get_paper_summary_sections_merges_same_section(tmp_path):
    """Multiple chunks in the same section are merged."""
    db_path = tmp_path / "gsr.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS paper_chunks (
            id TEXT PRIMARY KEY, paper_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL, section TEXT NOT NULL,
            page INTEGER NOT NULL, text TEXT NOT NULL,
            char_start INTEGER NOT NULL, char_end INTEGER NOT NULL,
            chunk_size INTEGER NOT NULL, chunked_at TEXT NOT NULL
        )"""
    )
    conn.execute(
        "INSERT INTO paper_chunks VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("c0", "p-merge", 0, "Results", 3, "First result.", 0, 13, 512, "2026-04-25T00:00:00Z"),
    )
    conn.execute(
        "INSERT INTO paper_chunks VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("c1", "p-merge", 1, "Results", 3, "Second result.", 0, 14, 512, "2026-04-25T00:00:00Z"),
    )
    conn.commit()
    conn.close()

    result = get_paper_summary_sections("p-merge", workspace=tmp_path)
    assert result.ok is True
    combined = result.sections["Results"]
    assert "First result." in combined
    assert "Second result." in combined


# ---------------------------------------------------------------------------
# 6. get_seed_evidence_candidates_from_gsr()
# ---------------------------------------------------------------------------

def test_get_seed_evidence_candidates_from_gsr_db_missing(tmp_path):
    result = get_seed_evidence_candidates_from_gsr("p-001", workspace=tmp_path)
    assert result == []


def test_get_seed_evidence_candidates_from_gsr_paper_not_indexed(tmp_path):
    _init_gsr_db(tmp_path / "gsr.db", "p-other", with_chunks=True)
    result = get_seed_evidence_candidates_from_gsr("p-unknown", workspace=tmp_path)
    assert result == []


def test_get_seed_evidence_candidates_from_gsr_returns_candidates(tmp_path):
    _init_gsr_db(tmp_path / "gsr.db", "p-indexed", with_chunks=True)
    result = get_seed_evidence_candidates_from_gsr("p-indexed", workspace=tmp_path)
    assert isinstance(result, list)
    assert len(result) == 1
    cand = result[0]
    assert isinstance(cand, SeedEvidenceCandidate)
    assert cand.claim
    assert cand.location
    assert 0.0 <= cand.confidence <= 1.0


def test_get_seed_evidence_candidates_from_gsr_respects_max(tmp_path):
    db_path = tmp_path / "gsr.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS evidence_objects (
            id TEXT PRIMARY KEY, paper_id TEXT NOT NULL,
            object_type TEXT NOT NULL, label TEXT, page INTEGER,
            section TEXT, retrieval_text TEXT NOT NULL,
            caption_text TEXT, content_text TEXT
        )"""
    )
    for i in range(10):
        conn.execute(
            "INSERT INTO evidence_objects VALUES (?,?,?,?,?,?,?,?,?)",
            (f"eo-{i}", "p-many", "text_chunk", None, i, f"Sec{i}",
             f"Evidence text {i}.", None, None),
        )
    conn.commit()
    conn.close()

    result = get_seed_evidence_candidates_from_gsr("p-many", workspace=tmp_path, max_candidates=3)
    assert len(result) <= 3


# ---------------------------------------------------------------------------
# 7. Tier 1 backward compat
# ---------------------------------------------------------------------------

def test_index_paper_for_koala_builds_index():
    paper = _make_paper()
    idx = index_paper_for_koala(paper)
    assert isinstance(idx, PaperIndex)
    assert idx.paper_id == paper.paper_id
    assert idx.abstract == paper.abstract
    assert idx.domains == list(paper.domains)


def test_get_seed_evidence_candidates_from_index_returns_abstract():
    paper = _make_paper()
    idx = index_paper_for_koala(paper)
    candidates = get_seed_evidence_candidates(idx)
    assert len(candidates) == 1
    assert candidates[0].location == "abstract"
    assert candidates[0].claim == paper.abstract


def test_get_seed_evidence_candidates_from_index_empty_when_no_abstract():
    paper = _make_paper()
    paper.abstract = ""
    idx = index_paper_for_koala(paper)
    assert get_seed_evidence_candidates(idx) == []


# ---------------------------------------------------------------------------
# 8. Module isolation — gsr_agent modules must not import gsr.* at module scope
# ---------------------------------------------------------------------------

def test_gsr_agent_adapters_does_not_import_gsr_at_module_scope():
    """gsr.* must not appear as a top-level import in gsr_runner.py.

    Imports inside functions are allowed; module-level imports are not,
    because they would make all of gsr_agent depend on GSR's heavy deps.
    """
    runner_path = (
        Path(__file__).resolve().parents[1]
        / "src" / "gsr_agent" / "adapters" / "gsr_runner.py"
    )
    source = runner_path.read_text(encoding="utf-8")
    lines = source.splitlines()
    for lineno, line in enumerate(lines, 1):
        # Module-level imports have no leading whitespace.
        # Function-scoped imports (indented) are allowed by design.
        if line != line.lstrip():
            continue
        stripped = line.strip()
        if stripped.startswith("import gsr") or stripped.startswith("from gsr"):
            pytest.fail(
                f"gsr_runner.py:{lineno}: top-level gsr import found: {line!r}. "
                "All gsr.* imports must be inside functions."
            )
