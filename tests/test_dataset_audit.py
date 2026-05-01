"""Tests for scripts/dataset_audit.py."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

# Import directly from the script — it lives outside `src/` so we add the scripts dir to sys.path.
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from dataset_audit import (
    _normalize_verdict,
    _section_table_inventory,
    _section_paper_coverage,
    _section_comment_coverage,
    _section_claim_coverage,
    _section_verification_distribution,
    _section_action_telemetry,
    _section_eligibility_telemetry,
    _section_readiness,
    _render_markdown,
    run_audit,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _empty_koala_db(tmp_path: Path) -> tuple[str, sqlite3.Connection]:
    """Create an empty DB with the full koala_agent schema, return path + conn."""
    db_path = str(tmp_path / "koala_agent.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE koala_papers (
            paper_id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT '',
            open_time TEXT NOT NULL,
            review_end_time TEXT NOT NULL,
            verdict_end_time TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'NEW',
            pdf_url TEXT NOT NULL DEFAULT '',
            local_pdf_path TEXT,
            last_synced_at TEXT NOT NULL
        );
        CREATE TABLE koala_comments (
            comment_id TEXT PRIMARY KEY,
            paper_id TEXT NOT NULL,
            thread_id TEXT,
            parent_id TEXT,
            author_agent_id TEXT NOT NULL DEFAULT '',
            text TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            is_ours INTEGER NOT NULL DEFAULT 0,
            is_citable INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE koala_agent_actions (
            action_id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT NOT NULL,
            action_type TEXT NOT NULL,
            external_id TEXT,
            github_file_url TEXT,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            error_message TEXT,
            details TEXT
        );
        CREATE TABLE koala_karma_ledger (
            ledger_id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT NOT NULL,
            action_type TEXT NOT NULL,
            cost REAL NOT NULL,
            karma_before REAL NOT NULL,
            karma_after REAL NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE koala_extracted_claims (
            claim_id TEXT PRIMARY KEY,
            comment_id TEXT NOT NULL,
            paper_id TEXT NOT NULL,
            claim_text TEXT NOT NULL,
            category TEXT,
            confidence REAL,
            challengeability REAL,
            binary_question TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE koala_claim_verifications (
            verification_id TEXT PRIMARY KEY,
            claim_id TEXT NOT NULL,
            comment_id TEXT NOT NULL,
            paper_id TEXT NOT NULL,
            verdict TEXT NOT NULL,
            confidence REAL,
            reasoning TEXT,
            supporting_quote TEXT,
            model_id TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE koala_reactive_drafts (
            draft_id TEXT PRIMARY KEY,
            comment_id TEXT NOT NULL,
            paper_id TEXT NOT NULL,
            recommendation TEXT NOT NULL,
            draft_text TEXT,
            analysis_json TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE koala_verdict_state (
            paper_id TEXT PRIMARY KEY,
            has_our_participation INTEGER NOT NULL DEFAULT 0,
            distinct_citable_other_agents INTEGER NOT NULL DEFAULT 0,
            eligibility_state TEXT NOT NULL DEFAULT 'NOT_PARTICIPATED',
            reachability_score REAL,
            internal_confidence REAL,
            submitted INTEGER NOT NULL DEFAULT 0,
            skip_reason TEXT,
            updated_at TEXT NOT NULL
        );
    """)
    conn.commit()
    return db_path, conn


def _small_synthetic_db(tmp_path: Path) -> tuple[str, sqlite3.Connection]:
    """Populate a small DB with known synthetic data for count assertions."""
    db_path, conn = _empty_koala_db(tmp_path)
    conn.executescript("""
        INSERT INTO koala_papers VALUES
            ('p1','Paper One','2026-04-25T00:00:00','2026-04-27T00:00:00','2026-04-28T00:00:00',
             'REVIEW_ACTIVE','https://x/p1.pdf',NULL,'2026-04-25T01:00:00'),
            ('p2','Paper Two','2026-04-25T00:00:00','2026-04-27T00:00:00','2026-04-28T00:00:00',
             'REVIEW_ACTIVE','https://x/p2.pdf',NULL,'2026-04-25T01:00:00');

        INSERT INTO koala_comments VALUES
            ('c1','p1',NULL,NULL,'other-agent-A','Claim X holds.','2026-04-25T02:00:00',0,1),
            ('c2','p1',NULL,'c1','other-agent-A','Reply.','2026-04-25T02:10:00',0,1),
            ('c3','p1',NULL,NULL,'our-agent','Our comment.','2026-04-25T02:20:00',1,0);

        INSERT INTO koala_extracted_claims VALUES
            ('cl1','c1','p1','Claim X holds.','empirical',0.9,0.8,'Is X true?','2026-04-25T03:00:00'),
            ('cl2','c1','p1','Also Y.','empirical',0.7,0.6,'Is Y true?','2026-04-25T03:01:00');

        INSERT INTO koala_claim_verifications VALUES
            ('v1','cl1','c1','p1','supported',0.9,'Strong support.','Quote A','m1','2026-04-25T03:05:00'),
            ('v2','cl2','c1','p1','refuted',0.85,'Paper disagrees.','Quote B','m1','2026-04-25T03:06:00'),
            ('v3','cl1','c1','p1','insufficient_evidence',0.5,'Limited.','','m1','2026-04-25T03:07:00');

        INSERT INTO koala_agent_actions VALUES
            (1,'p1','reactive_comment',NULL,NULL,'2026-04-25T04:00:00','dry_run',NULL,
             '{"run_mode":"dry_run","source_comment_id":"c1"}'),
            (2,'p1','seed_comment','ext-1',NULL,'2026-04-25T04:10:00','success',NULL,
             '{"run_mode":"live","skip_reason":null}');

        INSERT INTO koala_verdict_state VALUES
            ('p1',1,2,'ELIGIBLE_LOW_CONFIDENCE',0.6,0.7,0,NULL,'2026-04-25T05:00:00'),
            ('p2',0,0,'NOT_PARTICIPATED',NULL,NULL,0,NULL,'2026-04-25T05:00:00');
    """)
    conn.commit()
    return db_path, conn


# ---------------------------------------------------------------------------
# Test: empty DB does not crash
# ---------------------------------------------------------------------------

class TestEmptyDB:

    def test_run_audit_empty_db_no_crash(self, tmp_path):
        db_path, conn = _empty_koala_db(tmp_path)
        conn.close()
        data = run_audit(db_path)
        assert isinstance(data, dict)

    def test_empty_db_zero_papers(self, tmp_path):
        db_path, conn = _empty_koala_db(tmp_path)
        conn.close()
        data = run_audit(db_path)
        assert data["paper_coverage"]["total_papers"] == 0

    def test_empty_db_zero_claims(self, tmp_path):
        db_path, conn = _empty_koala_db(tmp_path)
        conn.close()
        data = run_audit(db_path)
        assert data["claim_coverage"]["total_extracted_claims"] == 0

    def test_empty_db_overall_red(self, tmp_path):
        db_path, conn = _empty_koala_db(tmp_path)
        conn.close()
        data = run_audit(db_path)
        assert data["readiness"]["overall_dataset_color"] == "red"

    def test_empty_db_live_red(self, tmp_path):
        db_path, conn = _empty_koala_db(tmp_path)
        conn.close()
        data = run_audit(db_path)
        assert data["readiness"]["live_interaction_color"] == "red"

    def test_empty_db_markdown_renders(self, tmp_path):
        db_path, conn = _empty_koala_db(tmp_path)
        conn.close()
        data = run_audit(db_path)
        md = _render_markdown(data, db_path)
        assert "## A. Table Inventory" in md
        assert "## H. Dataset Readiness Judgement" in md


# ---------------------------------------------------------------------------
# Test: missing optional tables do not crash
# ---------------------------------------------------------------------------

class TestMissingOptionalTables:

    def test_missing_extracted_claims_no_crash(self, tmp_path):
        db_path, conn = _empty_koala_db(tmp_path)
        conn.execute("DROP TABLE IF EXISTS koala_extracted_claims")
        conn.commit()
        conn.close()
        data = run_audit(db_path)
        assert data["claim_coverage"].get("available") is False

    def test_missing_claim_verifications_no_crash(self, tmp_path):
        db_path, conn = _empty_koala_db(tmp_path)
        conn.execute("DROP TABLE IF EXISTS koala_claim_verifications")
        conn.commit()
        conn.close()
        data = run_audit(db_path)
        assert data["verification_distribution"].get("available") is False

    def test_missing_verdict_state_no_crash(self, tmp_path):
        db_path, conn = _empty_koala_db(tmp_path)
        conn.execute("DROP TABLE IF EXISTS koala_verdict_state")
        conn.commit()
        conn.close()
        data = run_audit(db_path)
        assert data["eligibility_telemetry"].get("available") is False

    def test_missing_agent_actions_no_crash(self, tmp_path):
        db_path, conn = _empty_koala_db(tmp_path)
        conn.execute("DROP TABLE IF EXISTS koala_agent_actions")
        conn.commit()
        conn.close()
        data = run_audit(db_path)
        assert data["action_telemetry"].get("available") is False

    def test_completely_minimal_db_no_crash(self, tmp_path):
        db_path = str(tmp_path / "bare.db")
        conn = sqlite3.connect(db_path)
        conn.close()
        data = run_audit(db_path)
        assert isinstance(data, dict)
        assert data["table_inventory"]["all_tables"] == {}

    def test_missing_tables_reported_in_key_tables_missing(self, tmp_path):
        db_path, conn = _empty_koala_db(tmp_path)
        conn.execute("DROP TABLE IF EXISTS koala_extracted_claims")
        conn.commit()
        conn.close()
        data = run_audit(db_path)
        assert "koala_extracted_claims" in data["table_inventory"]["key_tables_missing"]

    def test_design_spec_tables_absent_in_missing_list(self, tmp_path):
        db_path, conn = _empty_koala_db(tmp_path)
        conn.close()
        data = run_audit(db_path)
        missing = data["table_inventory"]["key_tables_missing"]
        assert "decision_states" in missing
        assert "novelty_assessments" in missing
        assert "severe_issue_reports" in missing


# ---------------------------------------------------------------------------
# Test: small synthetic DB produces expected counts
# ---------------------------------------------------------------------------

class TestSyntheticDB:

    def test_paper_count(self, tmp_path):
        db_path, conn = _small_synthetic_db(tmp_path)
        conn.close()
        data = run_audit(db_path)
        assert data["paper_coverage"]["total_papers"] == 2

    def test_comment_total(self, tmp_path):
        db_path, conn = _small_synthetic_db(tmp_path)
        conn.close()
        data = run_audit(db_path)
        assert data["comment_coverage"]["total_comments"] == 3

    def test_unique_comment_authors(self, tmp_path):
        db_path, conn = _small_synthetic_db(tmp_path)
        conn.close()
        data = run_audit(db_path)
        assert data["comment_coverage"]["unique_authors"] == 2

    def test_top_level_vs_replies(self, tmp_path):
        db_path, conn = _small_synthetic_db(tmp_path)
        conn.close()
        data = run_audit(db_path)
        assert data["comment_coverage"]["top_level_comments"] == 2
        assert data["comment_coverage"]["reply_comments"] == 1

    def test_extracted_claims_count(self, tmp_path):
        db_path, conn = _small_synthetic_db(tmp_path)
        conn.close()
        data = run_audit(db_path)
        assert data["claim_coverage"]["total_extracted_claims"] == 2

    def test_verification_total(self, tmp_path):
        db_path, conn = _small_synthetic_db(tmp_path)
        conn.close()
        data = run_audit(db_path)
        assert data["verification_distribution"]["total_verified"] == 3

    def test_live_comment_count(self, tmp_path):
        db_path, conn = _small_synthetic_db(tmp_path)
        conn.close()
        data = run_audit(db_path)
        assert data["action_telemetry"]["live_comment_count"] == 1

    def test_verdict_state_rows(self, tmp_path):
        db_path, conn = _small_synthetic_db(tmp_path)
        conn.close()
        data = run_audit(db_path)
        assert data["eligibility_telemetry"]["total_verdict_state_rows"] == 2

    def test_papers_with_our_participation(self, tmp_path):
        db_path, conn = _small_synthetic_db(tmp_path)
        conn.close()
        data = run_audit(db_path)
        assert data["paper_coverage"]["papers_with_our_participation"] == 1

    def test_table_inventory_has_all_tables(self, tmp_path):
        db_path, conn = _small_synthetic_db(tmp_path)
        conn.close()
        data = run_audit(db_path)
        inv = data["table_inventory"]["all_tables"]
        assert "koala_papers" in inv
        assert "koala_comments" in inv
        assert "koala_extracted_claims" in inv


# ---------------------------------------------------------------------------
# Test: verification label normalization
# ---------------------------------------------------------------------------

class TestNormalizeVerdict:

    def test_supported(self):
        assert _normalize_verdict("supported") == "supported"

    def test_refuted(self):
        assert _normalize_verdict("refuted") == "refuted"

    def test_contradicted_normalizes_to_refuted(self):
        assert _normalize_verdict("contradicted") == "refuted"

    def test_contradiction_normalizes_to_refuted(self):
        assert _normalize_verdict("contradiction") == "refuted"

    def test_insufficient_evidence(self):
        assert _normalize_verdict("insufficient_evidence") == "insufficient_evidence"

    def test_not_verifiable(self):
        assert _normalize_verdict("not_verifiable") == "not_verifiable"

    def test_empty_string_becomes_unknown(self):
        assert _normalize_verdict("") == "unknown"

    def test_unknown_value_becomes_unknown(self):
        assert _normalize_verdict("some_weird_label") == "unknown"

    def test_case_insensitive(self):
        assert _normalize_verdict("Refuted") == "refuted"
        assert _normalize_verdict("SUPPORTED") == "supported"

    def test_normalized_counts_combine_refuted_variants(self, tmp_path):
        db_path, conn = _empty_koala_db(tmp_path)
        conn.executescript("""
            INSERT INTO koala_extracted_claims VALUES
                ('cl1','c1','p1','Claim.','t',0.9,0.8,'Q','2026-04-25T00:00:00');
            INSERT INTO koala_claim_verifications VALUES
                ('v1','cl1','c1','p1','refuted',0.9,'','','m','2026-04-25T00:00:00'),
                ('v2','cl1','c1','p1','contradicted',0.8,'','','m','2026-04-25T00:00:00'),
                ('v3','cl1','c1','p1','contradiction',0.7,'','','m','2026-04-25T00:00:00'),
                ('v4','cl1','c1','p1','supported',0.9,'','','m','2026-04-25T00:00:00');
        """)
        conn.commit()
        conn.close()
        data = run_audit(db_path)
        labels = data["verification_distribution"]["normalized_label_counts"]
        assert labels["refuted"] == 3
        assert labels["supported"] == 1


# ---------------------------------------------------------------------------
# Test: readiness heuristics
# ---------------------------------------------------------------------------

class TestReadinessHeuristics:

    def _make_data(self, papers=0, claims=0, verifs=0, labels=0, live=0):
        return {
            "paper_coverage": {"total_papers": papers},
            "claim_coverage": {"total_extracted_claims": claims},
            "verification_distribution": {
                "total_verified": verifs,
                "distinct_normalized_labels": labels,
            },
            "action_telemetry": {"live_comment_count": live, "live_verdict_count": 0},
        }

    def test_all_zero_is_red(self):
        data = self._make_data()
        rd = _section_readiness(data)
        assert rd["overall_dataset_color"] == "red"

    def test_green_threshold(self):
        data = self._make_data(papers=50, claims=500, verifs=500, labels=3, live=20)
        rd = _section_readiness(data)
        assert rd["overall_dataset_color"] == "green"

    def test_yellow_threshold(self):
        data = self._make_data(papers=20, claims=100, verifs=100, labels=1)
        rd = _section_readiness(data)
        assert rd["overall_dataset_color"] == "yellow"

    def test_live_green_at_20(self):
        data = self._make_data(live=20)
        rd = _section_readiness(data)
        assert rd["live_interaction_color"] == "green"

    def test_live_yellow_at_1(self):
        data = self._make_data(live=1)
        rd = _section_readiness(data)
        assert rd["live_interaction_color"] == "yellow"

    def test_live_red_at_0(self):
        data = self._make_data(live=0)
        rd = _section_readiness(data)
        assert rd["live_interaction_color"] == "red"

    def test_bottleneck_lists_missing_papers(self):
        data = self._make_data(claims=500, verifs=500, labels=3, live=20)
        rd = _section_readiness(data)
        assert any("paper" in b for b in rd["primary_bottlenecks"])


# ---------------------------------------------------------------------------
# Test: JSON output is valid and complete
# ---------------------------------------------------------------------------

class TestJSONOutput:

    def test_json_output_valid(self, tmp_path):
        db_path, conn = _small_synthetic_db(tmp_path)
        conn.close()
        data = run_audit(db_path)
        json_str = json.dumps(data, default=str)
        parsed = json.loads(json_str)
        assert "readiness" in parsed
        assert "table_inventory" in parsed

    def test_json_has_generated_at(self, tmp_path):
        db_path, conn = _empty_koala_db(tmp_path)
        conn.close()
        data = run_audit(db_path)
        assert "generated_at" in data
        assert "2026" in data["generated_at"] or "2025" in data["generated_at"]

    def test_markdown_has_all_section_headings(self, tmp_path):
        db_path, conn = _small_synthetic_db(tmp_path)
        conn.close()
        data = run_audit(db_path)
        md = _render_markdown(data, db_path)
        for heading in [
            "## A. Table Inventory",
            "## B. Paper-Level Coverage",
            "## C. Comment-Level Coverage",
            "## D. Claim-Level Coverage",
            "## E. Verification Label Distribution",
            "## F. Action / Trajectory Telemetry",
            "## G. Eligibility / Strategy Telemetry",
            "## H. Dataset Readiness Judgement",
        ]:
            assert heading in md, f"Missing section: {heading}"
