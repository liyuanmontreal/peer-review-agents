"""Tests for src/gsr_agent/datasets/export.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gsr_agent.datasets.export import (
    _try_parse_json,
    _write_jsonl,
    export_competition_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_db(tmp_path):
    """Real KoalaDB backed by a temp SQLite file with known test data."""
    from gsr_agent.storage.db import KoalaDB
    from gsr_agent.koala.models import Paper, Comment
    from datetime import datetime, timezone

    db = KoalaDB(str(tmp_path / "test.db"))
    now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)
    later = datetime(2026, 4, 30, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 7, 0, 0, 0, tzinfo=timezone.utc)

    paper = Paper(
        paper_id="paper-export-001",
        title="Export Test Paper",
        open_time=now,
        review_end_time=later,
        verdict_end_time=end,
        state="REVIEW_ACTIVE",
        pdf_url="https://example.com/paper.pdf",
    )
    db.upsert_paper(paper)

    comment_ours = Comment(
        comment_id="cmt-ours-001",
        paper_id="paper-export-001",
        author_agent_id="our-agent",
        text="Our review comment.",
        created_at=now,
        is_ours=True,
        is_citable=False,
    )
    db.upsert_comment(comment_ours)

    comment_other = Comment(
        comment_id="cmt-other-001",
        paper_id="paper-export-001",
        author_agent_id="other-agent-A",
        text="Claim: X is true.",
        created_at=now,
        is_ours=False,
        is_citable=True,
    )
    db.upsert_comment(comment_other)

    db.log_action(
        paper_id="paper-export-001",
        action_type="verdict_draft",
        status="dry_run",
        details={"reason_code": "eligible", "heat_band": "goldilocks"},
    )
    db.record_karma(
        paper_id="paper-export-001",
        action_type="comment",
        cost=1.0,
        karma_before=100.0,
        karma_after=99.0,
    )
    db.upsert_verdict_state(
        "paper-export-001",
        has_our_participation=True,
        distinct_citable_other_agents=3,
        eligibility_state="ELIGIBLE",
        internal_confidence=0.85,
    )

    db._conn.execute(
        """INSERT INTO koala_extracted_claims
           (claim_id, comment_id, paper_id, claim_text, category, confidence,
            challengeability, binary_question, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "claim-001", "cmt-other-001", "paper-export-001",
            "X is true.", "empirical", 0.8, 0.7, "Is X true?",
            "2026-04-26T12:00:00+00:00",
        ),
    )
    db._conn.commit()

    db._conn.execute(
        """INSERT INTO koala_claim_verifications
           (verification_id, claim_id, comment_id, paper_id, verdict, confidence,
            reasoning, supporting_quote, model_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "verif-001", "claim-001", "cmt-other-001", "paper-export-001",
            "refuted", 0.9, "Paper section 3 contradicts this.", "Quote…",
            "heuristic_v0", "2026-04-26T12:01:00+00:00",
        ),
    )
    db._conn.commit()

    db._conn.execute(
        """INSERT INTO koala_reactive_drafts
           (draft_id, comment_id, paper_id, recommendation, draft_text,
            analysis_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            "draft-001", "cmt-other-001", "paper-export-001",
            "react", "Draft reactive comment text.",
            '{"refuted": 1, "supported": 0}',
            "2026-04-26T12:02:00+00:00",
        ),
    )
    db._conn.commit()

    return db


def _empty_db(tmp_path):
    """KoalaDB with schema initialized but no data."""
    from gsr_agent.storage.db import KoalaDB
    return KoalaDB(str(tmp_path / "empty.db"))


# ---------------------------------------------------------------------------
# TestTryParseJson
# ---------------------------------------------------------------------------

class TestTryParseJson:

    def test_dict_string_parsed(self):
        result = _try_parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_list_string_parsed(self):
        result = _try_parse_json('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_plain_string_unchanged(self):
        assert _try_parse_json("hello world") == "hello world"

    def test_non_string_unchanged(self):
        assert _try_parse_json(42) == 42
        assert _try_parse_json(None) is None
        assert _try_parse_json(3.14) == 3.14

    def test_malformed_json_string_unchanged(self):
        bad = '{"key": broken}'
        assert _try_parse_json(bad) == bad

    def test_empty_string_unchanged(self):
        assert _try_parse_json("") == ""

    def test_nested_json_parsed(self):
        result = _try_parse_json('{"a": {"b": [1, 2]}}')
        assert result == {"a": {"b": [1, 2]}}


# ---------------------------------------------------------------------------
# TestWriteJsonl
# ---------------------------------------------------------------------------

class TestWriteJsonl:

    def test_creates_file(self, tmp_path):
        path = tmp_path / "out.jsonl"
        _write_jsonl(path, [{"a": 1}])
        assert path.exists()

    def test_returns_row_count(self, tmp_path):
        path = tmp_path / "out.jsonl"
        n = _write_jsonl(path, [{"a": 1}, {"a": 2}, {"a": 3}])
        assert n == 3

    def test_one_json_per_line(self, tmp_path):
        path = tmp_path / "out.jsonl"
        rows = [{"x": 1}, {"x": 2}]
        _write_jsonl(path, rows)
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"x": 1}
        assert json.loads(lines[1]) == {"x": 2}

    def test_empty_list_creates_empty_file(self, tmp_path):
        path = tmp_path / "out.jsonl"
        n = _write_jsonl(path, [])
        assert n == 0
        assert path.exists()
        assert path.read_text() == ""

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "out.jsonl"
        _write_jsonl(path, [])
        assert path.exists()


# ---------------------------------------------------------------------------
# TestExportDirectoryStructure
# ---------------------------------------------------------------------------

class TestExportDirectoryStructure:

    def test_raw_dir_created(self, tmp_path):
        db = _empty_db(tmp_path)
        out_dir = str(tmp_path / "exports")
        export_competition_dataset(db, out_dir)
        assert (tmp_path / "exports" / "raw").is_dir()

    def test_gsr_structured_dir_created(self, tmp_path):
        db = _empty_db(tmp_path)
        out_dir = str(tmp_path / "exports")
        export_competition_dataset(db, out_dir)
        assert (tmp_path / "exports" / "gsr_structured").is_dir()

    def test_policy_dir_created(self, tmp_path):
        db = _empty_db(tmp_path)
        out_dir = str(tmp_path / "exports")
        export_competition_dataset(db, out_dir)
        assert (tmp_path / "exports" / "policy").is_dir()

    def test_reports_dir_created(self, tmp_path):
        db = _empty_db(tmp_path)
        out_dir = str(tmp_path / "exports")
        export_competition_dataset(db, out_dir)
        assert (tmp_path / "exports" / "reports").is_dir()

    def test_all_expected_jsonl_files_created(self, tmp_path):
        db = _empty_db(tmp_path)
        out_dir = str(tmp_path / "exports")
        export_competition_dataset(db, out_dir)
        expected = [
            "raw/papers.jsonl",
            "raw/comments.jsonl",
            "raw/actions.jsonl",
            "raw/karma_ledger.jsonl",
            "raw/verdict_state.jsonl",
            "gsr_structured/extracted_claims.jsonl",
            "gsr_structured/claim_verifications.jsonl",
            "gsr_structured/reactive_drafts.jsonl",
            "policy/run_summary.jsonl",
            "policy/action_traces.jsonl",
        ]
        root = tmp_path / "exports"
        for rel in expected:
            assert (root / rel).exists(), f"Missing: {rel}"

    def test_dataset_card_created(self, tmp_path):
        db = _empty_db(tmp_path)
        export_competition_dataset(db, str(tmp_path / "exports"))
        assert (tmp_path / "exports" / "reports" / "dataset_card.md").exists()

    def test_summary_stats_created(self, tmp_path):
        db = _empty_db(tmp_path)
        export_competition_dataset(db, str(tmp_path / "exports"))
        assert (tmp_path / "exports" / "reports" / "summary_stats.md").exists()


# ---------------------------------------------------------------------------
# TestExportCounts
# ---------------------------------------------------------------------------

class TestExportCounts:

    def test_returns_dict(self, tmp_path):
        db = _empty_db(tmp_path)
        result = export_competition_dataset(db, str(tmp_path / "exports"))
        assert isinstance(result, dict)

    def test_empty_db_all_counts_zero(self, tmp_path):
        db = _empty_db(tmp_path)
        counts = export_competition_dataset(db, str(tmp_path / "exports"))
        assert counts["papers.jsonl"] == 0
        assert counts["comments.jsonl"] == 0
        assert counts["actions.jsonl"] == 0
        assert counts["extracted_claims.jsonl"] == 0

    def test_populated_papers_count(self, tmp_path):
        db = _make_db(tmp_path)
        counts = export_competition_dataset(db, str(tmp_path / "exports"))
        assert counts["papers.jsonl"] == 1

    def test_populated_comments_count(self, tmp_path):
        db = _make_db(tmp_path)
        counts = export_competition_dataset(db, str(tmp_path / "exports"))
        assert counts["comments.jsonl"] == 2

    def test_populated_actions_count(self, tmp_path):
        db = _make_db(tmp_path)
        counts = export_competition_dataset(db, str(tmp_path / "exports"))
        assert counts["actions.jsonl"] == 1

    def test_populated_karma_count(self, tmp_path):
        db = _make_db(tmp_path)
        counts = export_competition_dataset(db, str(tmp_path / "exports"))
        assert counts["karma_ledger.jsonl"] == 1

    def test_populated_extracted_claims_count(self, tmp_path):
        db = _make_db(tmp_path)
        counts = export_competition_dataset(db, str(tmp_path / "exports"))
        assert counts["extracted_claims.jsonl"] == 1

    def test_populated_claim_verifications_count(self, tmp_path):
        db = _make_db(tmp_path)
        counts = export_competition_dataset(db, str(tmp_path / "exports"))
        assert counts["claim_verifications.jsonl"] == 1

    def test_populated_reactive_drafts_count(self, tmp_path):
        db = _make_db(tmp_path)
        counts = export_competition_dataset(db, str(tmp_path / "exports"))
        assert counts["reactive_drafts.jsonl"] == 1

    def test_policy_run_summary_has_paper_row(self, tmp_path):
        db = _make_db(tmp_path)
        counts = export_competition_dataset(db, str(tmp_path / "exports"))
        assert counts["run_summary.jsonl"] == 1

    def test_policy_action_traces_has_action_row(self, tmp_path):
        db = _make_db(tmp_path)
        counts = export_competition_dataset(db, str(tmp_path / "exports"))
        assert counts["action_traces.jsonl"] == 1


# ---------------------------------------------------------------------------
# TestJsonlContent
# ---------------------------------------------------------------------------

class TestJsonlContent:

    def _load_jsonl(self, path: Path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def test_papers_jsonl_has_paper_id(self, tmp_path):
        db = _make_db(tmp_path)
        out = tmp_path / "exports"
        export_competition_dataset(db, str(out))
        rows = self._load_jsonl(out / "raw" / "papers.jsonl")
        assert rows[0]["paper_id"] == "paper-export-001"

    def test_comments_jsonl_has_both_comments(self, tmp_path):
        db = _make_db(tmp_path)
        out = tmp_path / "exports"
        export_competition_dataset(db, str(out))
        rows = self._load_jsonl(out / "raw" / "comments.jsonl")
        ids = {r["comment_id"] for r in rows}
        assert "cmt-ours-001" in ids
        assert "cmt-other-001" in ids

    def test_actions_json_details_parsed(self, tmp_path):
        db = _make_db(tmp_path)
        out = tmp_path / "exports"
        export_competition_dataset(db, str(out))
        rows = self._load_jsonl(out / "raw" / "actions.jsonl")
        details = rows[0]["details"]
        assert isinstance(details, dict), "details should be parsed from JSON string"
        assert details.get("reason_code") == "eligible"

    def test_reactive_drafts_analysis_json_parsed(self, tmp_path):
        db = _make_db(tmp_path)
        out = tmp_path / "exports"
        export_competition_dataset(db, str(out))
        rows = self._load_jsonl(out / "gsr_structured" / "reactive_drafts.jsonl")
        analysis = rows[0]["analysis_json"]
        assert isinstance(analysis, dict), "analysis_json should be parsed"
        assert analysis.get("refuted") == 1

    def test_run_summary_includes_action_count(self, tmp_path):
        db = _make_db(tmp_path)
        out = tmp_path / "exports"
        export_competition_dataset(db, str(out))
        rows = self._load_jsonl(out / "policy" / "run_summary.jsonl")
        assert rows[0]["action_count"] == 1

    def test_verdict_state_jsonl_has_eligibility(self, tmp_path):
        db = _make_db(tmp_path)
        out = tmp_path / "exports"
        export_competition_dataset(db, str(out))
        rows = self._load_jsonl(out / "raw" / "verdict_state.jsonl")
        assert rows[0]["eligibility_state"] == "ELIGIBLE"


# ---------------------------------------------------------------------------
# TestMissingTablesGraceful
# ---------------------------------------------------------------------------

class TestMissingTablesGraceful:

    def test_missing_optional_table_no_crash(self, tmp_path):
        """If a table doesn't exist, export produces 0 rows without raising."""
        from gsr_agent.storage.db import KoalaDB
        db = KoalaDB(str(tmp_path / "minimal.db"))
        # Drop an optional table to simulate an older schema
        db._conn.execute("DROP TABLE IF EXISTS koala_extracted_claims")
        db._conn.commit()

        counts = export_competition_dataset(db, str(tmp_path / "exports"))
        assert counts["extracted_claims.jsonl"] == 0

    def test_missing_table_produces_empty_jsonl(self, tmp_path):
        from gsr_agent.storage.db import KoalaDB
        db = KoalaDB(str(tmp_path / "minimal2.db"))
        db._conn.execute("DROP TABLE IF EXISTS koala_claim_verifications")
        db._conn.commit()

        out = tmp_path / "exports"
        export_competition_dataset(db, str(out))
        path = out / "gsr_structured" / "claim_verifications.jsonl"
        assert path.exists()
        assert path.read_text() == ""


# ---------------------------------------------------------------------------
# TestReportContent
# ---------------------------------------------------------------------------

class TestReportContent:

    def test_dataset_card_contains_purpose(self, tmp_path):
        db = _empty_db(tmp_path)
        export_competition_dataset(db, str(tmp_path / "exports"))
        card = (tmp_path / "exports" / "reports" / "dataset_card.md").read_text()
        assert "Purpose" in card

    def test_dataset_card_contains_privacy_note(self, tmp_path):
        db = _empty_db(tmp_path)
        export_competition_dataset(db, str(tmp_path / "exports"))
        card = (tmp_path / "exports" / "reports" / "dataset_card.md").read_text()
        assert "privacy" in card.lower() or "Privacy" in card

    def test_dataset_card_contains_generated_at(self, tmp_path):
        db = _empty_db(tmp_path)
        export_competition_dataset(db, str(tmp_path / "exports"))
        card = (tmp_path / "exports" / "reports" / "dataset_card.md").read_text()
        assert "Generated" in card
        assert "2026" in card

    def test_dataset_card_contains_file_list(self, tmp_path):
        db = _make_db(tmp_path)
        export_competition_dataset(db, str(tmp_path / "exports"))
        card = (tmp_path / "exports" / "reports" / "dataset_card.md").read_text()
        assert "papers.jsonl" in card

    def test_summary_stats_contains_papers_count(self, tmp_path):
        db = _make_db(tmp_path)
        export_competition_dataset(db, str(tmp_path / "exports"))
        stats = (tmp_path / "exports" / "reports" / "summary_stats.md").read_text()
        assert "Papers" in stats
        assert "1" in stats

    def test_summary_stats_has_table_format(self, tmp_path):
        db = _empty_db(tmp_path)
        export_competition_dataset(db, str(tmp_path / "exports"))
        stats = (tmp_path / "exports" / "reports" / "summary_stats.md").read_text()
        assert "|" in stats


# ---------------------------------------------------------------------------
# TestCLISmokeTest
# ---------------------------------------------------------------------------

class TestCLISmokeTest:

    def test_cli_runs_and_prints_done(self, tmp_path, capsys):
        from gsr_agent.datasets.export import main

        db_path = str(tmp_path / "smoke.db")
        out_dir = str(tmp_path / "smoke_out")

        with patch("sys.argv", ["export", "--db", db_path, "--out", out_dir]):
            main()

        out = capsys.readouterr().out
        assert "[export] START" in out
        assert "[export] DONE" in out

    def test_cli_creates_output_dir(self, tmp_path, capsys):
        from gsr_agent.datasets.export import main

        db_path = str(tmp_path / "smoke2.db")
        out_dir = str(tmp_path / "smoke_out2")

        with patch("sys.argv", ["export", "--db", db_path, "--out", out_dir]):
            main()

        assert (Path(out_dir) / "raw").is_dir()

    def test_cli_prints_file_counts(self, tmp_path, capsys):
        from gsr_agent.datasets.export import main

        db_path = str(tmp_path / "smoke3.db")
        out_dir = str(tmp_path / "smoke_out3")

        with patch("sys.argv", ["export", "--db", db_path, "--out", out_dir]):
            main()

        out = capsys.readouterr().out
        assert "papers.jsonl" in out
