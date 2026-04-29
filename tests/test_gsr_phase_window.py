"""Phase-window awareness tests.

Covers compute_phase_window for all five required scenarios, the safety-buffer
gate, 409-to-window_closed handling, and the window_skip early exit in _process_paper.
"""
from __future__ import annotations

import json
import urllib.error
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from gsr_agent.rules.timeline import (
    DELIBERATION_DURATION_H,
    REVIEW_DURATION_H,
    SAFETY_BUFFER_S,
    PhaseWindow,
    compute_phase_window,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_OPEN = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _h(hours: float) -> datetime:
    return _OPEN + timedelta(hours=hours)


def _pw(state: str, hours_elapsed: float, deliberating_at=None) -> PhaseWindow:
    return compute_phase_window(_h(hours_elapsed), state, _OPEN, deliberating_at)


# ---------------------------------------------------------------------------
# 1. in_review within 48h
# ---------------------------------------------------------------------------

class TestInReviewWithin48h:
    def test_phase_is_comment(self):
        assert _pw("REVIEW_ACTIVE", 24).phase == "comment"

    def test_ends_at_is_open_plus_48h(self):
        assert _pw("REVIEW_ACTIVE", 24).ends_at == _OPEN + timedelta(hours=REVIEW_DURATION_H)

    def test_seconds_left_matches_remaining_hours(self):
        pw = _pw("REVIEW_ACTIVE", 24)
        assert pw.seconds_left == pytest.approx(timedelta(hours=24).total_seconds(), abs=1)

    def test_seconds_left_positive(self):
        assert _pw("REVIEW_ACTIVE", 1).seconds_left > 0

    def test_above_safety_buffer(self):
        assert _pw("REVIEW_ACTIVE", 24).seconds_left > SAFETY_BUFFER_S

    def test_gate_allows_comment(self):
        pw = _pw("REVIEW_ACTIVE", 24)
        assert pw.phase == "comment" and pw.seconds_left > SAFETY_BUFFER_S


# ---------------------------------------------------------------------------
# 2. in_review expired
# ---------------------------------------------------------------------------

class TestInReviewExpired:
    def test_phase_is_still_comment(self):
        # state claims REVIEW_ACTIVE but time has passed — still "comment" phase type
        pw = _pw("REVIEW_ACTIVE", 50)
        assert pw.phase == "comment"

    def test_seconds_left_negative(self):
        assert _pw("REVIEW_ACTIVE", 50).seconds_left < 0

    def test_ends_at_still_48h(self):
        pw = _pw("REVIEW_ACTIVE", 50)
        assert pw.ends_at == _OPEN + timedelta(hours=REVIEW_DURATION_H)

    def test_gate_blocks_comment(self):
        pw = _pw("REVIEW_ACTIVE", 50)
        assert not (pw.phase == "comment" and pw.seconds_left > SAFETY_BUFFER_S)

    def test_exactly_at_boundary(self):
        pw = compute_phase_window(_OPEN + timedelta(hours=48), "REVIEW_ACTIVE", _OPEN)
        assert pw.seconds_left == pytest.approx(0.0, abs=1)


# ---------------------------------------------------------------------------
# 3. deliberating within 24h
# ---------------------------------------------------------------------------

class TestDeliberatingWithin24h:
    def _delib(self, delib_offset_h: float, elapsed_from_delib_h: float) -> PhaseWindow:
        delib_start = _OPEN + timedelta(hours=delib_offset_h)
        now = delib_start + timedelta(hours=elapsed_from_delib_h)
        return compute_phase_window(now, "VERDICT_ACTIVE", _OPEN, deliberating_at=delib_start)

    def test_phase_is_verdict(self):
        assert self._delib(48, 10).phase == "verdict"

    def test_ends_at_is_delib_start_plus_24h(self):
        delib_start = _OPEN + timedelta(hours=48)
        pw = compute_phase_window(
            delib_start + timedelta(hours=10), "VERDICT_ACTIVE", _OPEN, deliberating_at=delib_start
        )
        assert pw.ends_at == delib_start + timedelta(hours=DELIBERATION_DURATION_H)

    def test_seconds_left_positive(self):
        assert self._delib(48, 10).seconds_left > 0

    def test_seconds_left_matches_remaining(self):
        pw = self._delib(48, 10)
        assert pw.seconds_left == pytest.approx(timedelta(hours=14).total_seconds(), abs=1)

    def test_above_safety_buffer(self):
        assert self._delib(48, 10).seconds_left > SAFETY_BUFFER_S

    def test_gate_allows_verdict(self):
        pw = self._delib(48, 10)
        assert pw.phase == "verdict" and pw.seconds_left > SAFETY_BUFFER_S

    def test_early_deliberation_start(self):
        # deliberating_at earlier than open + 48h still uses the provided timestamp
        delib_start = _OPEN + timedelta(hours=46)
        pw = compute_phase_window(
            delib_start + timedelta(hours=5), "VERDICT_ACTIVE", _OPEN, deliberating_at=delib_start
        )
        assert pw.ends_at == delib_start + timedelta(hours=24)


# ---------------------------------------------------------------------------
# 4. deliberating expired
# ---------------------------------------------------------------------------

class TestDeliberatingExpired:
    def test_phase_is_still_verdict(self):
        delib_start = _OPEN + timedelta(hours=48)
        pw = compute_phase_window(
            delib_start + timedelta(hours=25), "VERDICT_ACTIVE", _OPEN, deliberating_at=delib_start
        )
        assert pw.phase == "verdict"

    def test_seconds_left_negative(self):
        delib_start = _OPEN + timedelta(hours=48)
        pw = compute_phase_window(
            delib_start + timedelta(hours=25), "VERDICT_ACTIVE", _OPEN, deliberating_at=delib_start
        )
        assert pw.seconds_left < 0

    def test_gate_blocks_verdict(self):
        delib_start = _OPEN + timedelta(hours=48)
        pw = compute_phase_window(
            delib_start + timedelta(hours=25), "VERDICT_ACTIVE", _OPEN, deliberating_at=delib_start
        )
        assert not (pw.phase == "verdict" and pw.seconds_left > SAFETY_BUFFER_S)

    def test_exactly_at_verdict_boundary(self):
        delib_start = _OPEN + timedelta(hours=48)
        pw = compute_phase_window(
            delib_start + timedelta(hours=24), "VERDICT_ACTIVE", _OPEN, deliberating_at=delib_start
        )
        assert pw.seconds_left == pytest.approx(0.0, abs=1)


# ---------------------------------------------------------------------------
# 5. deliberating_at null fallback
# ---------------------------------------------------------------------------

class TestDeliberatingNullFallback:
    def test_phase_is_verdict(self):
        pw = _pw("VERDICT_ACTIVE", 60, deliberating_at=None)
        assert pw.phase == "verdict"

    def test_ends_at_is_open_plus_72h(self):
        pw = _pw("VERDICT_ACTIVE", 60, deliberating_at=None)
        assert pw.ends_at == _OPEN + timedelta(hours=REVIEW_DURATION_H + DELIBERATION_DURATION_H)

    def test_seconds_left_correct_at_60h(self):
        # fallback: delib_start = open + 48h, ends = open + 72h, now = open + 60h → 12h left
        pw = _pw("VERDICT_ACTIVE", 60, deliberating_at=None)
        assert pw.seconds_left == pytest.approx(timedelta(hours=12).total_seconds(), abs=1)

    def test_seconds_left_positive_mid_window(self):
        pw = _pw("VERDICT_ACTIVE", 55, deliberating_at=None)
        assert pw.seconds_left > 0

    def test_expired_with_null_deliberating_at(self):
        pw = _pw("VERDICT_ACTIVE", 73, deliberating_at=None)
        assert pw.seconds_left < 0

    def test_gate_allows_verdict_mid_window(self):
        pw = _pw("VERDICT_ACTIVE", 55, deliberating_at=None)
        assert pw.phase == "verdict" and pw.seconds_left > SAFETY_BUFFER_S

    def test_gate_blocks_verdict_past_72h(self):
        pw = _pw("VERDICT_ACTIVE", 73, deliberating_at=None)
        assert not (pw.phase == "verdict" and pw.seconds_left > SAFETY_BUFFER_S)


# ---------------------------------------------------------------------------
# 6. Safety buffer boundary
# ---------------------------------------------------------------------------

class TestSafetyBuffer:
    def test_inside_buffer_blocks_comment(self):
        # 4 minutes before comment window closes — inside 10-minute buffer
        now = _OPEN + timedelta(hours=48) - timedelta(minutes=4)
        pw = compute_phase_window(now, "REVIEW_ACTIVE", _OPEN)
        assert not (pw.phase == "comment" and pw.seconds_left > SAFETY_BUFFER_S)

    def test_outside_buffer_allows_comment(self):
        # 15 minutes before comment window closes — outside buffer
        now = _OPEN + timedelta(hours=48) - timedelta(minutes=15)
        pw = compute_phase_window(now, "REVIEW_ACTIVE", _OPEN)
        assert pw.phase == "comment" and pw.seconds_left > SAFETY_BUFFER_S

    def test_inside_buffer_blocks_verdict(self):
        delib_start = _OPEN + timedelta(hours=48)
        now = delib_start + timedelta(hours=24) - timedelta(minutes=4)
        pw = compute_phase_window(now, "VERDICT_ACTIVE", _OPEN, deliberating_at=delib_start)
        assert not (pw.phase == "verdict" and pw.seconds_left > SAFETY_BUFFER_S)

    def test_outside_buffer_allows_verdict(self):
        delib_start = _OPEN + timedelta(hours=48)
        now = delib_start + timedelta(hours=24) - timedelta(minutes=15)
        pw = compute_phase_window(now, "VERDICT_ACTIVE", _OPEN, deliberating_at=delib_start)
        assert pw.phase == "verdict" and pw.seconds_left > SAFETY_BUFFER_S

    def test_safety_buffer_constant_is_600s(self):
        assert SAFETY_BUFFER_S == 600


# ---------------------------------------------------------------------------
# 7. EXPIRED / NEW / unknown states
# ---------------------------------------------------------------------------

class TestExpiredState:
    def test_expired_state_phase(self):
        assert _pw("EXPIRED", 75).phase == "expired"

    def test_new_state_phase(self):
        assert _pw("NEW", 0).phase == "expired"

    def test_unknown_state_phase(self):
        assert _pw("UNKNOWN_STATE", 0).phase == "expired"

    def test_expired_gate_always_blocks_comment(self):
        pw = _pw("EXPIRED", 75)
        assert not (pw.phase == "comment" and pw.seconds_left > SAFETY_BUFFER_S)

    def test_expired_gate_always_blocks_verdict(self):
        pw = _pw("EXPIRED", 75)
        assert not (pw.phase == "verdict" and pw.seconds_left > SAFETY_BUFFER_S)


# ---------------------------------------------------------------------------
# 8. Naive datetime handling
# ---------------------------------------------------------------------------

class TestNaiveDatetime:
    def test_naive_now_treated_as_utc(self):
        now_naive = datetime(2026, 1, 1, 12, 0, 0)  # no tzinfo
        pw = compute_phase_window(now_naive, "REVIEW_ACTIVE", _OPEN)
        assert pw.phase == "comment"
        assert pw.seconds_left == pytest.approx(timedelta(hours=36).total_seconds(), abs=1)

    def test_naive_open_time_treated_as_utc(self):
        open_naive = datetime(2026, 1, 1, 0, 0, 0)  # no tzinfo
        now = _OPEN + timedelta(hours=12)
        pw = compute_phase_window(now, "REVIEW_ACTIVE", open_naive)
        assert pw.phase == "comment"


# ---------------------------------------------------------------------------
# 9. window_skipped counter in process_paper (integration smoke)
# ---------------------------------------------------------------------------

def _make_mock_db():
    db = MagicMock()
    db.get_papers.return_value = []
    db.has_recent_reactive_action_for_comment.return_value = False
    db.has_recent_verdict_action_for_paper.return_value = False
    db.has_prior_participation.return_value = True
    db.get_strongest_contradiction_confidence.return_value = None
    db.get_phase5a_stats.return_value = {
        "react_count": 0, "skip_count": 0,
        "unclear_count": 0, "comments_analyzed": 0,
    }
    db.get_comment_stats.return_value = {"citable_other": 0, "ours": 0, "total": 0}
    db.get_distinct_other_agent_count.return_value = 0
    db.get_strongest_contradiction_confidence.return_value = None
    db.get_run_summary_rows.return_value = []
    return db


class TestWindowSkipCounter:
    """Verify window_skipped counter increments when paper's window is closed."""

    def _run_loop(self, paper_state: str, open_time: datetime):
        from gsr_agent.koala.models import Paper
        from gsr_agent.orchestration.operational_loop import run_operational_loop

        db = _make_mock_db()
        now = datetime.now(timezone.utc)
        paper_row = {
            "paper_id": "p-expired",
            "title": "Expired Paper",
            "open_time": open_time.isoformat(),
            "review_end_time": (open_time + timedelta(hours=48)).isoformat(),
            "verdict_end_time": (open_time + timedelta(hours=72)).isoformat(),
            "state": paper_state,
            "pdf_url": "",
            "local_pdf_path": None,
            "deliberating_at": None,
        }
        db.get_papers.return_value = [paper_row]
        return run_operational_loop(db, now, test_mode=True)

    def test_expired_paper_increments_window_skipped(self):
        # Paper opened 49h ago with state="REVIEW_ACTIVE" (stale state mismatch):
        # get_paper_phase → VERDICT_ACTIVE → competition filter passes (VERDICT_READY),
        # but compute_phase_window("REVIEW_ACTIVE"...) sees seconds_left < 0 → window_skipped.
        old_open = datetime.now(timezone.utc) - timedelta(hours=49)
        counters = self._run_loop("REVIEW_ACTIVE", old_open)
        assert counters["window_skipped"] == 1
        assert counters["papers_processed"] == 1

    def test_expired_paper_not_in_skipped_counter(self):
        old_open = datetime.now(timezone.utc) - timedelta(hours=100)
        counters = self._run_loop("REVIEW_ACTIVE", old_open)
        assert counters["skipped"] == 0

    def test_active_paper_not_in_window_skipped(self):
        # Paper opened 1 hour ago — still active
        recent_open = datetime.now(timezone.utc) - timedelta(hours=1)
        counters = self._run_loop("REVIEW_ACTIVE", recent_open)
        assert counters["window_skipped"] == 0


# ---------------------------------------------------------------------------
# 10. 409 raises KoalaWindowClosedError
# ---------------------------------------------------------------------------

class TestKoalaWindowClosedError:
    def test_request_raises_window_closed_on_409(self):
        from gsr_agent.koala.client import KoalaClient
        from gsr_agent.koala.errors import KoalaWindowClosedError

        client = KoalaClient(api_token="tok", test_mode=False)

        http_err = urllib.error.HTTPError(
            url="https://koala.science/api/v1/comments/",
            code=409,
            msg="Conflict",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )

        with patch("urllib.request.urlopen", side_effect=http_err):
            with pytest.raises(KoalaWindowClosedError):
                client._request("POST", "/comments/", body={"paper_id": "p1"})

    def test_409_does_not_trigger_retry(self):
        from gsr_agent.koala.client import KoalaClient
        from gsr_agent.koala.errors import KoalaWindowClosedError

        client = KoalaClient(api_token="tok", test_mode=False, _retry_delay_s=0)

        http_err = urllib.error.HTTPError(
            url="https://koala.science/api/v1/verdicts/",
            code=409,
            msg="Conflict",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )
        call_count = 0

        def fake_urlopen(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise http_err

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with pytest.raises(KoalaWindowClosedError):
                client._request("POST", "/verdicts/", body={})

        assert call_count == 1  # no retries on 409
