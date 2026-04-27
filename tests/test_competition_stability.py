"""Competition-stability patch tests (Tasks A–D, G).

Covers:
  1. Stale deferred marker → fresh-session cleanup, not infinite resume loop
  2. Unsafe curl -v / Authorization: Bearer commands are blocked
  3. Guessed comments endpoints are rejected
  4. Verified GET /api/v1/comments/paper/{id} path is allowed
  5. Verdict accepts exactly 3 citeable other-agent comments
  6. Verdict rejects 2 citeable other-agent comments
  7. VERDICT_READY (deliberating + participation) is prioritised over SEED
"""

from datetime import datetime, timedelta, timezone

import pytest

# ---------------------------------------------------------------------------
# Task A: stale session handling in generated launch script
# ---------------------------------------------------------------------------

from reva.tmux import _make_run_block


def test_stale_session_clears_and_logs_competition_message():
    block = _make_run_block(
        backend_command="claude -p 'work'",
        resume_command="claude --resume $SESSION_ID",
        timeout_expr="${SESSION_TIMEOUT}",
    )
    assert "[competition] stale_session_cleared reason=no_deferred_tool_marker" in block


def test_stale_session_block_removes_last_session_id():
    block = _make_run_block(
        backend_command="claude -p 'work'",
        resume_command="claude --resume $SESSION_ID",
        timeout_expr="${SESSION_TIMEOUT}",
    )
    # After the stale marker is detected, last_session_id must be cleared
    stale_section = block[block.find("_STALE"):]
    assert "rm -f last_session_id" in stale_section


def test_stale_session_checks_sigpipe_exit_code():
    block = _make_run_block(
        backend_command="claude -p 'work'",
        resume_command="claude --resume $SESSION_ID",
        timeout_expr="${SESSION_TIMEOUT}",
    )
    # Exit code 141 (SIGPIPE after successful post) is treated as stale
    assert "RESUME_RC" in block and "141" in block


def test_fresh_failure_logs_reva_message():
    block = _make_run_block(
        backend_command="claude -p 'work'",
        resume_command="claude --resume $SESSION_ID",
        timeout_expr="${SESSION_TIMEOUT}",
    )
    # Non-stale non-zero exit still gets the original [reva] message
    assert "[reva] resume failed" in block


# ---------------------------------------------------------------------------
# Task B: unsafe command guard
# ---------------------------------------------------------------------------

from agent_definition.harness.safety import check_bash_safety


@pytest.mark.parametrize("code,expected_reason", [
    ('import subprocess; subprocess.run(["curl", "-v", "https://koala.science"])', "curl_verbose"),
    ('import os; os.system("curl -Lv https://koala.science")', "curl_verbose"),
    ('import os; os.system("curl -sLv https://koala.science/api")', "curl_verbose"),
    ('import subprocess; subprocess.run(["curl", "-i", "https://koala.science"])', "curl_print_headers"),
    ('headers = {"Authorization": "Bearer " + api_key}; r = requests.get(url, headers=headers)', "auth_header_in_command"),
    ('token = "cs_abcdefghijk12345"; url = f"https://koala.science?key={token}"', "koala_token_literal"),
    ('key = "sk-ant-api03-abcdefghij12345678"; client = anthropic.Anthropic(api_key=key)', "anthropic_key_literal"),
])
def test_unsafe_command_blocked(code, expected_reason):
    is_safe, reason = check_bash_safety(code)
    assert not is_safe
    assert reason == expected_reason


def test_safe_curl_command_allowed():
    code = 'import subprocess; subprocess.run(["curl", "https://koala.science/api/v1/papers"])'
    is_safe, reason = check_bash_safety(code)
    assert is_safe
    assert reason == ""


def test_safe_requests_via_client_wrapper_allowed():
    code = (
        "import os\n"
        "from gsr_agent.koala.client import KoalaClient\n"
        "client = KoalaClient(api_token=os.environ['KOALA_API_TOKEN'])\n"
        "comments = client.list_comments(paper_id)"
    )
    # Uses the client wrapper — no explicit Authorization header
    is_safe, reason = check_bash_safety(code)
    assert is_safe


# ---------------------------------------------------------------------------
# Task C: Koala endpoint validation
# ---------------------------------------------------------------------------

from agent_definition.harness.safety import is_allowed_comments_endpoint


@pytest.mark.parametrize("path,method", [
    ("/comments/?paper_id=abc123", "GET"),
    ("/api/v1/comments/?paper_id=abc123", "GET"),
    ("/actors/some-actor-id/comments", "GET"),
    ("/papers/abc123?include_comments=true", "GET"),
    ("/comments/", "GET"),
    ("/api/v1/comments/", "GET"),
])
def test_guessed_comment_endpoints_are_blocked(path, method):
    allowed, reason = is_allowed_comments_endpoint(path, method)
    assert not allowed, f"Expected {path!r} ({method}) to be blocked, got allowed"


@pytest.mark.parametrize("path,method", [
    ("/api/v1/comments/paper/abc123def456-0000-0000-0000-000000000000", "GET"),
    ("/api/v1/comments/paper/abc123def456-0000-0000-0000-000000000000?limit=50", "GET"),
    ("/api/v1/comments/", "POST"),
    ("/api/v1/comments", "POST"),
])
def test_verified_comment_endpoints_are_allowed(path, method):
    allowed, reason = is_allowed_comments_endpoint(path, method)
    assert allowed, f"Expected {path!r} ({method}) to be allowed, got blocked: {reason}"


# ---------------------------------------------------------------------------
# Tasks D + G: Verdict citation threshold (MIN_VERDICT_CITATIONS = 3)
# ---------------------------------------------------------------------------

from gsr_agent.rules.verdict_eligibility import (
    MIN_DISTINCT_OTHER_AGENTS,
    VerdictEligibilityInput,
    can_submit_verdict,
    compute_eligibility_state,
    EligibilityState,
)

UTC = timezone.utc
_OPEN = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)


def _at(hours: float) -> datetime:
    return _OPEN + timedelta(hours=hours)


def _make(distinct_citable_other_agents: int = 3) -> VerdictEligibilityInput:
    return VerdictEligibilityInput(
        paper_id="paper-comp-001",
        has_our_participation=True,
        distinct_citable_other_agents=distinct_citable_other_agents,
        open_time=_OPEN,
        audit_artifact_ready=True,
        internal_score_confidence=0.8,
    )


_NOW_VERDICT = _at(60)   # inside SUBMISSION_WINDOW (60–72h)


def test_min_verdict_citations_constant_is_3():
    assert MIN_DISTINCT_OTHER_AGENTS == 3


def test_verdict_accepts_exactly_3_citeable_comments():
    assert can_submit_verdict(_make(distinct_citable_other_agents=3), _NOW_VERDICT) is True


def test_verdict_rejects_2_citeable_comments():
    assert can_submit_verdict(_make(distinct_citable_other_agents=2), _NOW_VERDICT) is False


def test_verdict_eligibility_state_insufficient_citations():
    state, reason = compute_eligibility_state(_make(distinct_citable_other_agents=2), _NOW_VERDICT)
    assert state == EligibilityState.PARTICIPATED_BUT_NOT_ENOUGH_OTHERS
    assert "2/3" in reason


# ---------------------------------------------------------------------------
# Task G-7: VERDICT_READY is prioritised over SEED
# ---------------------------------------------------------------------------

from gsr_agent.strategy.opportunity_manager import (
    OPPORTUNITY_PRIORITY,
    PaperOpportunity,
    classify_paper_opportunity,
)
from gsr_agent.koala.models import Paper


def _make_paper_for_classify() -> Paper:
    from gsr_agent.rules.timeline import compute_paper_windows
    w = compute_paper_windows(_OPEN)
    return Paper(
        paper_id="paper-classify-test",
        title="Test Paper",
        open_time=w.open_time,
        review_end_time=w.review_end_time,
        verdict_end_time=w.verdict_end_time,
        state="REVIEW_ACTIVE",
    )


def test_deliberating_paper_with_participation_is_verdict_ready():
    paper = _make_paper_for_classify()
    now = _OPEN + timedelta(hours=62)  # inside SUBMISSION_WINDOW (60–72h)
    opp = classify_paper_opportunity(paper, has_participated=True, karma_remaining=90.0, now=now)
    assert opp == PaperOpportunity.VERDICT_READY


def test_in_review_paper_without_participation_is_seed():
    paper = _make_paper_for_classify()
    now = _OPEN + timedelta(hours=6)   # inside SEED_WINDOW (0–12h)
    opp = classify_paper_opportunity(paper, has_participated=False, karma_remaining=90.0, now=now)
    assert opp == PaperOpportunity.SEED


def test_verdict_ready_is_higher_priority_than_seed():
    assert (
        OPPORTUNITY_PRIORITY[PaperOpportunity.VERDICT_READY]
        < OPPORTUNITY_PRIORITY[PaperOpportunity.SEED]
    ), "VERDICT_READY must have a lower (higher-priority) number than SEED"


def test_opportunity_priority_ordering():
    prio = OPPORTUNITY_PRIORITY
    assert prio[PaperOpportunity.VERDICT_READY] < prio[PaperOpportunity.FOLLOWUP]
    assert prio[PaperOpportunity.FOLLOWUP] < prio[PaperOpportunity.SEED]
    assert prio[PaperOpportunity.SEED] < prio[PaperOpportunity.SKIP]
