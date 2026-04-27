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


# ---------------------------------------------------------------------------
# Task F: Secret env-var exfiltration guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code,expected_reason", [
    ('echo "KOALA_API_KEY=${KOALA_API_KEY}"', "secret_env_echo"),
    ('echo "$COALESCENCE_API_KEY"', "secret_env_echo"),
    ('echo "${ANTHROPIC_API_KEY}"', "secret_env_echo"),
    ('echo "$MY_API_KEY"', "secret_env_echo"),
    ('printenv KOALA_API_KEY', "secret_env_echo"),
    ('printenv COALESCENCE_API_KEY', "secret_env_echo"),
    ('printenv | grep -i koala', "secret_env_grep"),
    ('env | grep -i api_key', "secret_env_grep"),
    ('export | grep API_KEY', "secret_env_grep"),
    ('set | grep API_KEY', "secret_env_grep"),
    ('printenv | grep -i coalescence', "secret_env_grep"),
    ('env | grep -i anthropic', "secret_env_grep"),
])
def test_secret_env_exfiltration_blocked(code, expected_reason):
    is_safe, reason = check_bash_safety(code)
    assert not is_safe, f"Expected {code!r} to be blocked"
    assert reason == expected_reason


@pytest.mark.parametrize("code", [
    'printenv | grep PATH',
    'env | grep HOME',
    'echo "$PWD"',
    'echo "hello world"',
])
def test_env_inspection_allowed_for_non_secret_vars(code):
    is_safe, reason = check_bash_safety(code)
    assert is_safe, f"Expected {code!r} to be allowed, got blocked: {reason}"
    assert reason == ""


# ---------------------------------------------------------------------------
# Startup hygiene: initial prompt must not direct the agent to read secrets
# or fetch the full skill guide unconditionally.
# ---------------------------------------------------------------------------

from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).parent.parent
_INITIAL_PROMPT = (_REPO_ROOT / "agent_configs" / "gsr_agent" / "initial_prompt.txt").read_text()
_GLOBAL_RULES = (_REPO_ROOT / "agent_definition" / "GLOBAL_RULES.md").read_text()
_PLATFORM_SKILLS = (_REPO_ROOT / "agent_definition" / "platform_skills.md").read_text()


def test_initial_prompt_forbids_api_key_file_read():
    lower = _INITIAL_PROMPT.lower()
    assert "do not read" in lower and ".api_key" in lower, (
        "initial_prompt.txt must instruct the agent not to read .api_key directly"
    )


def test_initial_prompt_forbids_env_var_inspection():
    lower = _INITIAL_PROMPT.lower()
    assert "do not inspect" in lower or "do not echo" in lower or "do not print" in lower, (
        "initial_prompt.txt must instruct the agent not to inspect/echo/print API key env vars"
    )


def test_initial_prompt_discourages_unconditional_skill_guide_fetch():
    assert "do not fetch" in _INITIAL_PROMPT.lower() or "only fetch" in _INITIAL_PROMPT.lower(), (
        "initial_prompt.txt must discourage fetching skill.md on every startup"
    )


def test_global_rules_discourages_unconditional_skill_guide_fetch():
    lower = _GLOBAL_RULES.lower()
    assert "do not fetch" in lower or "fallback" in lower, (
        "GLOBAL_RULES.md must not instruct unconditional skill.md fetch at startup"
    )


def test_global_rules_forbids_api_key_file_read():
    lower = _GLOBAL_RULES.lower()
    assert "do not read" in lower and ".api_key" in lower, (
        "GLOBAL_RULES.md must instruct the agent not to read .api_key directly"
    )


def test_platform_skills_discourages_unconditional_skill_guide_fetch():
    lower = _PLATFORM_SKILLS.lower()
    assert "do not fetch" in lower or "only fetch" in lower or "fallback" in lower, (
        "platform_skills.md must not instruct unconditional skill.md fetch at startup"
    )


def test_assembled_prompt_includes_local_competition_runtime_rules():
    from reva.prompt import assemble_prompt
    import os
    os.environ.setdefault("KOALA_BASE_URL", "https://koala.science")
    prompt = assemble_prompt(
        global_rules_path=_REPO_ROOT / "agent_definition" / "GLOBAL_RULES.md",
        platform_skills_path=_REPO_ROOT / "agent_definition" / "platform_skills.md",
        agent_prompt_path=_REPO_ROOT / "agent_definition" / "default_system_prompt.md",
    )
    lower = prompt.lower()
    assert "karma" in lower, "Assembled prompt must include competition runtime rules (karma)"
    assert "do not fetch" in lower or "fallback" in lower, (
        "Assembled prompt must include the no-unconditional-skill-fetch rule"
    )


# ---------------------------------------------------------------------------
# Phase 2: Verdict-first + crowded-paper strategy (Tasks A–G)
# ---------------------------------------------------------------------------

import logging
from unittest.mock import MagicMock, patch

from gsr_agent.orchestration.operational_loop import run_operational_loop

_LOOP_MOD = "gsr_agent.orchestration.operational_loop"
_LOOP_NOW = datetime(2026, 4, 26, 12, 0, 0, tzinfo=UTC)


def _loop_base_result(paper_id: str = "paper-001") -> dict:
    return {
        "paper_id": paper_id,
        "reactive_status": "none",
        "reactive_reason": None,
        "reactive_artifact": None,
        "reactive_live_posted": False,
        "reactive_live_reason": "no_candidate",
        "verdict_status": "ineligible",
        "verdict_reason": None,
        "verdict_artifact": None,
        "verdict_live_submitted": False,
        "verdict_live_reason": "no_eligible_verdict",
        "has_reactive_candidate": False,
        "reactive_draft_created": False,
        "verdict_eligible": False,
        "verdict_draft_created": False,
    }


def _verdict_paper_row(paper_id: str = "paper-verdict") -> dict:
    return {
        "paper_id": paper_id,
        "title": f"Verdict Paper {paper_id}",
        "open_time": (_LOOP_NOW - timedelta(hours=62)).isoformat(),
        "review_end_time": (_LOOP_NOW - timedelta(hours=14)).isoformat(),
        "verdict_end_time": (_LOOP_NOW + timedelta(hours=10)).isoformat(),
        "state": "VERDICT_ACTIVE",
        "pdf_url": "",
        "local_pdf_path": None,
    }


def _seed_paper_row(paper_id: str = "paper-seed") -> dict:
    return {
        "paper_id": paper_id,
        "title": f"Seed Paper {paper_id}",
        "open_time": (_LOOP_NOW - timedelta(hours=6)).isoformat(),
        "review_end_time": (_LOOP_NOW + timedelta(hours=42)).isoformat(),
        "verdict_end_time": (_LOOP_NOW + timedelta(hours=66)).isoformat(),
        "state": "REVIEW_ACTIVE",
        "pdf_url": "",
        "local_pdf_path": None,
    }


def _followup_paper_row(paper_id: str = "paper-followup") -> dict:
    return {
        "paper_id": paper_id,
        "title": f"Followup Paper {paper_id}",
        "open_time": (_LOOP_NOW - timedelta(hours=24)).isoformat(),
        "review_end_time": (_LOOP_NOW + timedelta(hours=24)).isoformat(),
        "verdict_end_time": (_LOOP_NOW + timedelta(hours=48)).isoformat(),
        "state": "REVIEW_ACTIVE",
        "pdf_url": "",
        "local_pdf_path": None,
    }


def _make_comp_db(
    paper_rows: list,
    participated_ids: set | None = None,
    comment_counts: dict | None = None,
) -> MagicMock:
    if participated_ids is None:
        participated_ids = set()
    if comment_counts is None:
        comment_counts = {}
    db = MagicMock()
    db.get_papers.return_value = paper_rows
    db.has_prior_participation.side_effect = lambda pid: pid in participated_ids
    db.get_comment_stats.side_effect = lambda pid: {
        "total": comment_counts.get(pid, 0),
        "ours": 0,
        "citable_other": comment_counts.get(pid, 0),
    }
    db.has_recent_reactive_action_for_comment.return_value = False
    db.has_recent_verdict_action_for_paper.return_value = False
    return db


def _run_comp_loop(
    paper_rows: list,
    *,
    participated_ids: set | None = None,
    comment_counts: dict | None = None,
) -> tuple[list, dict]:
    db = _make_comp_db(paper_rows, participated_ids, comment_counts)
    results = {r["paper_id"]: _loop_base_result(r["paper_id"]) for r in paper_rows}
    processed: list = []

    def _side(paper, *args, **kwargs):
        processed.append(paper.paper_id)
        return results[paper.paper_id]

    with (
        patch(f"{_LOOP_MOD}._process_paper", side_effect=_side),
        patch(f"{_LOOP_MOD}.build_run_summary", return_value=[]),
        patch(f"{_LOOP_MOD}.write_run_summary_markdown"),
        patch(f"{_LOOP_MOD}.write_run_summary_jsonl"),
    ):
        counters = run_operational_loop(db, _LOOP_NOW, output_dir="/tmp/comp_test")

    return processed, counters


# G-1: deliberating + 3 citeable outranks in_review SEED
def test_verdict_with_3_citations_outranks_seed():
    verdict_row = _verdict_paper_row("pv")
    seed_row = _seed_paper_row("ps")
    rows = [seed_row, verdict_row]  # seed listed first in DB
    processed, _ = _run_comp_loop(
        rows,
        participated_ids={"pv"},
        comment_counts={"pv": 3, "ps": 4},
    )
    assert processed[0] == "pv", "verdict paper must be processed before seed paper"


# G-2: deliberating + 2 citeable comments → insufficient_verdict_citations skip
def test_verdict_with_2_citations_skipped(caplog):
    verdict_row = _verdict_paper_row("pv2")
    rows = [verdict_row]
    with caplog.at_level(logging.INFO, logger="gsr_agent.orchestration.operational_loop"):
        _run_comp_loop(rows, participated_ids={"pv2"}, comment_counts={"pv2": 2})
    assert any(
        "insufficient_verdict_citations" in r.message and "count=2" in r.message
        for r in caplog.records
    ), "expected insufficient_verdict_citations log with count=2"


# G-3: 1–8 comment paper is preferred over 0-comment paper in SEED selection
def test_preferred_seed_beats_cold_seed():
    preferred_row = _seed_paper_row("pp")
    cold_row = _seed_paper_row("pc")
    rows = [cold_row, preferred_row]  # cold listed first in DB
    processed, _ = _run_comp_loop(
        rows,
        participated_ids=set(),
        comment_counts={"pp": 4, "pc": 0},
    )
    assert processed[0] == "pp", "preferred (4 comments) seed must be processed before cold (0)"


# G-4: >12 comments produces saturated_comments skip
def test_saturated_seed_skipped(caplog):
    saturated_row = _seed_paper_row("psat")
    rows = [saturated_row]
    with caplog.at_level(logging.INFO, logger="gsr_agent.orchestration.operational_loop"):
        processed, _ = _run_comp_loop(
            rows,
            participated_ids=set(),
            comment_counts={"psat": 13},
        )
    assert "psat" not in processed, "saturated paper must not be processed"
    assert any(
        "saturated_comments" in r.message and "comment_count=13" in r.message
        for r in caplog.records
    ), "expected saturated_comments log with comment_count=13"


# G-5: candidate inspection stops at max 3
def test_candidate_budget_stops_at_3():
    rows = [_followup_paper_row(f"pfp-{i}") for i in range(5)]
    processed, _ = _run_comp_loop(
        rows,
        participated_ids={r["paper_id"] for r in rows},
        comment_counts={},
    )
    assert len(processed) == 3, f"budget must cap at 3; got {len(processed)}"


# G-6: runtime logs include verdict_scan, selected_verdict_candidate, no_viable_verdict_candidates
def test_runtime_logs_verdict_scan_with_candidate(caplog):
    rows = [_verdict_paper_row("pvlog")]
    with caplog.at_level(logging.INFO, logger="gsr_agent.orchestration.operational_loop"):
        _run_comp_loop(rows, participated_ids={"pvlog"}, comment_counts={"pvlog": 3})
    messages = [r.message for r in caplog.records]
    assert any("verdict_scan" in m for m in messages), "expected verdict_scan log"
    assert any("selected_verdict_candidate" in m for m in messages), (
        "expected selected_verdict_candidate log"
    )


def test_runtime_logs_no_viable_verdict_candidates(caplog):
    rows = [_seed_paper_row("pslog")]
    with caplog.at_level(logging.INFO, logger="gsr_agent.orchestration.operational_loop"):
        _run_comp_loop(rows, participated_ids=set(), comment_counts={"pslog": 3})
    messages = [r.message for r in caplog.records]
    assert any("no_viable_verdict_candidates" in m for m in messages)


# G-7: MIN_VERDICT_CITATIONS is 3 in the opportunity_manager live path
def test_min_verdict_citations_is_3_in_opportunity_manager():
    from gsr_agent.strategy.opportunity_manager import MIN_VERDICT_CITATIONS
    assert MIN_VERDICT_CITATIONS == 3
    from gsr_agent.rules.verdict_eligibility import MIN_DISTINCT_OTHER_AGENTS
    assert MIN_VERDICT_CITATIONS == MIN_DISTINCT_OTHER_AGENTS
