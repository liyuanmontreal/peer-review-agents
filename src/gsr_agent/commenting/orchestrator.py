"""Comment action orchestrator — plan_and_post_seed_comment (Phase 4A).

Ties together opportunity classification, index building, seed comment
generation, preflight, artifact publishing, and the Koala API post.

Run mode
--------
test_mode=True      — unit-test path: fake artifact URL, stub client, no env required.
KOALA_RUN_MODE=dry_run — staging path: full preparation but no real Koala write.
KOALA_RUN_MODE=live    — production path: strict artifact validation + real write.

The default KOALA_RUN_MODE is 'dry_run' (safe by default). This means that
running the orchestrator in production mode requires an explicit opt-in:

    KOALA_RUN_MODE=live KOALA_ARTIFACT_MODE=github KOALA_GITHUB_REPO=...
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from ..adapters.gsr_runner import index_paper_for_koala
from ..artifacts.github import (
    get_run_mode,
    is_test_mode_artifact_url,
    publish_comment_artifact,
    validate_artifact_for_live_action,
)
from ..koala.models import Paper
from ..rules.karma import get_action_cost
from ..rules.preflight import CommentPreflightInput, preflight_comment_action
from ..strategy.opportunity_manager import PaperOpportunity, classify_paper_opportunity
from .reactive_analysis import ReactiveAnalysisResult
from .seed_comment import choose_best_seed_comment, generate_seed_comment_candidates

if TYPE_CHECKING:
    from ..koala.client import KoalaClient
    from ..storage.db import KoalaDB

log = logging.getLogger(__name__)


def plan_and_post_seed_comment(
    paper: Paper,
    client: "KoalaClient",
    db: "KoalaDB",
    karma_remaining: float,
    now: datetime,
    *,
    test_mode: bool = False,
) -> Optional[str]:
    """Plan and post a seed comment on a paper if conditions are met.

    Steps:
      1. Classify opportunity — bail if not SEED.
      2. Index paper and generate candidates — bail if none.
      3. Choose best candidate.
      4. Publish artifact (fake URL in test_mode/dry_run; real URL in live).
      5. Run preflight checks (loose tier).
      6. Enforce run mode:
           test_mode=True → proceed with stub client.
           dry_run        → log intent and return None (no real write).
           live           → run strict artifact validation, then post.
      7. Post comment via Koala client.
      8. Log action and karma in local DB with audit metadata.

    Args:
        paper:            the paper to comment on
        client:           Koala API client
        db:               local SQLite state store
        karma_remaining:  current karma budget
        now:              current UTC datetime
        test_mode:        if True, use fake artifact URL and stub client calls

    Returns:
        The created comment ID on success, None if conditions not met or dry_run.
    """
    has_participated = db.has_prior_participation(paper.paper_id)

    opp = classify_paper_opportunity(paper, has_participated, karma_remaining, now)
    if opp != PaperOpportunity.SEED:
        log.debug(
            "[orchestrator] skipping paper=%s opportunity=%s", paper.paper_id, opp.value
        )
        return None

    index = index_paper_for_koala(paper)
    candidates = generate_seed_comment_candidates(index)
    if not candidates:
        log.info(
            "[orchestrator] no seed candidates for paper=%s (no abstract?)", paper.paper_id
        )
        return None

    body = choose_best_seed_comment(candidates)
    if not body:
        return None

    # Determine run mode and artifact publish mode.
    # test_mode=True always uses fake URL. dry_run also uses fake URL so that
    # github credentials are not required for staging runs.
    run_mode = get_run_mode() if not test_mode else "test"
    artifact_test_mode = test_mode or (run_mode != "live")

    github_file_url = publish_comment_artifact(
        paper.paper_id, body, test_mode=artifact_test_mode
    )

    # Loose preflight (phase, karma, body, placeholder URL).
    preflight_comment_action(
        CommentPreflightInput(
            paper_id=paper.paper_id,
            body=body,
            github_file_url=github_file_url,
            open_time=paper.open_time,
            now=now,
            karma_remaining=karma_remaining,
            has_prior_participation=has_participated,
        )
    )

    artifact_mode_env = os.environ.get("KOALA_ARTIFACT_MODE", "local")

    # Dry-run gate: log intent and return without writing to Koala.
    if not test_mode and run_mode != "live":
        log.info(
            "[orchestrator] dry_run: would post seed comment for paper=%s "
            "run_mode=%r github_url=%s",
            paper.paper_id, run_mode, github_file_url,
        )
        db.log_action(
            paper_id=paper.paper_id,
            action_type="seed_comment",
            github_file_url=github_file_url,
            status="dry_run",
            details={
                "run_mode": run_mode,
                "artifact_mode": artifact_mode_env,
                "artifact_url": github_file_url,
                "is_test_url": is_test_mode_artifact_url(github_file_url),
                "publish_status": "dry_run",
                "blocked_reason": f"KOALA_RUN_MODE={run_mode!r}",
            },
        )
        return None

    # Live gate: strict artifact validation before any external write.
    if not test_mode:
        api_base_url = os.environ.get("KOALA_API_BASE_URL", "")
        if not api_base_url:
            from ..koala.errors import KoalaPreflightError
            raise KoalaPreflightError(
                "plan_and_post_seed_comment: KOALA_API_BASE_URL must be set in live mode."
            )
        validate_artifact_for_live_action(github_file_url)

    comment_id = client.post_comment(paper.paper_id, body, github_file_url)

    cost = get_action_cost("comment", has_prior_participation=False)
    db.log_action(
        paper_id=paper.paper_id,
        action_type="seed_comment",
        external_id=comment_id,
        github_file_url=github_file_url,
        status="success",
        details={
            "run_mode": run_mode,
            "artifact_mode": artifact_mode_env,
            "artifact_url": github_file_url,
            "is_test_url": is_test_mode_artifact_url(github_file_url),
            "publish_status": "published",
        },
    )
    db.record_karma(
        paper_id=paper.paper_id,
        action_type="seed_comment",
        cost=cost,
        karma_before=karma_remaining,
        karma_after=karma_remaining - cost,
    )

    log.info(
        "[orchestrator] posted seed comment id=%s on paper=%s karma_cost=%.1f run_mode=%s",
        comment_id, paper.paper_id, cost, run_mode,
    )
    return comment_id


# ---------------------------------------------------------------------------
# Phase 5B: reactive comment execution
# ---------------------------------------------------------------------------

_DRY_RUN_HEADER = "[DRY-RUN — not posted]"


def _prepare_reactive_body(draft_text: Optional[str]) -> Optional[str]:
    """Strip the Phase 5A DRY-RUN header from draft_text for actual posting."""
    if not draft_text:
        return None
    lines = draft_text.splitlines()
    if lines and lines[0].strip() == _DRY_RUN_HEADER:
        lines = lines[1:]
    body = "\n".join(lines).strip()
    return body if body else None


def plan_and_post_reactive_comment(
    paper: Paper,
    candidate: ReactiveAnalysisResult,
    client: "KoalaClient",
    db: "KoalaDB",
    karma_remaining: float,
    now: datetime,
    *,
    test_mode: bool = False,
) -> Optional[str]:
    """Plan and post a reactive comment based on a Phase 5A analysis result.

    Default is dry-run: prepares everything but does not call the Koala API
    unless KOALA_RUN_MODE=live or test_mode=True.

    Steps:
      1. Extract and clean the draft body (strip DRY-RUN header).
      2. Publish artifact (fake URL in non-live modes).
      3. Run preflight checks (loose tier).
      4. Dry-run gate: log intent and return None unless live/test_mode.
      5. Live gate: strict artifact validation.
      6. Post comment via Koala client.
      7. Log action and karma in local DB.

    Args:
        paper:           the paper containing the target comment
        candidate:       Phase 5A result with recommendation "react"
        client:          Koala API client
        db:              local SQLite state store
        karma_remaining: current karma budget
        now:             current UTC datetime
        test_mode:       if True, use fake artifact URL and stub client calls

    Returns:
        The created comment ID on success, None if dry_run or no body.
    """
    body = _prepare_reactive_body(candidate.draft_text)
    if not body:
        log.info(
            "[orchestrator] reactive: no postable draft text for comment=%s paper=%s",
            candidate.comment_id, paper.paper_id,
        )
        return None

    has_participated = db.has_prior_participation(paper.paper_id)
    run_mode = get_run_mode() if not test_mode else "test"
    artifact_test_mode = test_mode or (run_mode != "live")

    github_file_url = publish_comment_artifact(
        paper.paper_id, body, test_mode=artifact_test_mode
    )

    preflight_comment_action(
        CommentPreflightInput(
            paper_id=paper.paper_id,
            body=body,
            github_file_url=github_file_url,
            open_time=paper.open_time,
            now=now,
            karma_remaining=karma_remaining,
            has_prior_participation=has_participated,
        )
    )

    artifact_mode_env = os.environ.get("KOALA_ARTIFACT_MODE", "local")

    if not test_mode and run_mode != "live":
        log.info(
            "[orchestrator] dry_run: would post reactive comment for paper=%s "
            "source_comment=%s run_mode=%r github_url=%s",
            paper.paper_id, candidate.comment_id, run_mode, github_file_url,
        )
        db.log_action(
            paper_id=paper.paper_id,
            action_type="reactive_comment",
            github_file_url=github_file_url,
            status="dry_run",
            details={
                "run_mode": run_mode,
                "artifact_mode": artifact_mode_env,
                "source_comment_id": candidate.comment_id,
                "recommendation": candidate.recommendation,
                "artifact_url": github_file_url,
                "is_test_url": is_test_mode_artifact_url(github_file_url),
                "publish_status": "dry_run",
                "blocked_reason": f"KOALA_RUN_MODE={run_mode!r}",
            },
        )
        return None

    if not test_mode:
        api_base_url = os.environ.get("KOALA_API_BASE_URL", "")
        if not api_base_url:
            from ..koala.errors import KoalaPreflightError
            raise KoalaPreflightError(
                "plan_and_post_reactive_comment: KOALA_API_BASE_URL must be set in live mode."
            )
        validate_artifact_for_live_action(github_file_url)

    comment_id = client.post_comment(
        paper.paper_id,
        body,
        github_file_url,
        thread_id=candidate.thread_id,
        parent_id=candidate.comment_id,
    )

    cost = get_action_cost("comment", has_prior_participation=has_participated)
    db.log_action(
        paper_id=paper.paper_id,
        action_type="reactive_comment",
        external_id=comment_id,
        github_file_url=github_file_url,
        status="success",
        details={
            "run_mode": run_mode,
            "artifact_mode": artifact_mode_env,
            "source_comment_id": candidate.comment_id,
            "recommendation": candidate.recommendation,
            "artifact_url": github_file_url,
            "is_test_url": is_test_mode_artifact_url(github_file_url),
            "publish_status": "published",
        },
    )
    db.record_karma(
        paper_id=paper.paper_id,
        action_type="reactive_comment",
        cost=cost,
        karma_before=karma_remaining,
        karma_after=karma_remaining - cost,
    )

    log.info(
        "[orchestrator] posted reactive comment id=%s on paper=%s source=%s "
        "karma_cost=%.2f run_mode=%s",
        comment_id, paper.paper_id, candidate.comment_id, cost, run_mode,
    )
    return comment_id
