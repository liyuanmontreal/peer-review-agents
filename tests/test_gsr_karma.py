"""Tests for gsr_agent.rules.karma — karma budget engine."""

import pytest

from gsr_agent.rules.karma import (
    DEFAULT_RESERVE_FLOOR,
    FIRST_ACTION_COST,
    FOLLOWUP_ACTION_COST,
    INITIAL_KARMA,
    VERDICT_COST,
    can_afford,
    get_action_cost,
    should_block_new_paper_entry,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_initial_karma_is_100():
    assert INITIAL_KARMA == 100.0


def test_first_action_cost_is_1():
    assert FIRST_ACTION_COST == 1.0


def test_followup_action_cost_is_point_1():
    assert FOLLOWUP_ACTION_COST == 0.1


def test_verdict_cost_is_zero():
    assert VERDICT_COST == 0.0


def test_default_reserve_floor_is_15():
    assert DEFAULT_RESERVE_FLOOR == 15.0


# ---------------------------------------------------------------------------
# get_action_cost
# ---------------------------------------------------------------------------

def test_first_comment_costs_1():
    assert get_action_cost("comment", has_prior_participation=False) == 1.0


def test_followup_comment_costs_point_1():
    assert get_action_cost("comment", has_prior_participation=True) == 0.1


def test_first_thread_costs_1():
    assert get_action_cost("thread", has_prior_participation=False) == 1.0


def test_followup_thread_costs_point_1():
    assert get_action_cost("thread", has_prior_participation=True) == 0.1


def test_verdict_costs_zero_regardless_of_participation():
    assert get_action_cost("verdict", has_prior_participation=False) == 0.0
    assert get_action_cost("verdict", has_prior_participation=True) == 0.0


# ---------------------------------------------------------------------------
# can_afford
# ---------------------------------------------------------------------------

def test_can_afford_exact_amount():
    assert can_afford(1.0, 1.0) is True


def test_can_afford_surplus():
    assert can_afford(10.0, 1.0) is True


def test_cannot_afford_below():
    assert can_afford(0.5, 1.0) is False


def test_cannot_afford_zero_karma():
    assert can_afford(0.0, 0.1) is False


def test_can_afford_followup_with_small_balance():
    assert can_afford(0.1, 0.1) is True


def test_cannot_afford_followup_with_tiny_balance():
    assert can_afford(0.09, 0.1) is False


def test_can_afford_verdict_with_zero_karma():
    # Verdicts cost 0.0, so even 0 karma is enough.
    assert can_afford(0.0, 0.0) is True


# ---------------------------------------------------------------------------
# should_block_new_paper_entry
# ---------------------------------------------------------------------------

def test_blocks_when_below_floor():
    assert should_block_new_paper_entry(10.0) is True


def test_blocks_when_just_below_floor():
    assert should_block_new_paper_entry(14.99) is True


def test_allows_when_at_floor():
    assert should_block_new_paper_entry(15.0) is False


def test_allows_when_above_floor():
    assert should_block_new_paper_entry(50.0) is False


def test_allows_at_full_karma():
    assert should_block_new_paper_entry(100.0) is False


def test_custom_reserve_floor_blocks():
    assert should_block_new_paper_entry(20.0, reserve_floor=25.0) is True


def test_custom_reserve_floor_allows():
    assert should_block_new_paper_entry(25.0, reserve_floor=25.0) is False


def test_blocks_at_zero_karma():
    assert should_block_new_paper_entry(0.0) is True
