"""Tests for the analytics layer.

The emerging-issue detector is validated against the trends deliberately
planted in ticket_history.csv. See data/tickets/PLANTED_TRENDS.md.
"""
import pandas as pd

from src.analytics.metrics import (
    confidence_calibration,
    daily_volume,
    emerging_issues,
    escalation_analysis,
    intent_distribution,
    overview_dict,
    refund_exposure,
)

import pytest

pytestmark = pytest.mark.data


def test_overview_returns_sane_metrics():
    o = overview_dict()
    assert o["total_tickets"] > 10_000
    assert 0 < o["deflection_rate_pct"] < 100
    assert abs(o["deflection_rate_pct"] + o["escalation_rate_pct"] - 100) < 0.5
    assert o["avg_latency_s"] > 0


def test_intent_distribution_covers_all_intents():
    df = intent_distribution()
    assert len(df) == 11
    assert abs(df["share_pct"].sum() - 100) < 1.0
    assert df["tickets"].is_monotonic_decreasing


def test_detector_finds_planted_login_spike():
    """T1: account_management / login failure, planted at ~4.5x."""
    df = emerging_issues()
    hit = df[
        (df["intent"] == "account_management")
        & (df["subtopic"] == "login failure")
    ]
    assert not hit.empty, "planted trend T1 not detected"
    assert hit.iloc[0]["signal"] == "SPIKE"
    assert hit.iloc[0]["lift"] > 2.5


def test_detector_finds_planted_display_ramp():
    """T2: technical_support / display issue, a gradual ramp not a step."""
    df = emerging_issues()
    hit = df[
        (df["intent"] == "technical_support")
        & (df["subtopic"] == "display issue")
    ]
    assert not hit.empty, "planted trend T2 not detected"
    assert hit.iloc[0]["signal"] in ("SPIKE", "ELEVATED")


def test_login_spike_ranks_above_display_ramp():
    """T1 is sharper than T2 and should rank higher."""
    df = emerging_issues()
    top = df.iloc[0]
    assert top["intent"] == "account_management"
    assert top["subtopic"] == "login failure"


def test_detector_suppresses_low_volume_noise():
    df = emerging_issues(min_recent=5)
    assert (df["recent_n"] >= 5).all()


def test_daily_volume_has_moving_average():
    df = daily_volume(days=30)
    assert len(df) == 30
    assert "ma_7d" in df.columns
    assert df["ma_7d"].notna().all()
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_escalation_rises_with_negative_sentiment():
    """Sanity check on the simulated data: angry customers escalate more."""
    df = escalation_analysis()
    neg = df[df["sentiment"] == "negative"]["escalation_pct"].mean()
    pos = df[df["sentiment"] == "positive"]["escalation_pct"].mean()
    assert neg > pos


def test_confidence_calibration_produces_buckets():
    df = confidence_calibration()
    assert len(df) > 3
    assert df["n"].sum() > 100
    assert (df["down_rate_pct"] >= 0).all()


def test_refund_exposure_is_ranked():
    df = refund_exposure()
    assert not df.empty
    assert df["exposure"].is_monotonic_decreasing
    assert df.iloc[0]["exposure_rank"] == 1
