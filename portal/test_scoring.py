"""Unit tests for the RoboEval Diagnostic Score (no simulator or bundle needed)."""
import pytest

from roboeval.challenge import scoring, spec

REF = {"success": 1.0, "tp": 1.0, "cpl": 2.0, "jpl": 4.0, "opl": 3.0, "ct": 8.0, "tl": 160.0,
       "bavd": 0.2, "bgvd": 0.05, "scc": 0.0, "ecc": 1.0, "sc": 0.0, "mjj": 20.0, "mcj": 5.0}


@pytest.fixture(autouse=True)
def reference(monkeypatch):
    monkeypatch.setattr(scoring, "human_reference", lambda: {f.key: dict(REF) for f in spec.TASK_FAMILIES})


def test_doing_nothing_scores_zero():
    idle = {k: 0.0 for k in REF}          # no motion, no contacts, no success
    assert scoring.task_scores("LiftPot", idle)["score"] == 0.0


def test_human_level_success_scores_one():
    assert scoring.task_scores("LiftPot", REF)["score"] == pytest.approx(1.0)


def test_worse_behavior_lowers_score_smoothly():
    m = dict(REF, cpl=4.0)                 # twice the human path length
    assert scoring.metric_score("cpl", 4.0, 2.0) == pytest.approx(2.2 / 4.2)
    assert scoring.task_scores("LiftPot", m)["efficiency"] < 1.0
    assert scoring.task_scores("LiftPot", m)["score"] < 1.0


def test_count_metric_with_zero_reference():
    assert scoring.metric_score("scc", 1.0, 0.0) == pytest.approx(0.5)
    assert scoring.metric_score("scc", 3.0, 0.0) == pytest.approx(0.25)


def test_outcome_weighting_and_suite_mean():
    half = dict(REF, success=0.5, tp=0.5)
    t = scoring.task_scores("LiftPot", half)
    assert t["outcome"] == pytest.approx(0.5)
    assert t["score"] == pytest.approx(0.4 * 0.5 + 0.5 * 0.6)
    s = scoring.suite_scores({"LiftPot": REF, "PackBox": half})
    assert s["overall"]["score"] == pytest.approx((1.0 + t["score"]) / 2)
