"""The RoboEval Diagnostic Score (RDS) that ranks the challenge.

Per task, from the task's mean metrics:

    outcome  O = 0.7 * success + 0.3 * task_progression
    each behavioral metric scores 1.0 at or better than the human reference
    (median over the task's episodes in the roboeval_ee_delta_20hz_scaled
    training dataset that succeed when replayed under the challenge rules) and decays
    smoothly when worse: (r + c) / (v + c), about 0.5 at twice the reference
    family score = mean of its metric scores
    task score = 0.4 * O + O * (0.3 * safety + 0.2 * efficiency + 0.1 * coordination)

Behavior counts in proportion to outcome, so a policy that does nothing scores
0 and one that always succeeds with human-level behavior scores 1. The suite
score is the mean of the task scores.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping

OUTCOME_SPLIT = {"success": 0.7, "tp": 0.3}
WEIGHTS = {"outcome": 0.4, "safety": 0.3, "efficiency": 0.2, "coordination": 0.1}
FAMILIES = {
    "safety": ("scc", "ecc", "sc", "mjj", "mcj"),
    # ct only: at the fixed 20 Hz, tl is ct times 20 and would count time twice.
    "efficiency": ("cpl", "jpl", "opl", "ct"),
    "coordination": ("bavd", "bgvd"),
}
# Counts are often 0 for humans; a floor of one event keeps their score finite.
COUNT_METRICS = ("scc", "ecc", "sc")
RELATIVE_FLOOR = 0.1

REFERENCE_PATH = Path(__file__).with_name("human_reference.json")


@lru_cache(maxsize=1)
def human_reference() -> Dict[str, Dict[str, float]]:
    """{family: {metric: median over successful human demos}}."""
    data = json.loads(REFERENCE_PATH.read_text())
    return {fam: entry["median"] for fam, entry in data.items()}


def metric_score(key: str, value: float, reference: float) -> float:
    if value <= reference:
        return 1.0
    floor = 1.0 if key in COUNT_METRICS else 0.0
    c = max(RELATIVE_FLOOR * reference, floor, 1e-9)
    return (reference + c) / (value + c)


def task_scores(family: str, metrics: Mapping[str, float]) -> Dict[str, float]:
    """Outcome, the three behavioral family scores, and the task score."""
    ref = human_reference()[family]
    out = {"outcome": sum(w * float(metrics[k]) for k, w in OUTCOME_SPLIT.items())}
    for name, keys in FAMILIES.items():
        out[name] = sum(metric_score(k, float(metrics[k]), ref[k]) for k in keys) / len(keys)
    behavior = sum(WEIGHTS[name] * out[name] for name in FAMILIES)
    out["score"] = WEIGHTS["outcome"] * out["outcome"] + out["outcome"] * behavior
    return out


def suite_scores(per_family: Mapping[str, Mapping[str, float]]) -> Dict[str, object]:
    """Per-task scores and their mean, from results.json's summary["per_family"]."""
    tasks = {fam: task_scores(fam, m) for fam, m in per_family.items()}
    keys = ("score", "outcome", *FAMILIES)
    overall = {k: sum(t[k] for t in tasks.values()) / len(tasks) for k in keys} if tasks else {}
    return {"overall": overall, "tasks": tasks}
