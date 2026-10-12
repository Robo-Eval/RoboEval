"""Fixed definition of the challenge: tasks, budgets, seeds, and metric keys.

Everything a submission is scored against lives here so that the evaluator,
the validator, and the website can be checked against a single source.
Values marked TBC are placeholders until the challenge launch announcement.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Type

CHALLENGE_RELEASE = "corl2026-challenge"
HARNESS_VERSION = "1.0.0"

# TBC: confirmed at launch. Seeds control initial object poses (``env.reset``
# seeds NumPy's global RNG before the task samples its spawn poses).
EPISODES_PER_VARIATION = 50
EVAL_SEEDS: Tuple[int, ...] = tuple(range(1000, 1000 + EPISODES_PER_VARIATION))

# Hex digest of ``provenance.source_fingerprint`` for the tagged release.
# None until the tag is cut; the validator then only warns on a mismatch.
OFFICIAL_SOURCE_FINGERPRINT = None

DEFAULT_CONTROL_FREQUENCY = 20

# Cameras a submission may request (see roboeval/robots/configs/panda.py).
AVAILABLE_CAMERAS = ("external", "head", "front", "left_wrist", "right_wrist")

# Only each family's combined position + orientation variant is evaluated, to
# keep evaluation cost down; it is the hardest spatial setting of every family.
# Demonstrations from all variants may still be used for training.
VARIATIONS = ("PositionAndOrientation",)


@dataclass(frozen=True)
class TaskFamily:
    """One task family and the environment class behind each variation."""

    key: str
    display_name: str
    instruction_file: str
    budget_seconds: float
    variations: Dict[str, str]


# Budgets are simulated seconds, so they hold for any declared control
# frequency. Proposed values: ~1.5x the slowest variation's 95th-percentile
# human demonstration duration (500 Hz demo store), rounded up to 5 s. TBC.
TASK_FAMILIES: Tuple[TaskFamily, ...] = (
    TaskFamily(
        key="LiftPot",
        display_name="Lift Pot",
        instruction_file="lift_pot.json",
        budget_seconds=10.0,
        variations={
            "PositionAndOrientation": "roboeval.envs.lift_pot:LiftPotPositionAndOrientation",
        },
    ),
    TaskFamily(
        key="LiftTray",
        display_name="Lift Tray",
        instruction_file="lift_tray.json",
        budget_seconds=10.0,
        variations={
            "PositionAndOrientation": "roboeval.envs.lift_tray:LiftTrayPositionAndOrientation",
        },
    ),
    TaskFamily(
        key="StackTwoBlocks",
        display_name="Stack Two Cubes",
        instruction_file="stack_two_blocks.json",
        budget_seconds=20.0,
        variations={
            "PositionAndOrientation": "roboeval.envs.manipulation:StackTwoBlocksPositionAndOrientation",
        },
    ),
    TaskFamily(
        key="StackSingleBookShelf",
        display_name="Stack Single Book Shelf",
        instruction_file="stack_single_book_shelf.json",
        budget_seconds=40.0,
        variations={
            "PositionAndOrientation": "roboeval.envs.stack_books:StackSingleBookShelfPositionAndOrientation",
        },
    ),
    TaskFamily(
        key="CubeHandover",
        display_name="Cube Handover",
        instruction_file="cube_handover.json",
        budget_seconds=20.0,
        variations={
            "PositionAndOrientation": "roboeval.envs.manipulation:CubeHandoverPositionAndOrientation",
        },
    ),
    TaskFamily(
        key="PackBox",
        display_name="Pack Box",
        instruction_file="pack_box.json",
        budget_seconds=20.0,
        variations={
            "PositionAndOrientation": "roboeval.envs.pack_objects:PackBoxPositionAndOrientation",
        },
    ),
    TaskFamily(
        key="PickSingleBookFromTable",
        display_name="Pick Book From Table",
        instruction_file="pick_single_book_from_table.json",
        budget_seconds=15.0,
        variations={
            "PositionAndOrientation": "roboeval.envs.stack_books:PickSingleBookFromTablePositionAndOrientation",
        },
    ),
    TaskFamily(
        key="RotateValve",
        display_name="Rotate Valve",
        instruction_file="rotate_valve.json",
        budget_seconds=20.0,
        variations={
            "PositionAndOrientation": "roboeval.envs.rotate_utility_objects:RotateValvePositionAndOrientation",
        },
    ),
)

FAMILIES_BY_KEY: Dict[str, TaskFamily] = {f.key: f for f in TASK_FAMILIES}


def all_variations() -> List[Tuple[str, str]]:
    """Every (family, variation) pair a complete submission must cover."""
    return [(f.key, v) for f in TASK_FAMILIES for v in f.variations]


def variation_dirname(family: str, variation: str) -> str:
    return f"{family}_{variation}"


def max_steps(family: str, control_frequency: int) -> int:
    return int(round(FAMILIES_BY_KEY[family].budget_seconds * control_frequency))


def env_class(family: str, variation: str) -> Type:
    """Import the environment class lazily so schema checks need no simulator."""
    module_name, class_name = FAMILIES_BY_KEY[family].variations[variation].split(":")
    return getattr(importlib.import_module(module_name), class_name)


# ─────────────────────────────── metrics ────────────────────────────────
# Leaderboard key -> raw key in the episode's final ``info`` dict, as written
# by roboeval_metrics.MetricEvaluator. ``ct`` and ``tl`` are derived from the
# step count instead: the evaluator's ``completion_time`` is host wall-clock
# time, which would make scores depend on hardware.
RAW_METRIC_KEYS: Dict[str, str] = {
    "success": "success",
    "tp": "subtask_progress",
    "cpl": "total_cartesian_path_length",
    "jpl": "total_joint_path_length",
    "opl": "total_orientation_path_length",
    "bavd": "bimanual_arm_velocity_difference",
    "bgvd": "bimanual_gripper_vertical_difference",
    "scc": "self_collision_count",
    "ecc": "env_collision_count",
    "sc": "slip_count",
    "mjj": "overall_avg_joint_jerk",
    "mcj": "overall_avg_cartesian_jerk",
}

# Order matches the website's static/data/leaderboard_schema.json.
METRIC_KEYS: Tuple[str, ...] = (
    "success", "tp", "bavd", "bgvd", "cpl", "ct", "ecc",
    "jpl", "mcj", "mjj", "opl", "scc", "sc", "tl",
)

TERMINATIONS = ("success", "fail", "budget", "unhealthy", "error")
