#!/usr/bin/env python3
"""Replay a fixed number of demos per task and assert success after the full trajectory.

Runs one or more **check presets** (default: both used in CI):

1. ``joint_abs_500`` — joint absolute @ 500 Hz (native demo rate).
2. ``ee_delta_20`` — demos converted to EE delta, replayed @ 20 Hz.

Each preset loads **lightweight** joint-absolute demos from DemoStore at the preset
frequency (same cache layout as the public release; no ``State``/``Pixel`` re-bake),
replays every timestep in the sim (no early exit), then checks ``env.success`` on the
last step.

Use ``--clamp-feasible`` to apply ``DemoConverter.clamp_absolute_joint_velocity`` at
the preset frequency and replay with ``enforce_joint_velocity_limits=True`` (matches
slew-limited sim). Without it, replay uses open-loop targets (``enforce_joint_velocity_limits=False``).

Usage:
    python examples/replay_sample_check_success.py
    python examples/replay_sample_check_success.py --amount 3 --checks joint_abs_500
    python examples/replay_sample_check_success.py --checks ee_delta_20 --tasks LiftPot
    python examples/replay_sample_check_success.py --clamp-feasible --checks joint_abs_500

CI: ``.github/workflows/demo-replay.yml`` runs ``--amount 2 --seed 0`` (both presets).
MuJoCo demo vs. install version warnings are suppressed by default; set
``ROBOEVAL_SKIP_DEMO_VERSION_CHECK=0`` to see them.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np

from roboeval.action_modes import JointPositionActionMode
from roboeval.robots.configs.panda import BimanualPanda
from roboeval.demonstrations.demo_player import DemoPlayer
from roboeval.demonstrations.demo_store import DemoStore, DemoNotFoundError, TooManyDemosRequestedError
from roboeval.demonstrations.demo_converter import DemoConverter
from roboeval.demonstrations.utils import Metadata, ObservationMode

from roboeval.envs.lift_pot import (
    LiftPot,
    LiftPotPosition,
    LiftPotOrientation,
    LiftPotPositionAndOrientation,
)
from roboeval.envs.manipulation import (
    CubeHandover,
    CubeHandoverPosition,
    CubeHandoverOrientation,
    CubeHandoverPositionAndOrientation,
    VerticalCubeHandover,
    StackTwoBlocks,
    StackTwoBlocksPosition,
    StackTwoBlocksOrientation,
    StackTwoBlocksPositionAndOrientation,
)
from roboeval.envs.stack_books import (
    StackSingleBookShelf,
    StackSingleBookShelfPosition,
    StackSingleBookShelfPositionAndOrientation,
    PickSingleBookFromTable,
    PickSingleBookFromTablePosition,
    PickSingleBookFromTableOrientation,
    PickSingleBookFromTablePositionAndOrientation,
)
from roboeval.envs.pack_objects import (
    PackBox,
    PackBoxOrientation,
    PackBoxPosition,
    PackBoxPositionAndOrientation,
)
from roboeval.envs.lift_tray import (
    LiftTray,
    LiftTrayPosition,
    LiftTrayOrientation,
    LiftTrayPositionAndOrientation,
    DragOverAndLiftTray,
)
from roboeval.envs.rotate_utility_objects import (
    RotateValve,
    RotateValvePosition,
    RotateValvePositionAndOrientation,
    RotateValveObstacle,
)

TASKS: dict[str, type] = {
    "LiftPot": LiftPot,
    "LiftPotPosition": LiftPotPosition,
    "LiftPotOrientation": LiftPotOrientation,
    "LiftPotPositionAndOrientation": LiftPotPositionAndOrientation,
    "CubeHandover": CubeHandover,
    "CubeHandoverPosition": CubeHandoverPosition,
    "CubeHandoverOrientation": CubeHandoverOrientation,
    "CubeHandoverPositionAndOrientation": CubeHandoverPositionAndOrientation,
    "VerticalCubeHandover": VerticalCubeHandover,
    "StackTwoBlocks": StackTwoBlocks,
    "StackTwoBlocksPosition": StackTwoBlocksPosition,
    "StackTwoBlocksOrientation": StackTwoBlocksOrientation,
    "StackTwoBlocksPositionAndOrientation": StackTwoBlocksPositionAndOrientation,
    "StackSingleBookShelf": StackSingleBookShelf,
    "StackSingleBookShelfPosition": StackSingleBookShelfPosition,
    "StackSingleBookShelfPositionAndOrientation": StackSingleBookShelfPositionAndOrientation,
    "PickSingleBookFromTable": PickSingleBookFromTable,
    "PickSingleBookFromTablePosition": PickSingleBookFromTablePosition,
    "PickSingleBookFromTableOrientation": PickSingleBookFromTableOrientation,
    "PickSingleBookFromTablePositionAndOrientation": PickSingleBookFromTablePositionAndOrientation,
    "PackBox": PackBox,
    "PackBoxOrientation": PackBoxOrientation,
    "PackBoxPosition": PackBoxPosition,
    "PackBoxPositionAndOrientation": PackBoxPositionAndOrientation,
    "LiftTray": LiftTray,
    "LiftTrayPosition": LiftTrayPosition,
    "LiftTrayOrientation": LiftTrayOrientation,
    "LiftTrayPositionAndOrientation": LiftTrayPositionAndOrientation,
    "DragOverAndLiftTray": DragOverAndLiftTray,
    "RotateValve": RotateValve,
    "RotateValvePosition": RotateValvePosition,
    "RotateValvePositionAndOrientation": RotateValvePositionAndOrientation,
    "RotateValveObstacle": RotateValveObstacle,
}

# preset_id -> (ee, absolute, freq_hz, description)
CHECK_PRESETS: dict[str, Tuple[bool, bool, int, str]] = {
    "joint_abs_500": (False, True, 500, "joint absolute @ 500 Hz"),
    "ee_delta_20": (True, False, 20, "EE delta @ 20 Hz"),
}


def replay_to_end_then_success(demo, env) -> bool:
    """Replay all timesteps, then return whether the env reports success."""
    timesteps = DemoPlayer._get_timesteps_for_replay(
        demo, env, demo_frequency=env.control_frequency
    )
    env.reset(seed=demo.seed)
    for step in timesteps:
        env.step(step.executed_action, fast=True)
    return bool(env.success)


def load_demos(demo_store: DemoStore, metadata: Metadata, amount: int, frequency: int):
    """Load up to ``amount`` demos; if fewer exist, load all without error."""
    try:
        return demo_store.get_demos(metadata, amount=amount, frequency=frequency)
    except TooManyDemosRequestedError as e:
        if e.found <= 0:
            return []
        return demo_store.get_demos(metadata, amount=e.found, frequency=frequency)


def _target_action_mode(ee: bool, absolute: bool, *, enforce_limits: bool) -> JointPositionActionMode:
    return JointPositionActionMode(
        floating_base=True, absolute=absolute, ee=ee, floating_dofs=[],
        enforce_joint_velocity_limits=enforce_limits,
    )


def _joint_light_metadata(env_cls: type) -> Metadata:
    """DemoStore metadata matching released lightweight joint-absolute demos.

    Use ``floating_dofs=[]`` so ``action_mode_description`` matches the public
    demo zip (e.g. ``JointPositionActionMode_floating_absolute_joint``). Passing
    ``None`` would expand to ``DEFAULT_DOFS`` and a longer path with
    ``pelvis_x_...`` segments, which does not match those releases.
    """
    return Metadata.from_env_cls(
        env_cls=env_cls,
        action_mode=JointPositionActionMode,
        floating_dofs=[],
        obs_mode=ObservationMode.Lightweight,
        action_mode_absolute=True,
        end_effector_mode=False,
    )


def run_preset(
    preset_id: str,
    tasks: dict[str, type],
    demo_store: DemoStore,
    amount: int,
    fail_fast: bool,
    clamp_feasible: bool,
) -> tuple[int, int, bool]:
    """Run one check preset over all tasks. Returns (total_ok, total_ran, any_fail)."""
    ee, absolute, freq, desc = CHECK_PRESETS[preset_id]
    enforce_limits = bool(clamp_feasible)
    total_ok = 0
    total_ran = 0
    any_fail = False

    print(f"\n{'=' * 70}\nPreset: {preset_id} — {desc}\n{'=' * 70}\n")
    if clamp_feasible:
        print("(clamp-feasible: joint slew preprocessed; enforce_joint_velocity_limits=True)\n")

    for name, env_cls in tasks.items():
        try:
            load_metadata = _joint_light_metadata(env_cls)
        except Exception as e:
            print(f"{name}: [SKIP] could not build lightweight demo metadata: {e}")
            continue

        try:
            env = env_cls(
                action_mode=_target_action_mode(ee, absolute, enforce_limits=enforce_limits),
                render_mode=None,
                control_frequency=freq,
                robot_cls=BimanualPanda,
            )
        except Exception as e:
            print(f"{name}: [SKIP] could not create env: {e}")
            continue

        try:
            demos = load_demos(demo_store, load_metadata, amount, freq)
        except DemoNotFoundError:
            print(f"{name}: [SKIP] no demos in DemoStore")
            env.close()
            continue
        except Exception as e:
            print(f"{name}: [SKIP] demo load error: {e}")
            env.close()
            continue

        if not demos:
            print(f"{name}: [SKIP] 0 demos")
            env.close()
            continue

        n_fail = 0
        failed_ids: list[str] = []
        for demo in demos:
            total_ran += 1
            replay_demo = demo
            if clamp_feasible:
                try:
                    replay_demo = DemoConverter.clamp_absolute_joint_velocity(
                        replay_demo, control_frequency_hz=freq
                    )
                except Exception as e:
                    ok = False
                    failed_ids.append(f"{demo.uuid} (clamp: {e})")
                    n_fail += 1
                    any_fail = True
                    if fail_fast:
                        print(f"{name}: FAIL {demo.uuid} — clamp: {e}")
                        env.close()
                        return total_ok, total_ran, True
                    continue
            if ee and not absolute:
                try:
                    replay_demo = DemoConverter.joint_absolute_to_ee_delta(replay_demo)
                except Exception as e:
                    ok = False
                    failed_ids.append(f"{demo.uuid} (conversion: {e})")
                    n_fail += 1
                    any_fail = True
                    if fail_fast:
                        print(f"{name}: FAIL {demo.uuid} — conversion: {e}")
                        env.close()
                        return total_ok, total_ran, True
                    continue

            try:
                ok = replay_to_end_then_success(replay_demo, env)
            except Exception as e:
                ok = False
                failed_ids.append(f"{demo.uuid} (exception: {e})")
                n_fail += 1
                any_fail = True
                if fail_fast:
                    print(f"{name}: FAIL {demo.uuid} — {e}")
                    env.close()
                    return total_ok, total_ran, True
                continue

            if ok:
                total_ok += 1
            else:
                n_fail += 1
                failed_ids.append(demo.uuid)
                any_fail = True
                if fail_fast:
                    print(f"{name}: FAIL {demo.uuid} — env.success is False after full replay")
                    env.close()
                    return total_ok, total_ran, True

        print(
            f"{name}: {len(demos)} demo(s)  "
            f"pass {len(demos) - n_fail}/{len(demos)}  "
            f"({100.0 * (len(demos) - n_fail) / len(demos):.0f}%)"
        )
        if failed_ids:
            for uid in failed_ids:
                print(f"    - {uid}")

        env.close()

    return total_ok, total_ran, any_fail


def main() -> int:
    default_checks = tuple(CHECK_PRESETS.keys())
    parser = argparse.ArgumentParser(
        description="Replay N demos per task; check env.success after full replay.",
    )
    parser.add_argument("--amount", type=int, default=5, help="Max demos per task (default: 5)")
    parser.add_argument(
        "--checks",
        nargs="+",
        choices=list(CHECK_PRESETS.keys()),
        default=list(default_checks),
        metavar="PRESET",
        help=(
            "Which replay suites to run (default: %(default)s). "
            "joint_abs_500 = joint absolute 500 Hz; ee_delta_20 = EE delta 20 Hz."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for DemoStore demo subsampling (default: 0)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="*",
        default=None,
        help="Task class names to run (default: all)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failing demo",
    )
    parser.add_argument(
        "--clamp-feasible",
        action="store_true",
        help=(
            "Clamp limb joint targets with DemoConverter.clamp_absolute_joint_velocity "
            "at the preset Hz, then replay with enforce_joint_velocity_limits=True"
        ),
    )
    args = parser.parse_args()
    # Demos record MuJoCo 3.1.5; many dev installs use 3.3.x. Skip per-demo warnings
    # unless the user set ROBOEVAL_SKIP_DEMO_VERSION_CHECK explicitly (e.g. to "0").
    os.environ.setdefault("ROBOEVAL_SKIP_DEMO_VERSION_CHECK", "1")
    np.random.seed(args.seed)

    tasks: dict[str, type] = (
        {k: v for k, v in TASKS.items() if k in args.tasks}
        if args.tasks
        else dict(TASKS)
    )
    if args.tasks:
        unknown = set(args.tasks) - set(TASKS)
        if unknown:
            print(f"Warning: unknown task names ignored: {sorted(unknown)}")
        if not tasks:
            print("No valid tasks. Available:", ", ".join(sorted(TASKS)))
            return 1

    demo_store = DemoStore()
    demo_store.pull_demos()

    print(
        f"Up to {args.amount} demo(s) per task × {len(args.checks)} preset(s): "
        f"{', '.join(args.checks)}"
    )

    grand_ok = 0
    grand_ran = 0
    any_fail = False

    for preset_id in args.checks:
        ok, ran, failed = run_preset(
            preset_id,
            tasks,
            demo_store,
            args.amount,
            args.fail_fast,
            args.clamp_feasible,
        )
        grand_ok += ok
        grand_ran += ran
        any_fail = any_fail or failed

    print()
    if grand_ran == 0:
        print("No demos replayed (cache empty or all tasks skipped).")
        return 1
    print(
        f"Overall (all presets): {grand_ok}/{grand_ran} demos passed "
        f"({100.0 * grand_ok / grand_ran:.1f}%)"
    )
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
