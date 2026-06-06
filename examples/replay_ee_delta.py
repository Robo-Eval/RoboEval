#!/usr/bin/env python3
"""Replay demonstrations converted to delta end-effector mode and report success rates.

This verifies that the joint-absolute → delta-EE conversion (rotvec-based)
produces actions that successfully replay in the simulator.
"""

import sys
from pathlib import Path

import numpy as np

from roboeval.action_modes import JointPositionActionMode
from roboeval.robots.configs.panda import BimanualPanda
from roboeval.demonstrations.demo_converter import DemoConverter
from roboeval.demonstrations.demo_store import DemoStore, DemoNotFoundError
from roboeval.demonstrations.utils import Metadata
from roboeval.roboeval_env import CONTROL_FREQUENCY_MAX

# All environment classes
from roboeval.envs.lift_pot import (
    LiftPot, LiftPotPosition, LiftPotOrientation, LiftPotPositionAndOrientation,
)
from roboeval.envs.manipulation import (
    CubeHandover, CubeHandoverPosition, CubeHandoverOrientation,
    CubeHandoverPositionAndOrientation, VerticalCubeHandover,
    StackTwoBlocks, StackTwoBlocksPosition, StackTwoBlocksOrientation,
)
from roboeval.envs.stack_books import (
    StackSingleBookShelf, StackSingleBookShelfPosition,
    StackSingleBookShelfPositionAndOrientation,
    PickSingleBookFromTable, PickSingleBookFromTablePosition,
    PickSingleBookFromTableOrientation, PickSingleBookFromTablePositionAndOrientation,
)
from roboeval.envs.pack_objects import (
    PackBox, PackBoxOrientation, PackBoxPosition, PackBoxPositionAndOrientation,
)
from roboeval.envs.lift_tray import (
    LiftTray, LiftTrayPosition, LiftTrayOrientation,
    LiftTrayPositionAndOrientation, DragOverAndLiftTray,
)
from roboeval.envs.rotate_utility_objects import (
    RotateValve, RotateValvePosition, RotateValvePositionAndOrientation,
    RotateValveObstacle,
)

ENVIRONMENTS = {
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

CONTROL_FREQ = 20


def replay_ee_delta_check_success(demo, env):
    """Convert a joint-absolute demo to delta-EE and replay it, returning success."""
    new_demo = DemoConverter.create_demo_in_new_env(demo, env)
    # create_demo_in_new_env already steps through the env;
    # check if any timestep achieved reward > 0
    return any(ts.reward > 0 for ts in new_demo.timesteps)


def main():
    # --- Parse CLI args ---
    # Usage: python replay_ee_delta.py [ENV_NAME] [--max-demos N]
    env_filter = None
    max_demos = -1  # -1 = all
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--max-demos" and i + 1 < len(args):
            max_demos = int(args[i + 1])
            i += 2
        else:
            env_filter = args[i]
            i += 1

    if env_filter:
        envs_to_test = {k: v for k, v in ENVIRONMENTS.items() if env_filter.lower() in k.lower()}
        if not envs_to_test:
            print(f"No environments matching '{env_filter}'. Available:")
            for name in sorted(ENVIRONMENTS):
                print(f"  {name}")
            sys.exit(1)
    else:
        envs_to_test = ENVIRONMENTS

    # --- Load demos using a joint-absolute env (source format) ---
    joint_abs_action_mode = JointPositionActionMode(
        floating_base=True, absolute=True, floating_dofs=[],
        enforce_joint_velocity_limits=False,
    )

    results = {}
    grand_success = 0
    grand_total = 0
    skipped_envs = []

    print("=" * 70)
    print("REPLAYING DEMOS  |  Converted: Joint Absolute → Delta EE (rotvec)")
    print("=" * 70)

    for env_name, env_cls in envs_to_test.items():
        print(f"\n--- {env_name} ---")

        # 1) Create a joint-absolute env to load demos
        try:
            src_env = env_cls(
                action_mode=JointPositionActionMode(
                    floating_base=True, absolute=True, floating_dofs=[],
                    enforce_joint_velocity_limits=False,
                ),
                render_mode=None,
                control_frequency=CONTROL_FREQ,
                robot_cls=BimanualPanda,
            )
        except Exception as e:
            print(f"  [SKIP] Could not create source env: {e}")
            skipped_envs.append((env_name, f"src env failed: {e}"))
            continue

        # Load demos
        try:
            metadata = Metadata.from_env(src_env)
            demos = DemoStore().get_demos(metadata, amount=max_demos, frequency=CONTROL_FREQ)
        except DemoNotFoundError:
            print(f"  [SKIP] No demos found")
            skipped_envs.append((env_name, "no demos"))
            src_env.close()
            continue
        except Exception as e:
            print(f"  [SKIP] Error loading demos: {e}")
            skipped_envs.append((env_name, str(e)))
            src_env.close()
            continue

        src_env.close()

        if not demos:
            print(f"  [SKIP] 0 demos returned")
            skipped_envs.append((env_name, "0 demos"))
            continue

        # 2) Create a delta-EE env (target format)
        try:
            ee_env = env_cls(
                action_mode=JointPositionActionMode(
                    floating_base=True,
                    absolute=False,
                    ee=True,
                    floating_dofs=[],
                    enforce_joint_velocity_limits=False,
                ),
                render_mode=None,
                control_frequency=CONTROL_FREQ,
                robot_cls=BimanualPanda,
            )
        except Exception as e:
            print(f"  [SKIP] Could not create delta-EE env: {e}")
            skipped_envs.append((env_name, f"ee env failed: {e}"))
            continue

        # 3) Replay each demo through conversion
        successes = 0
        failures = 0
        errors = 0
        failed_ids = []

        for idx, demo in enumerate(demos):
            tag = f"[{idx + 1}/{len(demos)}]"
            try:
                ok = replay_ee_delta_check_success(demo, ee_env)
                if ok:
                    successes += 1
                    print(f"  {tag} ✓  {demo.uuid}")
                else:
                    failures += 1
                    failed_ids.append(demo.uuid)
                    print(f"  {tag} ✗  {demo.uuid}")
            except Exception as e:
                errors += 1
                failed_ids.append(f"{demo.uuid} (error: {e})")
                print(f"  {tag} ERROR  {demo.uuid}: {e}")

        total = len(demos)
        rate = successes / total * 100 if total > 0 else 0.0
        results[env_name] = {
            "total": total, "success": successes, "fail": failures,
            "error": errors, "rate": rate, "failed_ids": failed_ids,
        }
        grand_success += successes
        grand_total += total

        print(
            f"  Demos: {total}  |  Success: {successes}  |  Fail: {failures}  "
            f"|  Error: {errors}  |  Rate: {rate:.1f}%"
        )
        ee_env.close()

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY  (Joint Absolute → Delta EE rotvec)")
    print("=" * 70)
    print(f"{'Environment':<50} {'Total':>5} {'Pass':>5} {'Fail':>5} {'Rate':>7}")
    print("-" * 70)
    for env_name, r in results.items():
        print(
            f"{env_name:<50} {r['total']:>5} {r['success']:>5} "
            f"{r['fail']:>5} {r['rate']:>6.1f}%"
        )
    print("-" * 70)
    overall_rate = grand_success / grand_total * 100 if grand_total > 0 else 0.0
    print(
        f"{'OVERALL':<50} {grand_total:>5} {grand_success:>5} "
        f"{grand_total - grand_success:>5} {overall_rate:>6.1f}%"
    )

    if skipped_envs:
        print(f"\nSkipped environments ({len(skipped_envs)}):")
        for name, reason in skipped_envs:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()
