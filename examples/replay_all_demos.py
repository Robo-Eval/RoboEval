#!/usr/bin/env python3
"""Replay all demonstrations with joint absolute action mode and report success rates."""

import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

from roboeval.action_modes import JointPositionActionMode
from roboeval.robots.configs.panda import BimanualPanda
from roboeval.demonstrations.demo_player import DemoPlayer
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


def replay_demo_check_success(demo, env, demo_frequency):
    """Replay a single demo and return whether it succeeded (reward > 0)."""
    timesteps = DemoPlayer._get_timesteps_for_replay(demo, env, demo_frequency)
    env.reset(seed=demo.seed)
    for step in timesteps:
        env.step(step.executed_action, fast=True)
        if env.reward > 0:
            return True
    return False


def main():
    demo_store = DemoStore()
    action_mode = JointPositionActionMode(
        floating_base=True, absolute=True, floating_dofs=[],
        enforce_joint_velocity_limits=False,
    )

    # Per-environment results
    results = {}
    grand_success = 0
    grand_total = 0
    skipped_envs = []

    print("=" * 70)
    print("REPLAYING ALL DEMOS  |  Action mode: Joint Absolute")
    print("=" * 70)

    for env_name, env_cls in ENVIRONMENTS.items():
        print(f"\n--- {env_name} ---")

        # Create environment (no rendering for speed)
        try:
            env = env_cls(
                action_mode=JointPositionActionMode(
                    floating_base=True, absolute=True, floating_dofs=[],
                    enforce_joint_velocity_limits=False,
                ),
                render_mode=None,
                control_frequency=CONTROL_FREQ,
                robot_cls=BimanualPanda,
            )
        except Exception as e:
            print(f"  [SKIP] Could not create environment: {e}")
            skipped_envs.append((env_name, f"env creation failed: {e}"))
            continue

        # Load demos
        try:
            metadata = Metadata.from_env(env)
            demos = demo_store.get_demos(
                metadata, amount=-1, frequency=CONTROL_FREQ
            )
        except DemoNotFoundError:
            print(f"  [SKIP] No demos found in DemoStore")
            skipped_envs.append((env_name, "no demos"))
            env.close()
            continue
        except Exception as e:
            print(f"  [SKIP] Error loading demos: {e}")
            skipped_envs.append((env_name, str(e)))
            env.close()
            continue

        if not demos:
            print(f"  [SKIP] 0 demos returned")
            skipped_envs.append((env_name, "0 demos"))
            env.close()
            continue

        # Replay each demo
        successes = 0
        failures = 0
        errors = 0
        failed_ids = []

        for i, demo in enumerate(demos):
            try:
                ok = replay_demo_check_success(demo, env, CONTROL_FREQ)
                if ok:
                    successes += 1
                else:
                    failures += 1
                    failed_ids.append(demo.uuid)
            except Exception as e:
                errors += 1
                failed_ids.append(f"{demo.uuid} (error: {e})")

        total = len(demos)
        rate = successes / total * 100 if total > 0 else 0.0
        results[env_name] = {
            "total": total,
            "success": successes,
            "fail": failures,
            "error": errors,
            "rate": rate,
            "failed_ids": failed_ids,
        }
        grand_success += successes
        grand_total += total

        print(
            f"  Demos: {total}  |  Success: {successes}  |  Fail: {failures}  "
            f"|  Error: {errors}  |  Rate: {rate:.1f}%"
        )

        env.close()

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
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

    # Print failed demo IDs
    any_failures = any(r["failed_ids"] for r in results.values())
    if any_failures:
        print("\nFailed demo IDs per environment:")
        for env_name, r in results.items():
            if r["failed_ids"]:
                print(f"  {env_name}:")
                for fid in r["failed_ids"]:
                    print(f"    - {fid}")


if __name__ == "__main__":
    main()
