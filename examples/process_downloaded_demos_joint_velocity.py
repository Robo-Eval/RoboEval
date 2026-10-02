#!/usr/bin/env python3
"""Process all joint-absolute demos in the DemoStore cache for joint velocity feasibility.

For each task and each ``*.safetensors`` file under the downloaded layout, compares limb
targets to the same discrete slew rule as ``JointPositionActionMode`` with
``enforce_joint_velocity_limits=True``. Demos that already satisfy the rule are skipped
unless ``--write-all`` is set. Otherwise writes adjusted trajectories under
``--output-dir`` using the **same relative paths** as in the DemoStore cache
(``robot_name / env_name / action_mode / observation_mode / … / <uuid>.safetensors``),
i.e. ``output_dir / path.relative_to(demo_store.cache_path)`` for each source file.

Observations in saved files are still the originals; re-simulate if you need aligned
pixels/state (see ``examples/8_replay_to_lerobot.py`` docstring).

Examples:
    # Default: native 500 Hz, only write demos that need adjustment
    python examples/process_downloaded_demos_joint_velocity.py --output-dir ~/roboeval_feasible

    # Dry run — report counts only
    python examples/process_downloaded_demos_joint_velocity.py --check-only --freq 500

    # Subset of tasks
    python examples/process_downloaded_demos_joint_velocity.py --output-dir ./out --tasks LiftPot
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from roboeval.demonstrations.demo import Demo
from roboeval.demonstrations.demo_converter import DemoConverter
from roboeval.demonstrations.demo_store import DemoStore
from roboeval.roboeval_env import CONTROL_FREQUENCY_MAX, CONTROL_FREQUENCY_MIN

from roboeval.demonstrations.demo_store_tasks import (
    DEMO_STORE_TASKS,
    joint_light_joint_abs_metadata,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root directory for output (required unless --check-only)",
    )
    parser.add_argument(
        "--freq",
        type=int,
        default=CONTROL_FREQUENCY_MAX,
        help=f"Control frequency in Hz (default: {CONTROL_FREQUENCY_MAX}, native demo rate)",
    )
    parser.add_argument(
        "--v-max",
        type=float,
        default=None,
        help="Max joint velocity rad/s (default: Franka nominal in JointPositionActionMode)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="*",
        default=None,
        help="Task class names (default: all known DemoStore tasks)",
    )
    parser.add_argument(
        "--amount",
        type=int,
        default=-1,
        help="Max demos per task (-1 = all files listed for that task)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Treat limb delta below this as unchanged (default: 1e-5 rad)",
    )
    parser.add_argument(
        "--retime",
        action="store_true",
        help="Write time-stretched (waypoint-preserving) demos instead of plain "
        "slew-clamped ones. Violation detection is unchanged. For self-consistent "
        "observations, re-simulate via examples/clamp_demo_joint_velocity.py --resim.",
    )
    parser.add_argument(
        "--write-all",
        action="store_true",
        help="Write every successfully processed demo, even if clamp does not change it",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Print per-task violation counts; do not write files",
    )
    args = parser.parse_args()

    if not (CONTROL_FREQUENCY_MIN <= args.freq <= CONTROL_FREQUENCY_MAX):
        print(f"--freq must be in [{CONTROL_FREQUENCY_MIN}, {CONTROL_FREQUENCY_MAX}]")
        return 1
    if not args.check_only and args.output_dir is None:
        print("Either pass --output-dir or use --check-only")
        return 1

    os.environ.setdefault("ROBOEVAL_SKIP_DEMO_VERSION_CHECK", "1")

    tasks = (
        {k: v for k, v in DEMO_STORE_TASKS.items() if k in args.tasks}
        if args.tasks
        else dict(DEMO_STORE_TASKS)
    )
    if args.tasks:
        unknown = set(args.tasks) - set(DEMO_STORE_TASKS)
        if unknown:
            print(f"Warning: unknown tasks ignored: {sorted(unknown)}")
        if not tasks:
            print("No valid tasks.")
            return 1

    demo_store = DemoStore()
    demo_store.pull_demos()
    cache_root = demo_store.cache_path

    total_files = 0
    total_viol = 0
    total_written = 0
    total_skip_ok = 0
    total_skip_err = 0

    for name, env_cls in tasks.items():
        meta = joint_light_joint_abs_metadata(env_cls)
        try:
            paths = demo_store.list_demo_paths(meta)
        except Exception as e:
            print(f"{name}: [SKIP] list_demo_paths: {e}")
            continue
        paths = [p for p in paths if p.suffix == ".safetensors"]
        if args.amount > 0 and len(paths) > args.amount:
            paths = paths[: args.amount]
        if not paths:
            print(f"{name}: no demos")
            continue

        viol = 0
        written = 0
        skip_ok = 0
        skip_err = 0

        for path in paths:
            total_files += 1
            demo = Demo.from_safetensors(path)
            if demo is None:
                skip_err += 1
                total_skip_err += 1
                continue
            try:
                delta = DemoConverter.max_limb_discrepancy_after_velocity_clamp(
                    demo, args.freq, args.v_max
                )
            except ValueError as e:
                skip_err += 1
                total_skip_err += 1
                print(f"  {path.name}: skip ({e})")
                continue

            needs_write = delta > args.atol or args.write_all
            if delta > args.atol:
                viol += 1
                total_viol += 1

            if args.check_only:
                continue

            assert args.output_dir is not None
            if not needs_write:
                skip_ok += 1
                total_skip_ok += 1
                continue

            try:
                if args.retime:
                    fixed = DemoConverter.retime_absolute_joint_velocity(
                        demo, control_frequency_hz=args.freq, max_joint_velocity_rad_s=args.v_max
                    )
                else:
                    fixed = DemoConverter.clamp_absolute_joint_velocity(
                        demo, control_frequency_hz=args.freq, max_joint_velocity_rad_s=args.v_max
                    )
            except ValueError as e:
                skip_err += 1
                total_skip_err += 1
                print(f"  {path.name}: clamp failed ({e})")
                continue

            try:
                rel = path.relative_to(cache_root)
            except ValueError:
                rel = Path(name) / path.name
            out_path = args.output_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fixed.save(out_path)
            written += 1
            total_written += 1

        mode = "check" if args.check_only else f"→ {args.output_dir} (DemoStore layout)"
        print(
            f"{name}: {len(paths)} file(s) {mode} | "
            f"violations {viol} | written {written} | already_ok {skip_ok} | errors {skip_err}"
        )

    print()
    print(
        f"Totals: files={total_files}, violations={total_viol}, "
        f"written={total_written}, already_feasible={total_skip_ok}, errors={total_skip_err}"
    )
    if args.output_dir and not args.check_only:
        print(f"Output root: {args.output_dir.resolve()}")
    return 0 if total_skip_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
