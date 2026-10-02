#!/usr/bin/env python3
"""Rewrite joint-absolute demos so limb targets respect per-step velocity slew.

Applies :func:`roboeval.demonstrations.demo_converter.DemoConverter.clamp_absolute_joint_velocity`
at a chosen control frequency. Observations in the file are unchanged; for learning
pipelines, re-simulate with ``DemoConverter.create_demo_in_new_env`` or LeRobot replay.

Modes:
  * ``--demos-dir`` / ``--output-dir``: copy every ``*.safetensors`` under input to output.
  * ``--use-demo-store``: under ``--output-dir``, mirror each file's path relative to the
    DemoStore versioned cache root (same tree as downloaded ``roboeval_demos/<version>/``).

To process **all** cached tasks with violation detection and selective writes, prefer
``examples/process_downloaded_demos_joint_velocity.py``.

Examples:
    python examples/clamp_demo_joint_velocity.py --demos-dir ./my_demos --output-dir ./out --freq 500

    python examples/clamp_demo_joint_velocity.py --use-demo-store --freq 500 \\
        --tasks LiftPot CubeHandover --output-dir ~/.roboeval/feasible_demos --amount 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from roboeval.demonstrations.demo import Demo
from roboeval.demonstrations.demo_converter import DemoConverter
from roboeval.demonstrations.demo_store import DemoStore
from roboeval.demonstrations.demo_store_tasks import (
    DEMO_STORE_TASKS,
    joint_light_joint_abs_metadata,
)
from roboeval.roboeval_env import CONTROL_FREQUENCY_MIN, CONTROL_FREQUENCY_MAX

# Alias for CLI help / backward readability
TASKS = DEMO_STORE_TASKS


def _transform_demo(demo: Demo, retime: bool, freq: int, v_max: float | None) -> Demo:
    """Slew-clamp (default) or time-stretch (retime) the demo's joint targets."""
    if retime:
        return DemoConverter.retime_absolute_joint_velocity(
            demo, control_frequency_hz=freq, max_joint_velocity_rad_s=v_max
        )
    return DemoConverter.clamp_absolute_joint_velocity(
        demo, control_frequency_hz=freq, max_joint_velocity_rad_s=v_max
    )


def _get_env(demo: Demo, freq: int, env_cache: dict):
    """Build (and cache) an enforcing env matching the demo's task/frequency."""
    key = (demo.metadata.env_cls.__name__, freq)
    if key not in env_cache:
        # action mode defaults to enforce_joint_velocity_limits=True
        env_cache[key] = demo.metadata.get_env(freq)
    return env_cache[key]


def process_one_demo(
    path: Path,
    out_path: Path,
    freq: int,
    v_max: float | None,
    retime: bool = False,
    resim: bool = False,
    env_cache: dict | None = None,
) -> bool:
    demo = Demo.from_safetensors(path)
    if demo is None:
        print(f"  skip (load failed): {path}")
        return False

    # For a faithful re-sim, feasibility must hold at the *replay* control rate,
    # so decimate the native-rate demo to `freq` before transforming.
    work = demo
    if resim and freq != CONTROL_FREQUENCY_MAX:
        work = DemoConverter.decimate(work, target_freq=freq)

    try:
        fixed = _transform_demo(work, retime, freq, v_max)
    except ValueError as e:
        print(f"  skip ({e}): {path.name}")
        return False

    if resim:
        # Roll the feasible targets forward in a fresh enforcing env so the saved
        # observations/actions are self-consistent under the velocity limit.
        env = _get_env(demo, freq, env_cache if env_cache is not None else {})
        fixed = DemoConverter.create_demo_in_new_env(fixed, env)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fixed.save(out_path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--demos-dir",
        type=Path,
        help="Directory containing .safetensors demos (recursive *.safetensors)",
    )
    src.add_argument(
        "--use-demo-store",
        action="store_true",
        help="Load from DemoStore cache (same paths as release zips)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output root directory for clamped .safetensors",
    )
    parser.add_argument(
        "--freq",
        type=int,
        required=True,
        help=f"Control frequency in Hz ({CONTROL_FREQUENCY_MIN}–{CONTROL_FREQUENCY_MAX})",
    )
    parser.add_argument(
        "--v-max",
        type=float,
        default=None,
        help="Max joint velocity rad/s (default: JointPositionActionMode default)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="*",
        default=None,
        help="With --use-demo-store: task class names (default: all TASKS)",
    )
    parser.add_argument(
        "--amount",
        type=int,
        default=-1,
        help="With --use-demo-store: max demos per task (-1 = all)",
    )
    parser.add_argument(
        "--preserve-relative-paths",
        action="store_true",
        help="With --demos-dir: mirror subdirs under output (default: flat by filename)",
    )
    parser.add_argument(
        "--retime",
        action="store_true",
        help="Time-stretch (insert substeps so every waypoint is reachable) instead "
        "of plain slew-clamp. Recommended when feeding learning pipelines.",
    )
    parser.add_argument(
        "--resim",
        action="store_true",
        help="Re-simulate the feasible targets in a fresh enforcing env to write "
        "self-consistent observations. Decimates to --freq first; slower.",
    )
    args = parser.parse_args()

    env_cache: dict = {}

    if not (CONTROL_FREQUENCY_MIN <= args.freq <= CONTROL_FREQUENCY_MAX):
        print(f"freq must be in [{CONTROL_FREQUENCY_MIN}, {CONTROL_FREQUENCY_MAX}]")
        return 1

    ok_count = 0
    skip_count = 0

    if args.demos_dir:
        paths = sorted(args.demos_dir.rglob("*.safetensors"))
        if not paths:
            print(f"No .safetensors under {args.demos_dir}")
            return 1
        for path in paths:
            if args.preserve_relative_paths:
                rel = path.relative_to(args.demos_dir)
                out_path = args.output_dir / rel
            else:
                out_path = args.output_dir / path.name
            if process_one_demo(
                path, out_path, args.freq, args.v_max,
                retime=args.retime, resim=args.resim, env_cache=env_cache,
            ):
                ok_count += 1
            else:
                skip_count += 1

    else:
        demo_store = DemoStore()
        demo_store.pull_demos()
        cache_root = demo_store.cache_path
        tasks = (
            {k: v for k, v in TASKS.items() if k in args.tasks}
            if args.tasks
            else dict(TASKS)
        )
        if args.tasks:
            unknown = set(args.tasks) - set(TASKS)
            if unknown:
                print(f"Warning: unknown tasks ignored: {sorted(unknown)}")
            if not tasks:
                print("No valid tasks.")
                return 1

        for name, env_cls in tasks.items():
            meta = joint_light_joint_abs_metadata(env_cls)
            try:
                paths = demo_store.list_demo_paths(meta)
            except Exception as e:
                print(f"{name}: skip list paths ({e})")
                continue
            paths = [p for p in paths if p.suffix == ".safetensors"]
            if args.amount > 0 and len(paths) > args.amount:
                paths = paths[: args.amount]
            if not paths:
                print(f"{name}: no files")
                continue
            print(f"{name}: {len(paths)} file(s) → {args.output_dir} (DemoStore layout)")
            for path in paths:
                try:
                    rel = path.relative_to(cache_root)
                except ValueError:
                    rel = Path(name) / path.name
                out_path = args.output_dir / rel
                if process_one_demo(
                    path, out_path, args.freq, args.v_max,
                    retime=args.retime, resim=args.resim, env_cache=env_cache,
                ):
                    ok_count += 1
                else:
                    skip_count += 1

    print(f"Done: wrote {ok_count}, skipped {skip_count} → {args.output_dir.resolve()}")
    return 0 if ok_count else 1


if __name__ == "__main__":
    sys.exit(main())
