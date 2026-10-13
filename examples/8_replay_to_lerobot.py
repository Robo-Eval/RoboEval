#!/usr/bin/env python3
"""Replay RoboEval demos, convert successful ones to LeRobot datasets.

Supports all action mode combinations:
  - joint absolute  (stored demos, no conversion needed)
  - joint delta     (DemoConverter.absolute_to_delta)
  - ee absolute     (DemoConverter.joint_to_ee)
  - ee delta        (DemoConverter.joint_absolute_to_ee_delta)

The repo name is auto-assembled from the action mode and frequency:
  roboeval_{ee|joint}_{absolute|delta}_{freq}hz/{TaskName}

For each available task:
1. Loads demos one-at-a-time from the DemoStore cache (memory-efficient)
2. Recreates observations via replay in an env with cameras
3. Converts actions to the target action mode
4. Replays converted actions to verify success (reward > 0)
5. Writes successful demos to a per-task LeRobot dataset
6. Outputs used trajectory names and counts to a text file

Prerequisites:
    Run 1_data_replay.py first to download demo data (triggers pull_demos).

Usage:
    # EE delta at 20 Hz (default)
    python examples/8_replay_to_lerobot.py

    # Joint delta at 10 Hz
    python examples/8_replay_to_lerobot.py --ee --no-ee --no-absolute --freq 10

    # All four modes explicitly:
    python examples/8_replay_to_lerobot.py --ee --delta          # ee delta
    python examples/8_replay_to_lerobot.py --ee --absolute       # ee absolute
    python examples/8_replay_to_lerobot.py --joint --delta       # joint delta
    python examples/8_replay_to_lerobot.py --joint --absolute    # joint absolute

    # Limit demos / tasks
    python examples/8_replay_to_lerobot.py --amount 10 --tasks LiftPot CubeHandover
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil

# ---- LeRobot
from lerobot.datasets.lerobot_dataset import (
    HF_LEROBOT_HOME,
    LeRobotDataset,
)
from lerobot.datasets.utils import embed_images, hf_transform_to_torch
import datasets as hf_datasets

# ---- RoboEval
from roboeval.action_modes import JointPositionActionMode
from roboeval.robots.configs.panda import BimanualPanda
from roboeval.utils.observation_config import ObservationConfig, CameraConfig
from roboeval.demonstrations.demo_store import DemoStore, DemoNotFoundError
from roboeval.demonstrations.demo_converter import DemoConverter
from roboeval.demonstrations.demo import Demo
from roboeval.demonstrations.utils import Metadata, ObservationMode
from roboeval.roboeval_env import CONTROL_FREQUENCY_MAX

# Import all available task environments
from roboeval.envs.lift_pot import (
    LiftPot, LiftPotPosition, LiftPotOrientation, LiftPotPositionAndOrientation,
)
from roboeval.envs.manipulation import (
    CubeHandover, CubeHandoverPosition, CubeHandoverOrientation,
    CubeHandoverPositionAndOrientation, VerticalCubeHandover,
    StackTwoBlocks, StackTwoBlocksPosition, StackTwoBlocksOrientation,
    StackTwoBlocksPositionAndOrientation,
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
)

# ====================== task registry ======================

TASKS = {
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
}

CAMERA_CONFIG = ObservationConfig(
    cameras=[
        CameraConfig(name="external", rgb=True, depth=False, resolution=(256, 256)),
        CameraConfig(name="head",     rgb=True, depth=False, resolution=(256, 256)),
        CameraConfig(name="front",    rgb=True, depth=False, resolution=(256, 256)),
        CameraConfig(name="left_wrist", rgb=True, depth=False, resolution=(256, 256)),
        CameraConfig(name="right_wrist", rgb=True, depth=False, resolution=(256, 256)),
    ],
)


# ====================== action mode helpers =================================

def make_action_mode_label(ee: bool, absolute: bool) -> str:
    """Build a human-readable label like 'ee_delta' or 'joint_absolute'."""
    space = "ee" if ee else "joint"
    ref = "absolute" if absolute else "delta"
    return f"{space}_{ref}"


def make_repo_id(ee: bool, absolute: bool, freq: int) -> str:
    """Auto-assemble repo id: roboeval_<space>_<ref>_<freq>hz"""
    label = make_action_mode_label(ee, absolute)
    return f"roboeval_{label}_{freq}hz"


def convert_demo(demo: Demo, ee: bool, absolute: bool) -> Demo:
    """Convert a joint-absolute demo to the target action mode.

    Source demos are always stored as joint absolute. The four targets are:
      joint absolute → no conversion
      joint delta    → DemoConverter.absolute_to_delta
      ee absolute    → DemoConverter.joint_to_ee
      ee delta       → DemoConverter.joint_absolute_to_ee_delta
    """
    if ee and not absolute:
        return DemoConverter.joint_absolute_to_ee_delta(demo)
    elif ee and absolute:
        return DemoConverter.joint_to_ee(demo)
    elif not ee and not absolute:
        return DemoConverter.absolute_to_delta(demo)
    else:
        # joint absolute — no conversion needed
        from copy import deepcopy
        return Demo(metadata=demo.metadata, timesteps=deepcopy(demo.timesteps))


# ====================== helpers ==============================================

def _camel_to_snake(name: str) -> str:
    """Convert CamelCase task name to snake_case (e.g. LiftPot → lift_pot)."""
    import re
    s = re.sub(r"([A-Z])", r"_\1", name).lower().lstrip("_")
    return s


def _load_task_instructions(task_name: str) -> Optional[List[str]]:
    """Load instruction variations from roboeval/envs/instructions/<task_family>.json.

    Task variant names (e.g. LiftPotPosition) are mapped to their base task
    family (e.g. lift_pot) by stripping known suffixes before lookup.
    """
    try:
        import roboeval
        roboeval_path = Path(roboeval.__file__).parent
    except ImportError:
        return None

    # Strip variant prefixes and suffixes to get the base task family name
    # e.g. VerticalCubeHandover → CubeHandover, DragOverAndLiftTray → LiftTray
    base = task_name
    for prefix in ("Vertical", "DragOverAnd"):
        if base.startswith(prefix) and base != prefix:
            base = base[len(prefix):]
            break
    for suffix in ("PositionAndOrientation", "Position", "Orientation"):
        if base.endswith(suffix) and base != suffix:
            base = base[: -len(suffix)]
            break

    snake_name = _camel_to_snake(base)
    json_path = roboeval_path / "envs" / "instructions" / f"{snake_name}.json"

    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                instructions = json.load(f).get("instructions", [])
            if instructions:
                return instructions
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def _to_image(arr: np.ndarray) -> np.ndarray:
    """(C,H,W) or (H,W,C) float/uint8 → uint8 (H,W,C) in [0,255]."""
    img = arr
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype in (np.float32, np.float64):
        img = np.clip(img, 0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img


def _detect_cameras(obs: Dict[str, np.ndarray]) -> List[Tuple[str, str]]:
    """Find 'rgb_<name>' keys → (obs_key, feature_key) pairs."""
    return [(k, f"cam_{k[4:]}") for k in sorted(obs) if k.startswith("rgb_")]


def _first_available(obs: Dict[str, np.ndarray], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in obs:
            return k
    return None


def _patched_save_episode_table(self, episode_buffer: dict, episode_index: int) -> None:
    """Memory-efficient episode save that skips in-memory concatenation."""
    episode_dict = {key: episode_buffer[key] for key in self.hf_features}
    ep_dataset = hf_datasets.Dataset.from_dict(
        episode_dict, features=self.hf_features, split="train"
    )
    ep_dataset = embed_images(ep_dataset)
    ep_dataset.set_transform(hf_transform_to_torch)

    ep_data_path = self.root / self.meta.get_data_file_path(ep_index=episode_index)
    ep_data_path.parent.mkdir(parents=True, exist_ok=True)
    ep_dataset.to_parquet(ep_data_path)

    del episode_dict, ep_dataset
    gc.collect()


# ====================== per-task processing ================================

def process_task(
    task_name: str,
    task_cls,
    demo_store: DemoStore,
    repo_id_base: str,
    ee: bool,
    absolute: bool,
    freq: int,
    amount: int = -1,
    overwrite: bool = True,
    image_writer_threads: int = 4,
    image_writer_processes: int = 2,
) -> List[str]:
    """Process all demos for one task: load → convert → success check → LeRobot.

    Returns:
        List of successful demo UUIDs.
    """
    process = psutil.Process()
    mode_label = make_action_mode_label(ee, absolute)

    # ---- 1. Resolve lightweight demo paths from the cache ----
    light_metadata = Metadata.from_env_cls(
        env_cls=task_cls,
        action_mode=JointPositionActionMode,
        floating_dofs=[],
        obs_mode=ObservationMode.Lightweight,
        action_mode_absolute=True,
        end_effector_mode=False,
    )
    # Ensure demos are pulled / cached
    demo_store.pull_demos()
    demo_paths = demo_store.list_demo_paths(light_metadata)
    # Filter for actual safetensors files
    demo_paths = [p for p in demo_paths if p.suffix == ".safetensors"]
    if not demo_paths:
        print("  No demo files found in cache. Skipping.")
        return []

    if 0 < amount < len(demo_paths):
        demo_paths = demo_paths[:amount]
    print(f"  Found {len(demo_paths)} demo file(s)")

    # ---- 2. Create environments ----
    # Absolute joint env with cameras (for observation recreation)
    abs_env = task_cls(
        action_mode=JointPositionActionMode(
            floating_base=True, absolute=True, floating_dofs=[],
        ),
        render_mode=None,
        control_frequency=freq,
        robot_cls=BimanualPanda,
        observation_config=CAMERA_CONFIG,
    )
    # Target action mode env (for success checking — no cameras needed)
    target_env = task_cls(
        action_mode=JointPositionActionMode(
            floating_base=True, absolute=absolute, ee=ee, floating_dofs=[],
        ),
        render_mode=None,
        control_frequency=freq,
        robot_cls=BimanualPanda,
    )

    # ---- 3. Probe observation / action shapes from the first demo ----
    first_demo_raw = Demo.from_safetensors(demo_paths[0])
    if first_demo_raw is None:
        print(f"  Failed to load first demo from {demo_paths[0]}. Skipping.")
        abs_env.close(); target_env.close()
        return []

    robot = first_demo_raw.metadata.get_robot()
    first_demo = DemoConverter.decimate(
        first_demo_raw, freq, CONTROL_FREQUENCY_MAX, robot=robot,
    )
    first_demo = DemoConverter.create_demo_in_new_env(first_demo, abs_env)

    # Convert first demo to get target action shape
    first_converted = convert_demo(first_demo, ee, absolute)
    target_act_shape = tuple(
        np.asarray(first_converted.timesteps[0].executed_action).shape
    )

    # Probe observations from env reset
    obs0, _ = abs_env.reset(seed=first_demo.seed)
    cam_pairs = _detect_cameras(obs0)
    if not cam_pairs:
        print("  No camera keys found in observation. Skipping.")
        abs_env.close(); target_env.close()
        return []

    state_key = _first_available(
        obs0, ["proprioception", "state", "robot_state", "obs_state"],
    )
    if state_key is None:
        print("  No state key found in observation. Skipping.")
        abs_env.close(); target_env.close()
        return []

    gripper_key = _first_available(
        obs0, ["proprioception_grippers", "gripper_state", "state_gripper"],
    )

    # ---- 4. Build LeRobot feature spec ----
    features = {}
    for obs_key, feat_key in cam_pairs:
        img = _to_image(obs0[obs_key])
        H, W, C = img.shape
        features[feat_key] = {
            "dtype": "image", "shape": (H, W, C),
            "names": ["height", "width", "channel"],
        }

    state0 = np.asarray(obs0[state_key], dtype=np.float32)
    features["state"] = {
        "dtype": "float32", "shape": tuple(state0.shape), "names": ["state"],
    }
    if gripper_key is not None:
        gr0 = np.asarray(obs0[gripper_key], dtype=np.float32)
        features["state_gripper"] = {
            "dtype": "float32", "shape": tuple(gr0.shape), "names": ["state_gripper"],
        }

    features["actions"] = {
        "dtype": "float32", "shape": target_act_shape, "names": ["actions"],
    }
    features["reward"] = {"dtype": "float32", "shape": (1,), "names": ["reward"]}
    features["is_terminal"] = {"dtype": "bool", "shape": (1,), "names": ["is_terminal"]}
    features["truncate"] = {"dtype": "bool", "shape": (1,), "names": ["truncate"]}
    features["uuid"] = {"dtype": "string", "shape": (1,), "names": ["uuid"]}

    # ---- 5. Create LeRobot dataset ----
    task_repo_id = f"{repo_id_base}/{task_name}"
    out_path = HF_LEROBOT_HOME / task_repo_id
    if overwrite and out_path.exists():
        shutil.rmtree(out_path)

    dataset = LeRobotDataset.create(
        repo_id=task_repo_id,
        robot_type="panda",
        fps=freq,
        features=features,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )
    # Monkey-patch to prevent memory accumulation
    dataset._save_episode_table = (
        lambda buf, idx: _patched_save_episode_table(dataset, buf, idx)
    )

    print(f"  LeRobot dataset: {out_path}")
    print(f"  Action mode: {mode_label} @ {freq} Hz")
    print(f"  Cameras: {[f'{ok}→{fk}' for ok, fk in cam_pairs]}")
    print(f"  State: {state_key} {state0.shape}  |  Actions: {target_act_shape}")

    # Load task instructions for the "task" field
    instruction_variations = _load_task_instructions(task_name)
    if instruction_variations is None:
        instruction_variations = [task_name]

    # Free schema-probing objects
    del first_demo_raw, first_demo, first_converted
    gc.collect()

    # ---- 6. Process demos one at a time ----
    successful_uuids: List[str] = []
    for demo_idx, path in enumerate(demo_paths):
        mem = process.memory_info().rss / 1024 / 1024
        print(f"  [{demo_idx+1}/{len(demo_paths)}] Loading {path.stem}… (mem {mem:.0f} MB)")

        # Load and prepare demo with observations
        demo = Demo.from_safetensors(path)
        if demo is None:
            print(f"    ⚠ Failed to load. Skipping.")
            continue
        demo_robot = demo.metadata.get_robot()
        demo = DemoConverter.decimate(
            demo, freq, CONTROL_FREQUENCY_MAX, robot=demo_robot,
        )
        demo = DemoConverter.create_demo_in_new_env(demo, abs_env)

        # Convert actions to target mode
        try:
            converted_demo = convert_demo(demo, ee, absolute)
        except Exception as e:
            print(f"    ⚠ {mode_label} conversion failed: {e}. Skipping.")
            del demo; gc.collect()
            continue

        # Success check: replay converted actions in target-mode env
        target_env.reset(seed=converted_demo.seed)
        is_successful = False
        for step in converted_demo.timesteps:
            target_env.step(step.executed_action, fast=True)
            if target_env.reward > 0:
                is_successful = True
                break

        if not is_successful:
            print(f"    ✗ Not successful. Skipping.")
            del demo, converted_demo; gc.collect()
            continue

        # ---- Write successful demo to LeRobot ----
        print(f"    ✓ Successful — writing to LeRobot")
        successful_uuids.append(demo.uuid)
        sampled_instruction = random.choice(instruction_variations)

        prev_reward, prev_is_term, prev_trunc = 0.0, False, False
        for t, ts in enumerate(demo.timesteps):
            if t == 0:
                reward, is_term, trunc = 0.0, False, False
            else:
                reward, is_term, trunc = prev_reward, prev_is_term, prev_trunc

            # Observations come from the abs-joint replay (demo with images)
            frame_imgs = {
                fk: _to_image(ts.observation[ok]).copy()
                for ok, fk in cam_pairs
            }
            state = np.asarray(ts.observation[state_key], dtype=np.float32).copy()

            # Actions come from the converted demo
            action = np.asarray(
                converted_demo.timesteps[t].executed_action, dtype=np.float32
            ).copy()

            prev_reward = float(ts.reward)
            prev_is_term = bool(ts.termination)
            prev_trunc = bool(ts.truncation)

            frame = {
                **frame_imgs,
                "state": state,
                "actions": action,
                "reward": np.asarray(reward, dtype=np.float32).reshape((1,)),
                "is_terminal": np.asarray(is_term, dtype=bool).reshape((1,)),
                "truncate": np.asarray(trunc, dtype=bool).reshape((1,)),
                "uuid": demo.uuid,
            }
            if gripper_key:
                frame["state_gripper"] = np.asarray(
                    ts.observation[gripper_key], dtype=np.float32
                ).copy()

            dataset.add_frame(frame, task=sampled_instruction)

        # Clean up loop variables that hold demo references
        del ts, frame_imgs, state, action, frame
        del demo, converted_demo
        gc.collect()

        dataset.save_episode()

        if (demo_idx + 1) % 10 == 0:
            mem = process.memory_info().rss / 1024 / 1024
            print(f"  … {demo_idx+1}/{len(demo_paths)} done, "
                  f"{len(successful_uuids)} successful so far (mem {mem:.0f} MB)")

    abs_env.close()
    target_env.close()

    print(f"  → {len(successful_uuids)}/{len(demo_paths)} successful")
    if successful_uuids:
        print(f"  LeRobot path: {out_path}")
    return successful_uuids


# ====================== main ======================

def main():
    parser = argparse.ArgumentParser(
        description="Replay RoboEval demos with a target action mode, "
                    "convert successful ones to LeRobot format."
    )
    # ---- Action mode ----
    space_group = parser.add_mutually_exclusive_group()
    space_group.add_argument(
        "--ee", dest="ee", action="store_true", default=True,
        help="End-effector control space (default)",
    )
    space_group.add_argument(
        "--joint", dest="ee", action="store_false",
        help="Joint control space",
    )
    ref_group = parser.add_mutually_exclusive_group()
    ref_group.add_argument(
        "--delta", dest="absolute", action="store_false", default=False,
        help="Delta (relative) actions (default)",
    )
    ref_group.add_argument(
        "--absolute", dest="absolute", action="store_true",
        help="Absolute actions",
    )
    parser.add_argument(
        "--freq", type=int, default=20,
        help="Control frequency in Hz (default: 20)",
    )

    # ---- Dataset / IO ----
    parser.add_argument(
        "--repo_id", type=str, default=None,
        help="Override auto-generated LeRobot repo-id. "
             "Default: roboeval_<space>_<ref>_<freq>hz",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output text file for successful trajectory names. "
             "Default: successful_demos_<mode>_<freq>hz.txt",
    )
    parser.add_argument(
        "--amount", type=int, default=-1,
        help="Max demos per task (-1 = all)",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="*", default=None,
        help="Specific task class names to process (default: all). "
             "Example: --tasks LiftPot CubeHandover",
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=True,
        help="Overwrite existing LeRobot datasets",
    )
    parser.add_argument(
        "--no-overwrite", dest="overwrite", action="store_false",
        help="Do not overwrite existing LeRobot datasets",
    )
    parser.add_argument(
        "--image_writer_threads", type=int, default=8,
        help="Threads for LeRobot image writing (default: 8)",
    )
    parser.add_argument(
        "--image_writer_processes", type=int, default=4,
        help="Processes for LeRobot image writing (default: 4)",
    )
    args = parser.parse_args()

    # ---- Derive repo_id and output filename from action mode ----
    mode_label = make_action_mode_label(args.ee, args.absolute)
    if args.repo_id is None:
        args.repo_id = make_repo_id(args.ee, args.absolute, args.freq)
    if args.output is None:
        args.output = f"successful_demos_{mode_label}_{args.freq}hz.txt"

    print(f"Action mode : {mode_label}")
    print(f"Frequency   : {args.freq} Hz")
    print(f"Repo ID     : {args.repo_id}")
    print(f"Output file : {args.output}")

    # Select tasks
    if args.tasks:
        tasks = {k: v for k, v in TASKS.items() if k in args.tasks}
        unknown = set(args.tasks) - set(TASKS.keys())
        if unknown:
            print(f"Warning: unknown task(s) ignored: {unknown}")
        if not tasks:
            print(f"No matching tasks. Available:\n  {list(TASKS.keys())}")
            return
    else:
        tasks = TASKS

    demo_store = DemoStore()
    all_successful: Dict[str, List[str]] = {}
    total_successful = 0

    for task_name, task_cls in tasks.items():
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")

        try:
            uuids = process_task(
                task_name=task_name,
                task_cls=task_cls,
                demo_store=demo_store,
                repo_id_base=args.repo_id,
                ee=args.ee,
                absolute=args.absolute,
                freq=args.freq,
                amount=args.amount,
                overwrite=args.overwrite,
                image_writer_threads=args.image_writer_threads,
                image_writer_processes=args.image_writer_processes,
            )
        except Exception as e:
            print(f"  ⚠ Task failed: {e}")
            uuids = []

        if uuids:
            all_successful[task_name] = uuids
        total_successful += len(uuids)

    # ---- Write results to txt file ----
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        f.write(f"# Successful RoboEval demos — {mode_label} @ {args.freq} Hz\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# LeRobot base repo_id: {args.repo_id}\n")
        f.write(f"# Total successful: {total_successful}\n\n")

        for task_name, uuids in all_successful.items():
            f.write(f"[{task_name}] ({len(uuids)} demos)\n")
            for uid in uuids:
                f.write(f"  {uid}\n")
            f.write("\n")

        f.write(f"Total: {total_successful}\n")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    tasks_with = sum(1 for t in tasks if t in all_successful)
    print(f"Action mode                 : {mode_label} @ {args.freq} Hz")
    print(f"Tasks with successful demos : {tasks_with}/{len(tasks)}")
    print(f"Successful demos            : {total_successful}")
    print(f"LeRobot datasets at         : {HF_LEROBOT_HOME / args.repo_id}")
    print(f"Trajectory list written to  : {output_path.resolve()}")


if __name__ == "__main__":
    main()
