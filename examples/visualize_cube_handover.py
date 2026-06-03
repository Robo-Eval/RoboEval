#!/usr/bin/env python3
"""Visualize CubeHandoverPositionAndOrientation demonstrations.

Saves each demo replay as an MP4 video in examples/cube_handover_demos/.
"""

import os
from pathlib import Path

import imageio
import numpy as np

from roboeval.action_modes import JointPositionActionMode
from roboeval.robots.configs.panda import BimanualPanda
from roboeval.utils.observation_config import ObservationConfig, CameraConfig
from roboeval.demonstrations.demo import Demo
from roboeval.demonstrations.demo_player import DemoPlayer
from roboeval.demonstrations.demo_store import DemoStore
from roboeval.demonstrations.utils import Metadata
from roboeval.demonstrations.const import SAFETENSORS_SUFFIX
from roboeval.envs.manipulation import CubeHandoverPositionAndOrientation
from roboeval.roboeval_env import CONTROL_FREQUENCY_MAX

CONTROL_FREQ = 20
NUM_DEMOS = 3
OUTPUT_DIR = Path(__file__).parent / "cube_handover_demos"


def replay_and_record(demo, env, demo_frequency):
    """Replay a demo and return list of RGB frames + success flag."""
    timesteps = DemoPlayer._get_timesteps_for_replay(demo, env, demo_frequency)
    env.reset(seed=demo.seed)
    frames = [env.render()]

    for step in timesteps:
        obs, reward, terminated, truncated, info = env.step(step.executed_action)
        frames.append(env.render())

    success = reward > 0
    return frames, success


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    env = CubeHandoverPositionAndOrientation(
        action_mode=JointPositionActionMode(
            floating_base=True, absolute=True, floating_dofs=[],
            # Diagnostic: enforce the joint velocity limit (slew clamp) and step
            # physics until each waypoint is reached. This time-stretches the
            # trajectory so a demo collected without velocity limits can still
            # reach its targets despite the per-step slew cap.
            enforce_joint_velocity_limits=False,
            block_until_reached=False,
        ),
        render_mode="rgb_array",
        control_frequency=CONTROL_FREQ,
        robot_cls=BimanualPanda,
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig(
                    name="external",
                    rgb=True,
                    depth=False,
                    resolution=(480, 640),
                    pos=[0.0, 10.0, 10.0],
                )
            ],
        ),
    )

    # Load lightweight demos directly from the cache at max frequency.
    # DemoPlayer handles decimation to CONTROL_FREQ during replay.
    metadata = Metadata.from_env(env, is_lightweight=True)
    demo_store = DemoStore()
    demos_dir = demo_store._create_path(metadata).parent
    demo_files = sorted(demos_dir.glob(f"*{SAFETENSORS_SUFFIX}"))[:NUM_DEMOS]
    demos = [Demo.from_safetensors(f) for f in demo_files]
    print(f"Loaded {len(demos)} demos from {demos_dir}")

    for i, demo in enumerate(demos):
        print(f"\n[{i+1}/{len(demos)}] Replaying demo {demo.uuid} ...", end=" ")
        frames, success = replay_and_record(demo, env, CONTROL_FREQUENCY_MAX)
        tag = "success" if success else "fail"
        out_path = OUTPUT_DIR / f"demo_{i:02d}_{tag}_{demo.uuid[:8]}.mp4"
        imageio.mimsave(str(out_path), frames, fps=CONTROL_FREQ)
        print(f"{'SUCCESS' if success else 'FAIL'} -> {out_path.name}")

    env.close()
    print(f"\nVideos saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
