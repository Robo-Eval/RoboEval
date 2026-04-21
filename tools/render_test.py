"""Render-one-frame smoke test.

Boots a RotateValve env with render_mode="rgb_array", renders a single frame
with each camera setting we care about, and writes the result to /tmp as PNG.
Lets us tell apart a MuJoCo rendering problem from a pipeline problem.
"""
import os
import sys
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Match the production sim's headless rendering config before anything imports mujoco.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from PIL import Image
from roboeval.action_modes import JointPositionActionMode
from roboeval.envs.rotate_utility_objects import RotateValve

OUT = Path("/tmp/playground_render_test")
OUT.mkdir(exist_ok=True)


def save(name, frame):
    """Save frame to PNG. Print shape/dtype/mean-per-channel for diagnostics."""
    if frame is None:
        print(f"[{name}] frame is None")
        return
    print(f"[{name}] shape={frame.shape} dtype={frame.dtype}")
    if frame.ndim == 3:
        means = frame.reshape(-1, frame.shape[-1]).mean(axis=0)
        print(f"[{name}] per-channel mean = {means}")
    path = OUT / f"{name}.png"
    Image.fromarray(frame).save(path)
    print(f"[{name}] wrote {path}")


def main():
    action_mode = JointPositionActionMode(absolute=True, floating_base=True, floating_dofs=[])
    env = RotateValve(render_mode="rgb_array", action_mode=action_mode, robot_cls=None)
    env.mojo.model.vis.global_.offwidth = 1280
    env.mojo.model.vis.global_.offheight = 720
    env.reset()

    renderer = env.mujoco_renderer

    # 1. No kwargs — whatever mujoco's default camera is.
    save("default", renderer.render("rgb_array"))

    # 2. Free camera explicit.
    try:
        save("camera_id_-1", renderer.render("rgb_array", camera_id=-1))
    except Exception as e:
        print(f"[camera_id_-1] failed: {e}")

    # 3. First fixed camera.
    try:
        save("camera_id_0", renderer.render("rgb_array", camera_id=0))
    except Exception as e:
        print(f"[camera_id_0] failed: {e}")

    # 4. By name — what our TelearmsTeleop asks for.
    try:
        save("camera_name_external", renderer.render("rgb_array", camera_name="external"))
    except Exception as e:
        print(f"[camera_name_external] failed: {e}")

    # 5. env.render() — whatever the Telearms env wrapper returns.
    try:
        save("env_render", env.render())
    except Exception as e:
        print(f"[env_render] failed: {e}")


if __name__ == "__main__":
    main()
