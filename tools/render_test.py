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
    model = env._mojo.model

    # Enumerate every camera registered in the compiled MuJoCo model and render
    # one frame per ID. Names come from mj_id2name so we can see exactly which
    # index maps to head/left_wrist/right_wrist/external/etc.
    import mujoco
    print(f"ncam = {model.ncam}")
    for i in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) or f"unnamed_{i}"
        safe = name.replace("/", "_").replace(" ", "_")
        try:
            save(f"cam_{i:02d}_{safe}", renderer.render("rgb_array", camera_id=i))
        except Exception as e:
            print(f"[cam_{i:02d}_{safe}] failed: {e}")

    # Composite env.render() — what actually hits the Agora pipeline.
    try:
        save("env_render_composite", env.render())
    except Exception as e:
        print(f"[env_render_composite] failed: {e}")


if __name__ == "__main__":
    main()
