"""Batch-render the composite view with many candidate external-cam poses.

For iterating on where the `external` camera sits without redeploying.
Writes one PNG per candidate to /tmp/playground_calibrate/ and prints the
xyaxes line you'd paste into roboeval/envs/xmls/world.xml for each.

Camera is always placed on the -X axis (behind the robot, Y=0) and looks
at a fixed target on the workspace, so we only sweep (distance, height).
"""
import os
import sys
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np
from PIL import Image

from roboeval.action_modes import JointPositionActionMode
from roboeval.envs.rotate_utility_objects import RotateValve

OUT = Path("/tmp/playground_calibrate")
OUT.mkdir(exist_ok=True)

# Target point the camera aims at — roughly workspace center (table surface).
TARGET = np.array([0.0, 0.0, 0.6])

# (distance_behind, height) pairs — distance is |X| from origin, height is Z.
# 5 distances × 4 heights = 20 candidates.
CANDIDATES = [
    (d, h)
    for d, h in product([0.4, 0.6, 0.8, 1.0, 1.2], [1.2, 1.5, 1.8, 2.1])
]

FOVY = 60.0


def pose_looking_at(pos, target):
    """Build (xyaxes_x, xyaxes_y, quat) for a camera at pos looking at target.

    Camera convention: -Z is forward (look dir), +X is right, +Y is up.
    Right axis is locked to world -Y so the robot's left arm appears on
    the left of the rendered frame (same as the Agora stream convention).
    """
    look_dir = target - pos
    look_dir /= np.linalg.norm(look_dir)

    z_axis = -look_dir                 # cam +Z
    x_axis = np.array([0.0, -1.0, 0.0])  # cam +X (right)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    # re-orthogonalise x in case (0,-1,0) isn't exactly perpendicular
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    rot = np.column_stack([x_axis, y_axis, z_axis]).flatten()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot)
    return x_axis, y_axis, quat


def main():
    action_mode = JointPositionActionMode(
        absolute=True, floating_base=True, floating_dofs=[]
    )
    env = RotateValve(render_mode="rgb_array", action_mode=action_mode, robot_cls=None)
    env.mojo.model.vis.global_.offwidth = 1280
    env.mojo.model.vis.global_.offheight = 720
    env.reset()

    ext_id = env._cameras_map["external"][0]
    m, d = env._mojo.model, env._mojo.data
    # Widen FOV uniformly so the candidates are comparable.
    m.cam_fovy[ext_id] = FOVY

    print(f"external cam id = {ext_id}, using fovy={FOVY}")
    print(f"rendering {len(CANDIDATES)} candidates → {OUT}")

    for dist, height in CANDIDATES:
        pos = np.array([-float(dist), 0.0, float(height)])
        x_ax, y_ax, quat = pose_looking_at(pos, TARGET)

        m.cam_pos[ext_id] = pos
        m.cam_quat[ext_id] = quat
        mujoco.mj_forward(m, d)

        frame = env.render()  # composite: external main + wrist overlays
        name = f"ext_d{dist:.2f}_h{height:.2f}"
        Image.fromarray(frame).save(OUT / f"{name}.png")
        xy = " ".join(f"{v:.3f}" for v in [*x_ax, *y_ax])
        print(
            f"  {name}: pos=\"{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}\" "
            f"xyaxes=\"{xy}\" fovy=\"{FOVY:g}\""
        )


if __name__ == "__main__":
    main()
