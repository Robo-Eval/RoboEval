"""Batch-render the composite view with candidate eagle-view poses.

Like `calibrate_external.py` but sweeps the space of steep-downtilt
top-down cameras: raised 1.8–3.0 m, offset -0.6 to +0.4 m in X, all
aimed at the same workspace target. Writes one PNG per candidate to
/tmp/playground_eagle_calibrate/ and prints the world.xml snippet.
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

OUT = Path("/tmp/playground_eagle_calibrate")
OUT.mkdir(exist_ok=True)

TARGET = np.array([0.40, 0.0, 0.80])

# Eagle-view sweep: X-offset (negative = behind robot, positive = in front)
# × height. Candidates near the steep end of the previous sweep plus a
# range above it.
_X_OFFSETS = [-0.60, -0.40, -0.20, 0.00, 0.20, 0.40]
_HEIGHTS = [1.80, 2.10, 2.40, 2.70, 3.00]
CANDIDATES = [(x, h) for x, h in product(_X_OFFSETS, _HEIGHTS)]

FOVY = 60.0


def pose_looking_at(pos, target):
    """Build (x_axis, y_axis, quat) for a camera at pos looking at target.

    Same convention as calibrate_external.py: cam +X locked to world -Y
    so the robot's left arm stays on the left of the rendered frame.
    """
    look_dir = target - pos
    look_dir /= np.linalg.norm(look_dir)

    z_axis = -look_dir
    x_axis = np.array([0.0, -1.0, 0.0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
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
    m.cam_fovy[ext_id] = FOVY

    print(f"external cam id = {ext_id}, using fovy={FOVY}")
    print(f"rendering {len(CANDIDATES)} eagle candidates → {OUT}")

    for x_off, height in CANDIDATES:
        pos = np.array([float(x_off), 0.0, float(height)])
        x_ax, y_ax, quat = pose_looking_at(pos, TARGET)

        m.cam_pos[ext_id] = pos
        m.cam_quat[ext_id] = quat
        mujoco.mj_forward(m, d)

        frame = env.render()
        name = f"eagle_x{x_off:+.2f}_h{height:.2f}"
        Image.fromarray(frame).save(OUT / f"{name}.png")
        xy = " ".join(f"{v:.3f}" for v in [*x_ax, *y_ax])
        print(
            f"  {name}: pos=\"{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}\" "
            f"xyaxes=\"{xy}\" fovy=\"{FOVY:g}\""
        )


if __name__ == "__main__":
    main()
