# `roboeval_ee_delta_20hz_scaled`

A LeRobot dataset built from the released RoboEval demonstrations, using
joint-velocity **scaling** rather than per-joint clipping when limb targets are
applied.

HuggingFace: `helen9975/roboeval_ee_delta_20hz_scaled` (private)

## What is in it

| | |
|---|---|
| Format | LeRobot v3.0 |
| Robot | `panda` (bimanual) |
| Control frequency | 20 Hz |
| Action space | end-effector delta, 14-D (6 pose + 1 gripper per arm) |
| Task variations | 30 (8 task families) |
| Source | `helen9975/RoboEval_data` v1.0.0 demos, 500 Hz |

Per-frame features:

| Key | Shape | Notes |
|---|---|---|
| `cam_external`, `cam_front`, `cam_head`, `cam_left_wrist`, `cam_right_wrist` | 256×256×3 | RGB |
| `state` | 36 | full proprioception |
| `state_gripper` | 2 | gripper state |
| `actions` | 14 | bimanual end-effector delta |
| `reward`, `is_terminal`, `truncate` | 1 | per step |
| `uuid` | 1 | source demonstration id |

Each episode is a demonstration that was replayed at 20 Hz and reached
`reward > 0` under the task's success condition. Demonstrations whose replay did
not succeed are not included. `uuid` maps every episode back to its source file
in the released demo archive.

## Control mode

`examples/8_replay_to_lerobot.py` limits each limb joint independently:

```python
delta = action[i] - actuator.ctrl
actuator.ctrl += np.clip(delta, -max_joint_delta, max_joint_delta)
```

Limiting joints separately changes their relative magnitudes, so the arm follows
a different joint-space direction on any step where one joint is limited.

This dataset is generated with `examples/11_replay_to_lerobot_scaled.py`, which
instead tracks a joint-space goal that advances by the full commanded delta and
moves `ctrl` toward it with a single scale factor:

```python
goal = desired                              # advances regardless of the limit
d    = goal - ctrl
s    = min(1, max_joint_delta / max(|d|))   # one factor for all joints
ctrl = ctrl + d * s                         # remainder stays in (goal - ctrl)
```

Direction is preserved, and any unapplied remainder is carried into subsequent
steps instead of being dropped.

Limb targets are additionally subdivided in joint space until every step
satisfies ``|dq_i| <= MAX_JOINT_VEL * control_dt``. The bound is a joint-space
condition, so feasibility is established there rather than in the end-effector
action space, where the Jacobian makes small Cartesian steps arbitrarily
expensive near singularities. Trajectories become longer (about 1.1x-1.7x
depending on the task) and every frame carries a real action. With subdivision
applied the limiter does not engage during replay. The dataset schema is
unchanged.

## Episode counts

<!--COUNTS-->

## Regenerating

### Environment

Python 3.11, on local disk (not a network mount — imports are slow over one):

```bash
git clone --recurse-submodules https://github.com/Robo-Eval/RoboEval.git
cd RoboEval

uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python \
  "numpy==1.26.*" "mujoco==3.3.3" "dm_control==1.0.31" safetensors \
  imageio pyquaternion mujoco_utils wget pyyaml "hydra-core==1.3.*" \
  psutil lerobot
uv pip install --python .venv/bin/python \
  "gymnasium @ git+https://github.com/stepjam/Gymnasium.git@0.29.2" \
  "mojo @ git+https://github.com/helen9975/mojo.git" \
  "roboeval-metrics @ git+https://github.com/Robo-Eval/roboeval-metrics.git"

export MUJOCO_GL=egl
export PYTHONPATH=$PWD
```

The submodule (`thirdparty/mujoco_menagerie`) supplies the arm meshes; without
`--recurse-submodules` every environment fails to construct.

### Demonstrations

```bash
python examples/0_download_data.py
```

Downloads and extracts the 216 MB archive to `~/.roboeval/roboeval_demos/1.0.0/`
(4155 files across 33 task directories).

### Conversion

```bash
export HF_HOME=/path/for/output
python examples/11_replay_to_lerobot_scaled.py --repo_id roboeval_ee_delta_20hz_scaled
```

Writes one dataset per task variation under
`$HF_HOME/lerobot/roboeval_ee_delta_20hz_scaled/`. One variation at a time can be
selected with `--tasks`, which parallelises well — one process per variation.

Rendering is GPU-bound. Workers default to EGL device 0, so on a multi-GPU node
set `MUJOCO_EGL_DEVICE_ID` per worker to spread the load; otherwise one GPU
saturates while the others idle.

Note: `examples/8_replay_to_lerobot.py` calls `add_frame(frame, task=...)`, which
newer lerobot releases do not accept. `11_replay_to_lerobot_scaled.py` adapts the
call when needed.

### Merging

The per-variation datasets can be merged into one with
`lerobot-edit-dataset --operation.type merge`.
