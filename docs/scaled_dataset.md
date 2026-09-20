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

Camera streams are stored as `image` features rather than encoded video, as in
`roboeval_ee_delta_20hz`, so the dataset is around 95 GB on disk. Five cameras
are kept; training runs that use a subset select them at load time.

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

| Task variation | Demonstrations | Episodes | Frames |
|---|---:|---:|---:|
| `CubeHandover` | 104 | 95 | 7,770 |
| `CubeHandoverOrientation` | 107 | 106 | 8,684 |
| `CubeHandoverPosition` | 109 | 104 | 9,790 |
| `CubeHandoverPositionAndOrientation` | 105 | 101 | 15,092 |
| `LiftPot` | 104 | 95 | 7,400 |
| `LiftPotOrientation` | 115 | 95 | 8,722 |
| `LiftPotPosition` | 221 | 154 | 13,616 |
| `LiftPotPositionAndOrientation` | 102 | 77 | 9,445 |
| `LiftTray` | 100 | 35 | 3,384 |
| `LiftTrayOrientation` | 104 | 76 | 8,392 |
| `LiftTrayPosition` | 144 | 135 | 13,713 |
| `LiftTrayPositionAndOrientation` | 204 | 154 | 21,447 |
| `PackBox` | 174 | 130 | 28,928 |
| `PackBoxOrientation` | 184 | 98 | 17,930 |
| `PackBoxPosition` | 115 | 96 | 14,451 |
| `PackBoxPositionAndOrientation` | 115 | 99 | 16,422 |
| `PickSingleBookFromTable` | 118 | 98 | 13,685 |
| `PickSingleBookFromTableOrientation` | 100 | 85 | 9,729 |
| `PickSingleBookFromTablePosition` | 116 | 96 | 14,395 |
| `PickSingleBookFromTablePositionAndOrientation` | 104 | 92 | 10,944 |
| `RotateValve` | 121 | 112 | 7,960 |
| `RotateValvePosition` | 113 | 110 | 23,903 |
| `RotateValvePositionAndOrientation` | 125 | 115 | 20,328 |
| `StackSingleBookShelf` | 103 | 62 | 19,219 |
| `StackSingleBookShelfPosition` | 101 | 75 | 12,644 |
| `StackSingleBookShelfPositionAndOrientation` | 100 | 73 | 13,660 |
| `StackTwoBlocks` | 203 | 182 | 27,998 |
| `StackTwoBlocksOrientation` | 100 | 78 | 7,041 |
| `StackTwoBlocksPosition` | 104 | 91 | 10,383 |
| `StackTwoBlocksPositionAndOrientation` | 197 | 141 | 14,083 |
| **Total** | **3812** | **3060** | **411,158** |

3060 of 3812 demonstrations replay successfully at 20 Hz under these settings.
Retention varies by task, from 35% (`LiftTray`) to 99% (`CubeHandoverOrientation`).


## Regenerating

### Environment

Python 3.11, on local disk (not a network mount — imports are slow over one):

```bash
git clone --recurse-submodules https://github.com/Robo-Eval/RoboEval.git
cd RoboEval

uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python \
  "mujoco==3.3.3" "dm_control==1.0.31" safetensors \
  imageio pyquaternion mujoco_utils wget pyyaml "hydra-core==1.3.*" \
  psutil "lerobot==0.3.3"
uv pip install --python .venv/bin/python \
  "gymnasium @ git+https://github.com/stepjam/Gymnasium.git@0.29.2" \
  "mojo @ git+https://github.com/helen9975/mojo.git" \
  "roboeval-metrics @ git+https://github.com/Robo-Eval/roboeval-metrics.git"
# Install numpy last: lerobot pulls numpy 2.x, which this pipeline does not
# support ("only 0-dimensional arrays can be converted to Python scalars").
uv pip install --python .venv/bin/python "numpy==1.26.*"

export MUJOCO_GL=egl
export PYTHONPATH=$PWD
```

The submodule (`thirdparty/mujoco_menagerie`) supplies the arm meshes; without
`--recurse-submodules` every environment fails to construct.

### Demonstrations

```bash
python -c "from roboeval.demonstrations.demo_store import DemoStore; \
s = DemoStore(); s.cached or s.pull_demos()"
```

Downloads and extracts the 216 MB archive to `~/.roboeval/roboeval_demos/1.0.0/`
(4155 files across 33 task directories), about 10 seconds.

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

### Converting and merging

`lerobot==0.3.3` writes v2.1 datasets and has no conversion or merge tooling;
`lerobot==0.4.4` provides both but pulls a numpy version the conversion cannot
use. Keep them in separate environments — the merge environment only needs
lerobot, never `roboeval`:

```bash
uv venv --python 3.11 .venv-merge
uv pip install --python .venv-merge/bin/python "lerobot==0.4.4"

# v2.1 -> v3.0, per dataset
.venv-merge/bin/python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
  --repo-id <TaskName> --root $HF_HOME/lerobot/<repo_id> \
  --push-to-hub=false --force-conversion

# optional: combine datasets
.venv-merge/bin/lerobot-edit-dataset --repo_id <out> \
  --operation.type merge --operation.repo_ids "['<a>', '<b>']"
```

Render each variation in a single process. Splitting one variation across
processes and converting the parts can produce data files numbered from 1 while
the episode metadata references 0, which makes the merge fail; after any
conversion, check that `data/chunk-000/file-000.parquet` exists and that
`meta/info.json` reports the expected episode count.
