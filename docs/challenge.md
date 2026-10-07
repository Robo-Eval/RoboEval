# CoRL 2026 RoboEval Diagnostic Bimanual Challenge

Participant-facing tooling for Track 2 of the CoRL 2026 Embodied AI Benchmark
and Sim-to-Real Challenge Suite. The rules and timeline live on the website:
<https://robo-eval.github.io/challenge.html>. This page covers the commands.

## Install

```bash
git clone https://github.com/Robo-Eval/RoboEval.git
cd RoboEval
git checkout corl2026-challenge          # release tag, created at launch
git submodule update --init

conda create -n roboeval python=3.10 -y
conda activate roboeval
pip install -e ".[examples]"

export MUJOCO_GL=egl                     # on machines without a display
```

Check the install (no display needed, about a minute):

```bash
python -m roboeval.challenge.evaluate --submission examples/challenge_submission \
    --tasks LiftPot/Static --episodes 2 --output-dir /tmp/roboeval_check --overwrite
```

## Workflow

```bash
cp -r examples/challenge_submission ~/submission
# 1. edit ~/submission/metadata.yaml        team, method, interface
# 2. put weights in ~/submission/checkpoint/
# 3. implement ~/submission/policy/policy.py

# smoke test
python -m roboeval.challenge.evaluate --submission ~/submission \
    --tasks LiftPot/Static --episodes 2 --output-dir /tmp/smoke --overwrite

# full run: every variation, official seeds -> logs/ and results.json
python -m roboeval.challenge.evaluate --submission ~/submission --workers 16 --gpus 0,1

# 4. add ~/submission/report.pdf, then
python -m roboeval.challenge.package ~/submission
```

`package` validates the bundle (including replaying 10 logged episodes),
writes `MANIFEST.sha256`, and builds `<team>_<method>.tar.gz`. Upload the
archive and paste its link and printed SHA-256 into the submission form.

## Policy interface

`policy/policy.py` defines `make_policy(checkpoint_dir, metadata)` and returns
an object with:

- `reset(episode)`: called before every episode. `episode` has `family`,
  `variation`, `env_name`, `instruction`, `episode_index`, `seed`,
  `policy_seed`, `control_frequency`, `max_steps`, and `action_space`.
- `act(observation) -> np.ndarray`: one action of shape
  `episode.action_space.shape`. Observation keys are `proprioception`,
  `proprioception_grippers`, and `rgb_<camera>` / `depth_<camera>` for each
  declared camera.

The environment is built from the `interface` block of `metadata.yaml`
(action mode, control frequency, cameras), so the declared interface is the one
that runs. The template policy in `examples/challenge_submission/` holds still;
it scores zero but exercises the whole pipeline.

## What the tools check

| | |
|---|---|
| `evaluate` | Builds each env from `metadata.yaml`, resets with the official seeds, runs until success, the task's failure condition, an unhealthy simulator state, or the step budget. Writes one JSON and one `.npz` (the actions sent) per episode, then aggregates `results.json` from the logs. Records the git commit, a fingerprint of the benchmark source, and library versions. |
| `validate` | Layout, `metadata.yaml`, `results.json` coverage (all variations, all 14 metrics, official seeds), exact re-derivation of `results.json` from `logs/`, action-file checksums, a clean source tree, and with `--replay N` a replay of N logged episodes in the simulator. Exit status 0 means the bundle passes. |
| `package` | Runs `validate`, then writes the manifest and archive. Refuses invalid bundles. `--exclude-checkpoint` leaves weights out when `checkpoint_url` is set. |

Replay is exact only in the same floating-point environment. The three
commands pin it themselves (`roboeval/challenge/determinism.py`: one OpenBLAS
kernel, one thread) and record it in `results.json`, so evaluate on x86-64
Linux with AVX2 and the pip NumPy wheel. Bundles from other machines (e.g.
Apple silicon) go to organizer review instead of being accepted automatically.

Organizers run the same `validate` on receipt, and re-run prize-eligible
bundles with `evaluate --seeds-file <held-out seeds>`.

## Metric definitions

Leaderboard keys map onto the raw values written by `roboeval_metrics`:

| key | raw value | notes |
|---|---|---|
| `success` | `success` | 1 if the task's success check passed |
| `tp` | `subtask_progress` | completed stages / number of stages |
| `cpl` | `total_cartesian_path_length` | wrist path length, summed over both arms (m) |
| `jpl` | `total_joint_path_length` | per-step L2 joint displacement, summed over both arms (rad) |
| `opl` | `total_orientation_path_length` | wrist rotation angle, summed over both arms (rad) |
| `ct` | steps / control frequency | simulated seconds; the raw `completion_time` is host wall-clock and is not used |
| `tl` | steps | control steps executed |
| `bavd` | `bimanual_arm_velocity_difference` | mean abs. difference of the arms' joint-speed norms (rad/s) |
| `bgvd` | `bimanual_gripper_vertical_difference` | mean abs. wrist height difference (m) |
| `scc` / `ecc` | `self_collision_count` / `env_collision_count` | new contact pairs, sampled every 20 control steps |
| `sc` | `slip_count` | grasp lost without the gripper opening, sampled every 20 control steps |
| `mjj` / `mcj` | `overall_avg_joint_jerk` / `overall_avg_cartesian_jerk` | mean jerk magnitude by finite differences at the control rate |

## Evaluation cost

The step budget is fixed in simulated seconds per family (`roboeval/challenge/spec.py`),
so a policy that never succeeds runs the whole budget: about 560k control steps at
20 Hz over the suite. On an H100 host, one env step costs about 6 ms with
proprioception only and about 16 ms more per camera, roughly independent of
resolution. Policy inference adds to both.
