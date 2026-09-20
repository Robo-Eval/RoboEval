#!/usr/bin/env python3
"""Convert RoboEval demos to LeRobot with joint-velocity-scaled control.

Same pipeline as ``8_replay_to_lerobot.py`` (decimate -> convert -> replay ->
write), with one difference in how limb targets are applied each control step.

Default behaviour (``8_replay_to_lerobot.py``) clips each joint independently to
``MAX_JOINT_VEL * control_dt``.  Clipping joints separately changes the ratio
between them, so the arm follows a different joint-space direction whenever any
single joint is limited.

This script instead:

* keeps an explicit joint-space goal that advances by the full commanded delta,
  independently of what the limiter applies;
* moves ``ctrl`` toward that goal by scaling the whole delta vector with a
  single factor ``s = min(1, bound / max|delta|)``, which preserves direction
  and only shortens the step;
* leaves any unapplied remainder in ``goal - ctrl``, so it is carried into
  following steps rather than dropped.

It also subdivides limb targets in joint space so every step respects
``MAX_JOINT_VEL * control_dt``.  Trajectories become longer (about 1.1x-1.7x
depending on the task) and every frame carries a real action; with subdivision
applied the limiter does not engage during replay.  The dataset schema is
unchanged.  Demonstrations are still written only when the replay reaches
``reward > 0``.

Usage
-----
    # one task variation
    python examples/11_replay_to_lerobot_scaled.py --tasks LiftPot

    # all variations, custom repo id
    python examples/11_replay_to_lerobot_scaled.py --repo_id roboeval_ee_delta_20hz_scaled

Output goes to ``$HF_HOME/lerobot/<repo_id>/<TaskName>``.

Requires the same environment as ``8_replay_to_lerobot.py``; see
``docs/scaled_dataset.md`` for a from-scratch setup.
"""
from __future__ import annotations

import argparse
import inspect
import runpy
import sys
import textwrap

import numpy as np

import roboeval.action_modes as action_modes_module
from roboeval.action_modes import JointPositionActionMode
from roboeval.roboeval_env import RoboEvalEnv

SCRIPT = "8_replay_to_lerobot.py"

# --- the two edits applied to JointPositionActionMode.step -------------------

_OLD_REFERENCE = """                joint_positions = self._robot.get_initial_joint_positions()"""

_NEW_REFERENCE = """                joint_positions = (list(GOAL['g']) if GOAL['g'] is not None
                                   else self._robot.get_initial_joint_positions())"""

_OLD_LIMIT = """        for i, actuator in enumerate(self._robot.limb_actuators):
            actuator = self._mojo.physics.bind(actuator)
            if self.absolute or self.ee:
                delta = action[i] - actuator.ctrl
                clamped_delta = np.clip(delta, -max_joint_delta, max_joint_delta)
                actuator.ctrl = actuator.ctrl + clamped_delta
            else:
                actuator.ctrl = actuator.ctrl + action[i]"""

_NEW_LIMIT = """        _binds = [self._mojo.physics.bind(a) for a in self._robot.limb_actuators]
        if self.absolute or self.ee:
            _cur = np.array([b.ctrl for b in _binds], dtype=float).ravel()
            _des = np.asarray(action[:len(_binds)], dtype=float).ravel()
            _d = _des - _cur
            _m = float(np.abs(_d).max()) if _d.size else 0.0
            GOAL['g'] = _des.copy()
            _s = 1.0 if (_m <= max_joint_delta or _m == 0.0) else max_joint_delta / _m
            _new = _cur + _d * _s
            for _b, _v in zip(_binds, _new):
                _b.ctrl = _v
        else:
            for i, _b in enumerate(_binds):
                _b.ctrl = _b.ctrl + action[i]"""


def _strip_one_indent(text: str) -> str:
    return "\n".join(l[4:] if l.startswith("    ") else l for l in text.split("\n"))


def install_scaled_limiter() -> dict:
    """Replace per-joint clipping with goal-tracking uniform scaling.

    Returns the goal holder so it can be cleared between episodes.
    """
    source = textwrap.dedent(inspect.getsource(JointPositionActionMode.step))
    for old, new in ((_OLD_REFERENCE, _NEW_REFERENCE), (_OLD_LIMIT, _NEW_LIMIT)):
        old, new = _strip_one_indent(old), _strip_one_indent(new)
        if old not in source:
            raise RuntimeError(
                "JointPositionActionMode.step does not match the expected source; "
                "update examples/11_replay_to_lerobot_scaled.py"
            )
        source = source.replace(old, new)

    goal = {"g": None}
    namespace = dict(action_modes_module.__dict__)
    namespace.update(GOAL=goal, np=np)
    exec(source, namespace)  # noqa: S102 - rewriting a known method body
    JointPositionActionMode.step = namespace["step"]

    # The goal is per-episode: clear it whenever an environment resets.
    original_reset = RoboEvalEnv.reset

    def reset(self, *args, **kwargs):
        goal["g"] = None
        return original_reset(self, *args, **kwargs)

    RoboEvalEnv.reset = reset
    return goal


# --- joint-space subdivision -------------------------------------------------

def install_joint_subdivision() -> None:
    """Subdivide limb targets so every step respects the joint-velocity bound.

    The bound is ``|dq_i| <= MAX_JOINT_VEL * control_dt``, a joint-space
    condition, so feasibility is established in joint space before observations
    are regenerated. Each segment is split into
    ``ceil(max|dq| / bound)`` linear waypoints; the gripper holds through a
    segment and switches on its final waypoint so grasp ordering is preserved.

    Trajectories get longer (roughly 1.1x-1.7x depending on the task) and every
    frame carries a real action. With this applied the limiter does not engage
    during replay.
    """
    from copy import deepcopy
    from roboeval.demonstrations.demo import Demo
    from roboeval.demonstrations.demo_converter import DemoConverter

    original = DemoConverter.create_demo_in_new_env
    original = getattr(original, "__func__", original)

    def subdivide(demo, n_limb, bound):
        steps = demo.timesteps
        if len(steps) < 2 or bound <= 0:
            return demo
        out = [deepcopy(steps[0])]
        prev = np.asarray(steps[0].executed_action, dtype=float)
        for step in steps[1:]:
            target = np.asarray(step.executed_action, dtype=float)
            delta = target[:n_limb] - prev[:n_limb]
            parts = max(1, int(np.ceil(np.abs(delta).max() / bound)))
            for j in range(1, parts + 1):
                action = prev + (target - prev) * (j / parts)
                action[n_limb:] = prev[n_limb:] if j < parts else target[n_limb:]
                waypoint = deepcopy(step)
                waypoint.set_executed_action(action)
                out.append(waypoint)
            prev = target
        return Demo(demo.metadata, out)

    def create_demo_in_new_env(demo, env):
        mode = env.action_mode
        control_dt = (mode._sub_steps_count or 1) * env._mojo.physics.model.opt.timestep
        bound = mode.MAX_JOINT_VEL * control_dt
        n_limb = len(env._robot.limb_actuators)
        return original(subdivide(demo, n_limb, bound), env)

    DemoConverter.create_demo_in_new_env = staticmethod(create_demo_in_new_env)


def install_lerobot_compat() -> None:
    """Accept ``add_frame(frame, task=...)`` on lerobot versions that dropped it."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if "task" in inspect.signature(LeRobotDataset.add_frame).parameters:
        return

    original_add_frame = LeRobotDataset.add_frame

    def add_frame(self, frame, task=None, **kwargs):
        if task is not None:
            frame = dict(frame)
            frame["task"] = task
        return original_add_frame(self, frame, **kwargs)

    LeRobotDataset.add_frame = add_frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Task class names (default: all)")
    parser.add_argument("--repo_id", default="roboeval_ee_delta_20hz_scaled",
                        help="LeRobot repo id (default: roboeval_ee_delta_20hz_scaled)")
    parser.add_argument("--freq", type=int, default=20, help="Control frequency (default: 20)")
    parser.add_argument("--amount", type=int, default=-1,
                        help="Max demos per task, -1 for all (default: -1)")
    parser.add_argument("--output", default="successful_demos_scaled.txt",
                        help="Where to write the list of converted demos")
    args = parser.parse_args()

    install_scaled_limiter()
    install_joint_subdivision()
    install_lerobot_compat()

    forwarded = [SCRIPT, "--ee", "--delta",
                 "--freq", str(args.freq),
                 "--repo_id", args.repo_id,
                 "--output", args.output,
                 "--amount", str(args.amount)]
    if args.tasks:
        forwarded += ["--tasks", *args.tasks]

    sys.argv = forwarded
    script_path = str((__import__("pathlib").Path(__file__).parent / SCRIPT).resolve())
    runpy.run_path(script_path, run_name="__main__")


if __name__ == "__main__":
    main()
