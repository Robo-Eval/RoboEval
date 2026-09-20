"""Run RoboEval's own examples/8_replay_to_lerobot.py, but with the per-joint
clamp replaced by the goal-tracking uniform-scale limiter.

Everything else -- decimation, ee_delta conversion, 5-camera rendering, LeRobot
writing, success check -- is the project's unmodified pipeline, so the output is
a real LeRobot dataset directly comparable to roboeval_ee_delta_20hz.

Writes under $HF_HOME/lerobot/<repo_id>/<task>. Never touches existing data:
repo_id must be new.

usage: render_worker.py <Variation> <repo_id> <manifest_out> [amount]
"""
import sys, runpy, inspect, textwrap
import numpy as np

VAR = sys.argv[1]
REPO = sys.argv[2]
OUT = sys.argv[3]
AMOUNT = sys.argv[4] if len(sys.argv) > 4 else "-1"
SHARD = int(sys.argv[5]) if len(sys.argv) > 5 else 0
NSHARDS = int(sys.argv[6]) if len(sys.argv) > 6 else 1

import roboeval.action_modes as AM
from roboeval.action_modes import JointPositionActionMode


def _un4(t):
    return "\n".join(l[4:] if l.startswith("    ") else l for l in t.split("\n"))


OLD_REF = """                joint_positions = self._robot.get_initial_joint_positions()"""
NEW_REF = """                joint_positions = (list(GOAL['g']) if GOAL['g'] is not None
                                   else self._robot.get_initial_joint_positions())"""

OLD_BLOCK = """        for i, actuator in enumerate(self._robot.limb_actuators):
            actuator = self._mojo.physics.bind(actuator)
            if self.absolute or self.ee:
                delta = action[i] - actuator.ctrl
                clamped_delta = np.clip(delta, -max_joint_delta, max_joint_delta)
                actuator.ctrl = actuator.ctrl + clamped_delta
            else:
                actuator.ctrl = actuator.ctrl + action[i]"""

NEW_BLOCK = """        _binds = [self._mojo.physics.bind(a) for a in self._robot.limb_actuators]
        if self.absolute or self.ee:
            _cur = np.array([b.ctrl for b in _binds], dtype=float).ravel()
            _des = np.asarray(action[:len(_binds)], dtype=float).ravel()
            _d = _des - _cur
            _m = float(np.abs(_d).max()) if _d.size else 0.0
            GOAL['g'] = _des.copy()
            _s = 1.0 if (_m <= max_joint_delta or _m == 0.0) else max_joint_delta / _m
            INFO['sat'] = bool(_s < 1.0)
            INFO['n'] += int(_s < 1.0)
            _new = _cur + _d * _s
            for _b, _v in zip(_binds, _new):
                _b.ctrl = _v
        else:
            for i, _b in enumerate(_binds):
                _b.ctrl = _b.ctrl + action[i]"""

src = textwrap.dedent(inspect.getsource(JointPositionActionMode.step))
for o, n in ((OLD_REF, NEW_REF), (OLD_BLOCK, NEW_BLOCK)):
    a, b = _un4(o), _un4(n)
    assert a in src, "source pattern not found - cannot patch limiter"
    src = src.replace(a, b)

ns = dict(AM.__dict__)
ns.update(INFO={"sat": False, "n": 0}, GOAL={"g": None}, np=np)
exec(src, ns)
JointPositionActionMode.step = ns["step"]

# The goal reference is per-episode: clear it on every env reset, otherwise it
# leaks across demos and across the two envs (this is what standalone runs did
# explicitly).
GOAL_REF = ns["GOAL"]
from roboeval.roboeval_env import RoboEvalEnv
_orig_reset = RoboEvalEnv.reset


def _reset_with_goal(self, *args, **kwargs):
    GOAL_REF["g"] = None
    return _orig_reset(self, *args, **kwargs)


RoboEvalEnv.reset = _reset_with_goal

# Compatibility shim: the script targets an older lerobot where add_frame took
# `task=` separately; lerobot 0.4.4 expects it inside the frame dict.
from lerobot.datasets.lerobot_dataset import LeRobotDataset as _LRD
import inspect as _insp
if "task" not in _insp.signature(_LRD.add_frame).parameters:
    _orig_add = _LRD.add_frame

    def _add_frame_comp(self, frame, task=None, **kw):
        if task is not None:
            frame = dict(frame)
            frame["task"] = task
        return _orig_add(self, frame, **kw)

    _LRD.add_frame = _add_frame_comp
    print("[compat] add_frame(task=) shim installed for lerobot 0.4.4", flush=True)
from roboeval.demonstrations.demo_store import DemoStore as _DS
_orig_list = _DS.list_demo_paths


def _sharded_list(self, metadata):
    paths = [p for p in _orig_list(self, metadata) if p.suffix == ".safetensors"]
    return sorted(paths)[SHARD::NSHARDS]


if NSHARDS > 1:
    _DS.list_demo_paths = _sharded_list


# --- joint-space subdivision -------------------------------------------------
# The velocity limit is |dq_i| <= MAX_JOINT_VEL*dt, a joint-space condition, so
# feasibility is established in joint space BEFORE observations are regenerated.
# Hooked onto create_demo_in_new_env so the conversion script needs no changes.
from copy import deepcopy as _dc
from roboeval.demonstrations.demo import Demo as _Demo
from roboeval.demonstrations.demo_converter import DemoConverter as _DC

_orig_cdine = _DC.create_demo_in_new_env.__func__ if hasattr(_DC.create_demo_in_new_env, "__func__") else _DC.create_demo_in_new_env


def _subdivide(demo, n_limb, bound):
    ts = demo.timesteps
    if len(ts) < 2 or bound <= 0:
        return demo
    out = [_dc(ts[0])]
    prev = np.asarray(ts[0].executed_action, dtype=float)
    for step in ts[1:]:
        tgt = np.asarray(step.executed_action, dtype=float)
        d = tgt[:n_limb] - prev[:n_limb]
        k = max(1, int(np.ceil(np.abs(d).max() / bound)))
        for j in range(1, k + 1):
            a = prev + (tgt - prev) * (j / k)
            a[n_limb:] = prev[n_limb:] if j < k else tgt[n_limb:]
            nts = _dc(step)
            nts.set_executed_action(a)
            out.append(nts)
        prev = tgt
    return _Demo(demo.metadata, out)


def _cdine_subdiv(demo, env):
    am = env.action_mode
    dt = (am._sub_steps_count or 1) * env._mojo.physics.model.opt.timestep
    bound = am.MAX_JOINT_VEL * dt
    n_limb = len(env._robot.limb_actuators)
    return _orig_cdine(_subdivide(demo, n_limb, bound), env)


_DC.create_demo_in_new_env = staticmethod(_cdine_subdiv)
print(f"[{VAR}] joint-space subdivision installed", flush=True)

print(f"[{VAR}] shard {SHARD}/{NSHARDS}; limiter installed; starting render", flush=True)

sys.argv = [
    "8_replay_to_lerobot.py",
    "--ee", "--delta",
    "--freq", "20",
    "--tasks", VAR,
    "--repo_id", REPO,
    "--output", OUT,
    "--amount", AMOUNT,
]
runpy.run_path("/root/RoboEval-git/examples/8_replay_to_lerobot.py", run_name="__main__")
print(f"[{VAR}] RENDER_DONE", flush=True)
