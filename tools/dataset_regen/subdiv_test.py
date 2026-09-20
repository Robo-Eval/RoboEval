"""Joint-space subdivision + goal-tracking limiter.

The velocity limit is a joint-space condition (|dq_i| <= MAX_JOINT_VEL*dt), so
feasibility has to be established in joint space. Subdivide the decimated
joint-absolute trajectory until every consecutive pair respects the bound, THEN
regenerate observations and convert to ee_delta.

If it works, the limiter should never engage during replay -- and a replay where
the limiter never engages is bit-identical to the unlimited one, so outcomes are
preserved by construction rather than by luck.

Modes:
  none   : no limit                      (reference outcomes)
  goal   : goal-tracking limiter         (no subdivision)
  subdiv : goal-tracking + joint subdivision
"""
import glob, sys, logging, time, inspect, textwrap
from copy import deepcopy
import numpy as np

logging.disable(logging.WARNING)
DEMOS = "/root/.roboeval/roboeval_demos/1.0.0/BimanualPanda"
FREQ, SRC_FREQ = 20, 500
VAR = sys.argv[1] if len(sys.argv) > 1 else "RotateValve"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12

import roboeval.action_modes as AM
from roboeval.action_modes import JointPositionActionMode
from roboeval.robots.configs.panda import BimanualPanda
from roboeval.demonstrations.demo import Demo
from roboeval.demonstrations.demo_converter import DemoConverter
import roboeval.envs.lift_tray as lt, roboeval.envs.manipulation as mani, roboeval.envs.lift_pot as lp
import roboeval.envs.stack_books as sb, roboeval.envs.pack_objects as po, roboeval.envs.rotate_utility_objects as rv

REG = {}
for m in (lt, mani, lp, sb, po, rv):
    REG.update({k: getattr(m, k) for k in dir(m)})
cls = REG[VAR]
REAL_VEL = JointPositionActionMode.MAX_JOINT_VEL


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
            if MODE[0] == 'none':
                _new = _des
            else:
                GOAL['g'] = _des.copy()
                _s = 1.0 if (_m <= max_joint_delta or _m == 0.0) else max_joint_delta / _m
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
    assert a in src, "pattern not found"
    src = src.replace(a, b)

ns = dict(AM.__dict__)
INFO = {"n": 0}
MODE = ["none"]
GOAL = {"g": None}
ns.update(INFO=INFO, MODE=MODE, GOAL=GOAL, np=np)
exec(src, ns)
JointPositionActionMode.step = ns["step"]
print("patched step(): goal-tracking limiter installed", flush=True)

abs_env = cls(action_mode=JointPositionActionMode(floating_base=True, absolute=True, floating_dofs=[]),
              render_mode=None, control_frequency=FREQ, robot_cls=BimanualPanda)
tgt_env = cls(action_mode=JointPositionActionMode(floating_base=True, absolute=False, ee=True, floating_dofs=[]),
              render_mode=None, control_frequency=FREQ, robot_cls=BimanualPanda)
N_LIMB = len(abs_env._robot.limb_actuators)
CTRL_DT = (abs_env.action_mode._sub_steps_count or 1) * abs_env._mojo.physics.model.opt.timestep
BOUND = REAL_VEL * CTRL_DT
print(f"=== {VAR}: bound {BOUND:.5f} rad/step, {N_LIMB} limb joints ===", flush=True)


def subdivide(demo):
    """Insert waypoints so every consecutive |dq| <= BOUND (joint space)."""
    ts = demo.timesteps
    if len(ts) < 2:
        return demo, 1.0
    out = [deepcopy(ts[0])]
    prev = np.asarray(ts[0].executed_action, dtype=float)
    for step in ts[1:]:
        tgt = np.asarray(step.executed_action, dtype=float)
        d = tgt[:N_LIMB] - prev[:N_LIMB]
        k = max(1, int(np.ceil(np.abs(d).max() / BOUND))) if BOUND > 0 else 1
        for j in range(1, k + 1):
            a = prev + (tgt - prev) * (j / k)
            a[N_LIMB:] = prev[N_LIMB:] if j < k else tgt[N_LIMB:]
            nts = deepcopy(step)
            nts.set_executed_action(a)
            out.append(nts)
        prev = tgt
    return Demo(demo.metadata, out), len(out) / len(ts)


def run(path, mode):
    JointPositionActionMode.MAX_JOINT_VEL = 1e9 if mode == "none" else REAL_VEL
    MODE[0] = "none" if mode == "none" else "goal"
    GOAL["g"] = None
    INFO["n"] = 0

    d = Demo.from_safetensors(path)
    d = DemoConverter.decimate(d, FREQ, SRC_FREQ, robot=d.metadata.get_robot())
    ratio = 1.0
    if mode == "subdiv":
        d, ratio = subdivide(d)
    d = DemoConverter.create_demo_in_new_env(d, abs_env)
    conv = DemoConverter.joint_absolute_to_ee_delta(d)

    tgt_env.reset(seed=conv.seed)
    GOAL["g"] = None
    INFO["n"] = 0
    ok = False
    for tstep in conv.timesteps:
        tgt_env.step(np.asarray(tstep.executed_action, dtype=float), fast=True)
        if tgt_env.reward > 0:
            ok = True
            break
    return ok, INFO["n"], ratio, len(conv.timesteps)


paths = sorted(glob.glob(f"{DEMOS}/{VAR}/**/*.safetensors", recursive=True))[:N]
res, sat, rat = {}, {}, {}
t0 = time.time()
for m in ("none", "goal", "subdiv"):
    res[m], sat[m], rat[m] = [], [], []
for i, p in enumerate(paths):
    for m in ("none", "goal", "subdiv"):
        ok, ns_, ratio, L = run(p, m)
        res[m].append(ok); sat[m].append(ns_); rat[m].append(ratio)
    if (i + 1) % 3 == 0:
        print(f"  {i+1}/{len(paths)} ({time.time()-t0:.0f}s)", flush=True)

ref = res["none"]
print()
for m in ("none", "goal", "subdiv"):
    n = sum(res[m])
    agree = sum(1 for a, b in zip(res[m], ref) if a == b)
    tag = "reference" if m == "none" else f"agrees {agree}/{len(ref)}"
    extra = "" if m == "none" else f"  limiter engaged {np.mean(sat[m]):.0f} steps/ep"
    length = f"  length x{np.mean(rat[m]):.2f}" if m == "subdiv" else ""
    print(f"{m:7s}: success {n}/{len(paths)}   {tag}{extra}{length}")
print("\nSUBDIV_DONE", flush=True)
