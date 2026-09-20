"""Goal-tracking rate limiter vs per-joint clipping.

Root cause being fixed: in ee_delta mode the env composes each delta onto
actuator.ctrl (robot.get_initial_joint_positions -> ctrl). When the clamp
truncates ctrl, the untraveled motion is discarded and never recovered, so
error accumulates across the episode.

Fix: keep an explicit unclamped reference `goal`.
    goal <- IK(FK(goal) (+) delta)     # advances regardless of clamping
    d    <- goal - ctrl                # full outstanding error incl. residual
    s    <- min(1, bound / max|d|)     # uniform scale -> direction preserved
    ctrl <- ctrl + d * s               # residual implicit in (goal - ctrl)
Zero-delta "hold" steps then let ctrl catch up to a stationary goal.

Modes: none (no limit, reference) | clip (current) | goal (this fix).
Read-only: writes no dataset.
"""
import glob, sys, logging, time, inspect, textwrap
import numpy as np

logging.disable(logging.WARNING)
DEMOS = "/root/.roboeval/roboeval_demos/1.0.0/BimanualPanda"
FREQ, SRC_FREQ = 20, 500
VAR = sys.argv[1] if len(sys.argv) > 1 else "CubeHandoverPositionAndOrientation"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
MAX_CATCHUP = int(sys.argv[3]) if len(sys.argv) > 3 else 25

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


# --- 1. compose the delta onto the unclamped goal, not onto ctrl -------------
OLD_REF = """                joint_positions = self._robot.get_initial_joint_positions()"""
NEW_REF = """                joint_positions = (list(GOAL['g']) if GOAL['g'] is not None
                                   else self._robot.get_initial_joint_positions())"""

# --- 2. rate-limit ctrl toward the goal, preserving direction ---------------
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
                INFO['sat'] = False
                _new = _des
            elif MODE[0] == 'clip':
                INFO['sat'] = bool(np.any(np.abs(_d) > max_joint_delta))
                INFO['n'] += int(INFO['sat'])
                _new = _cur + np.clip(_d, -max_joint_delta, max_joint_delta)
            else:
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
for old, new in ((OLD_REF, NEW_REF), (OLD_BLOCK, NEW_BLOCK)):
    o, n = _un4(old), _un4(new)
    assert o in src, f"source pattern not found:\n{o[:120]}"
    src = src.replace(o, n)

ns = dict(AM.__dict__)
INFO = {"sat": False, "n": 0}
MODE = ["clip"]
GOAL = {"g": None}
ns.update(INFO=INFO, MODE=MODE, GOAL=GOAL, np=np)
exec(src, ns)
JointPositionActionMode.step = ns["step"]
print("patched step(): goal-tracking rate limiter installed", flush=True)

abs_env = cls(action_mode=JointPositionActionMode(floating_base=True, absolute=True, floating_dofs=[]),
              render_mode=None, control_frequency=FREQ, robot_cls=BimanualPanda)
tgt_env = cls(action_mode=JointPositionActionMode(floating_base=True, absolute=False, ee=True, floating_dofs=[]),
              render_mode=None, control_frequency=FREQ, robot_cls=BimanualPanda)
N_GRIP = len(tgt_env._robot.grippers)


def run(path, mode):
    JointPositionActionMode.MAX_JOINT_VEL = 1e9 if mode == "none" else REAL_VEL
    MODE[0] = mode
    INFO.update(sat=False, n=0)

    GOAL["g"] = None                      # conversion stage: plain behaviour
    MODE[0] = "none" if mode == "none" else mode
    d = Demo.from_safetensors(path)
    robot = d.metadata.get_robot()
    d = DemoConverter.decimate(d, FREQ, SRC_FREQ, robot=robot)
    d = DemoConverter.create_demo_in_new_env(d, abs_env)
    conv = DemoConverter.joint_absolute_to_ee_delta(d)

    tgt_env.reset(seed=conv.seed)
    GOAL["g"] = None                      # reset reference for the replay
    INFO.update(sat=False, n=0)
    ok, extra = False, 0
    for ts in conv.timesteps:
        a = np.asarray(ts.executed_action, dtype=float)
        tgt_env.step(a, fast=True)
        if tgt_env.reward > 0:
            ok = True
            break
        if mode == "goal":
            hold = np.zeros_like(a)
            hold[-N_GRIP:] = a[-N_GRIP:]
            k = 0
            while INFO["sat"] and k < MAX_CATCHUP:
                tgt_env.step(hold, fast=True)
                extra += 1
                k += 1
                if tgt_env.reward > 0:
                    ok = True
                    break
            if ok:
                break
    return ok, INFO["n"], extra, len(conv.timesteps)


paths = sorted(glob.glob(f"{DEMOS}/{VAR}/**/*.safetensors", recursive=True))[:N]
print(f"=== {VAR}: {len(paths)} demos  (catchup cap {MAX_CATCHUP}) ===", flush=True)
res = {m: [] for m in ("none", "clip", "goal")}
inf = {m: [] for m in ("none", "clip", "goal")}
t0 = time.time()
for i, p in enumerate(paths):
    for m in ("none", "clip", "goal"):
        ok, nsat, extra, L = run(p, m)
        res[m].append(ok)
        inf[m].append((nsat, extra, L))
    if (i + 1) % 3 == 0:
        print(f"  {i+1}/{len(paths)} ({time.time()-t0:.0f}s)", flush=True)

ref = res["none"]
print()
for m in ("none", "clip", "goal"):
    n = sum(res[m])
    agree = sum(1 for a, b in zip(res[m], ref) if a == b)
    tag = "reference" if m == "none" else f"agrees with unclamped {agree}/{len(ref)}"
    ex = np.mean([x[1] for x in inf[m]])
    L = np.mean([x[2] for x in inf[m]])
    suffix = f"   +{ex:.0f} steps (+{100*ex/max(1,L):.0f}% length)" if m == "goal" else ""
    print(f"{m:5s}: success {n}/{len(paths)}   {tag}{suffix}")
print()
print("GOAL_DONE", flush=True)
