"""Tests for joint velocity feasibility clamping on demonstrations."""
import numpy as np
import pytest

from roboeval.demonstrations.demo import Demo, DemoStep
from roboeval.demonstrations.demo_converter import (
    DemoConverter,
    _feasible_limb_targets_rowwise,
)
from roboeval.demonstrations.utils import Metadata, ObservationMode
from roboeval.action_modes import JointPositionActionMode
from roboeval.envs.lift_pot import LiftPot


def test_feasible_limb_targets_rowwise_respects_max_step():
    initial = np.array([0.0, 0.0])
    raw = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    max_step = 0.3
    out = _feasible_limb_targets_rowwise(initial, raw, max_step)
    diffs = np.diff(np.vstack([initial, out]), axis=0)
    assert np.all(np.abs(diffs) <= max_step + 1e-9)
    assert out.shape == raw.shape
    # First row moves toward raw[0] by at most max_step per component
    assert np.allclose(out[0], [0.3, 0.0])


def test_feasible_limb_targets_rowwise_reaches_target_if_enough_steps():
    initial = np.zeros(1)
    target = np.array([[5.0]])
    max_step = 1.0
    out = _feasible_limb_targets_rowwise(initial, target, max_step)
    assert out.shape == (1, 1)
    assert abs(out[0, 0] - 1.0) < 1e-9


def test_clamp_absolute_joint_velocity_rejects_ee_demo():
    meta = Metadata.from_env_cls(
        env_cls=LiftPot,
        action_mode=JointPositionActionMode,
        floating_dofs=[],
        obs_mode=ObservationMode.Lightweight,
        action_mode_absolute=True,
        end_effector_mode=True,
    )
    demo = Demo(meta, [DemoStep({}, 0.0, False, False, {}, np.zeros(8))])
    with pytest.raises(ValueError, match="joint-space"):
        DemoConverter.clamp_absolute_joint_velocity(demo, control_frequency_hz=500)


def test_clamp_absolute_joint_velocity_rejects_delta_demo():
    meta = Metadata.from_env_cls(
        env_cls=LiftPot,
        action_mode=JointPositionActionMode,
        floating_dofs=[],
        obs_mode=ObservationMode.Lightweight,
        action_mode_absolute=False,
        end_effector_mode=False,
    )
    demo = Demo(meta, [DemoStep({}, 0.0, False, False, {}, np.zeros(16))])
    with pytest.raises(ValueError, match="absolute joint"):
        DemoConverter.clamp_absolute_joint_velocity(demo, control_frequency_hz=500)


def test_max_limb_discrepancy_matches_clamp_effect():
    meta = Metadata.from_env_cls(
        env_cls=LiftPot,
        action_mode=JointPositionActionMode,
        floating_dofs=[],
        obs_mode=ObservationMode.Lightweight,
        action_mode_absolute=True,
        end_effector_mode=False,
    )
    robot = meta.get_robot()
    base_n = robot.floating_base.dof_amount if robot.floating_base else 0
    n_limb = len(robot.limb_actuators)
    n_grip = len(robot.grippers)
    dim = base_n + n_limb + n_grip
    q0 = np.asarray(robot._initial_qpos, dtype=np.float64)
    a = q0.copy()
    steps = [DemoStep({}, 0.0, False, False, {}, a)]
    demo = Demo(meta, steps)
    hz = 500
    d = DemoConverter.max_limb_discrepancy_after_velocity_clamp(demo, hz)
    assert d == 0.0


def _bimanual_layout(meta):
    """(q0, base_n, limb_lo, limb_hi, n_grip, dim) for an absolute joint demo."""
    robot = meta.get_robot()
    base_n = robot.floating_base.dof_amount if robot.floating_base else 0
    n_limb = len(robot.limb_actuators)
    n_grip = len(robot.grippers)
    dim = base_n + n_limb + n_grip
    q0 = np.asarray(robot._initial_qpos, dtype=np.float64)
    return q0, base_n, base_n, dim - n_grip, n_grip, dim


def _abs_joint_meta():
    return Metadata.from_env_cls(
        env_cls=LiftPot,
        action_mode=JointPositionActionMode,
        floating_dofs=[],
        obs_mode=ObservationMode.Lightweight,
        action_mode_absolute=True,
        end_effector_mode=False,
    )


def test_retime_is_feasible_and_reaches_every_waypoint():
    """Retimed limb targets respect the slew cap AND hit each original waypoint."""
    meta = _abs_joint_meta()
    q0, base_n, limb_lo, limb_hi, n_grip, dim = _bimanual_layout(meta)

    # Raw targets with limb jumps far larger than one step can cover.
    a1 = q0.copy(); a1[base_n] = q0[base_n] + 1.0
    a2 = a1.copy(); a2[base_n + 1] = a1[base_n + 1] - 0.8
    a3 = q0.copy()
    actions = [a1, a2, a3]
    demo = Demo(meta, [DemoStep({}, 0.0, False, False, {}, a) for a in actions])

    hz, v_max = 100, 2.0
    max_step = v_max / hz
    retimed = DemoConverter.retime_absolute_joint_velocity(demo, hz, v_max)

    # Time-stretched: more steps than the original waypoint count.
    assert len(retimed.timesteps) > len(actions)

    # Feasibility: each consecutive limb step is within the slew cap, from q0.
    limbs = [
        np.asarray(ts.executed_action, dtype=np.float64)[limb_lo:limb_hi]
        for ts in retimed.timesteps
    ]
    prev = q0[limb_lo:limb_hi]
    for limb in limbs:
        assert np.all(np.abs(limb - prev) <= max_step + 1e-8)
        prev = limb

    # Every original waypoint's limb target is reached exactly somewhere.
    for a in actions:
        wp = a[limb_lo:limb_hi]
        assert any(np.allclose(limb, wp, atol=1e-9) for limb in limbs)


def test_retime_gripper_switches_at_waypoint_not_midflight():
    """With interpolate_gripper=False, the gripper holds then snaps at arrival."""
    meta = _abs_joint_meta()
    q0, base_n, limb_lo, limb_hi, n_grip, dim = _bimanual_layout(meta)
    assert n_grip > 0

    a = q0.copy()
    a[base_n] = q0[base_n] + 1.0           # big limb jump -> many substeps
    a[dim - n_grip:] = q0[dim - n_grip:] + 0.5  # gripper command change
    demo = Demo(meta, [DemoStep({}, 0.0, False, False, {}, a)])

    retimed = DemoConverter.retime_absolute_joint_velocity(
        demo, control_frequency_hz=100, max_joint_velocity_rad_s=2.0
    )
    assert len(retimed.timesteps) > 1

    grip0 = q0[dim - n_grip:]
    for ts in retimed.timesteps[:-1]:
        g = np.asarray(ts.executed_action, dtype=np.float64)[dim - n_grip:]
        assert np.allclose(g, grip0)  # held during approach
    g_last = np.asarray(retimed.timesteps[-1].executed_action, dtype=np.float64)[
        dim - n_grip:
    ]
    assert np.allclose(g_last, a[dim - n_grip:])  # fires at the waypoint


def test_retime_rejects_ee_demo():
    meta = Metadata.from_env_cls(
        env_cls=LiftPot,
        action_mode=JointPositionActionMode,
        floating_dofs=[],
        obs_mode=ObservationMode.Lightweight,
        action_mode_absolute=True,
        end_effector_mode=True,
    )
    demo = Demo(meta, [DemoStep({}, 0.0, False, False, {}, np.zeros(8))])
    with pytest.raises(ValueError, match="joint-space"):
        DemoConverter.retime_absolute_joint_velocity(demo, control_frequency_hz=500)


def test_clamp_absolute_joint_velocity_limb_slew_matches_runtime_rule():
    """Limb slice should move at most v_max/hz per timestep (L-inf per joint)."""
    meta = Metadata.from_env_cls(
        env_cls=LiftPot,
        action_mode=JointPositionActionMode,
        floating_dofs=[],
        obs_mode=ObservationMode.Lightweight,
        action_mode_absolute=True,
        end_effector_mode=False,
    )
    robot = meta.get_robot()
    base_n = robot.floating_base.dof_amount if robot.floating_base else 0
    n_limb = len(robot.limb_actuators)
    n_grip = len(robot.grippers)
    dim = base_n + n_limb + n_grip
    q0 = np.asarray(robot._initial_qpos, dtype=np.float64)

    actions = []
    for _ in range(5):
        a = q0.copy()
        # Large jump on first limb joint only
        a[base_n] = q0[base_n] + 10.0
        actions.append(a)

    steps = [DemoStep({}, 0.0, False, False, {}, act) for act in actions]
    demo = Demo(meta, steps)
    hz = 100
    v_max = 2.0
    clamped = DemoConverter.clamp_absolute_joint_velocity(
        demo, control_frequency_hz=hz, max_joint_velocity_rad_s=v_max
    )
    max_step = v_max / hz
    prev_limb = q0[base_n : dim - n_grip]
    for ts in clamped.timesteps:
        limb = np.asarray(ts.executed_action, dtype=np.float64)[base_n : dim - n_grip]
        assert np.all(np.abs(limb - prev_limb) <= max_step + 1e-8)
        prev_limb = limb.copy()
    # Floating + gripper unchanged from raw
    for i, ts in enumerate(clamped.timesteps):
        full = np.asarray(ts.executed_action, dtype=np.float64)
        assert np.allclose(full[:base_n], actions[i][:base_n])
        assert np.allclose(full[dim - n_grip :], actions[i][dim - n_grip :])
