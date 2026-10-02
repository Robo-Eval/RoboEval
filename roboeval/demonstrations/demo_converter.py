"""Helper class for converting between different action representations."""
from __future__ import annotations

from copy import deepcopy
from typing import Optional

import numpy as np
from tqdm import tqdm

from roboeval.robots.robot import Robot
from roboeval.demonstrations.demo import Demo, DemoStep
from roboeval.demonstrations.utils import Metadata

from roboeval.roboeval_env import RoboEvalEnv, CONTROL_FREQUENCY_MAX
from scipy.spatial.transform import Rotation as R

def get_delta_action(
    prev_action: np.ndarray,
    action: np.ndarray,
    base_dof_count: int,
    grippers_count: int,
    ) -> np.ndarray:
    delta = action - prev_action
    delta[:base_dof_count] = action[:base_dof_count]
    delta[-grippers_count:] = action[-grippers_count:]
    return delta


def _feasible_limb_targets_rowwise(
    initial: np.ndarray, raw: np.ndarray, max_step: float
) -> np.ndarray:
    """Greedy per-step slew limiter on a sequence of absolute targets.

    Starting from ``initial``, each row of ``raw`` is approached by moving each
    component at most ``max_step`` (L-inf clamp). This mirrors the runtime joint
    velocity clamp in :class:`JointPositionActionMode` (``delta`` clipped to
    ``MAX_JOINT_VEL * control_dt``), but computed offline against the previous
    *commanded* target rather than the achieved state.

    :param initial: Starting target, shape ``(d,)``.
    :param raw: Desired absolute targets, shape ``(T, d)``.
    :param max_step: Max per-component change between consecutive rows.
    :return: Feasible targets, same shape as ``raw``.
    """
    initial = np.asarray(initial, dtype=np.float64)
    raw = np.asarray(raw, dtype=np.float64)
    out = np.empty_like(raw)
    prev = initial.copy()
    for i in range(raw.shape[0]):
        prev = prev + np.clip(raw[i] - prev, -max_step, max_step)
        out[i] = prev
    return out


def _limb_layout(robot):
    """Return ``(base_n, limb_lo, limb_hi, n_grip, dim)`` for an action vector.

    Action layout is ``[floating_base | limb joints | grippers]`` (matching
    ``robot._initial_qpos``). The limb slice is ``[base_n : dim - n_grip]``.
    """
    base_n = robot.floating_base.dof_amount if robot.floating_base else 0
    n_limb = len(robot.limb_actuators)
    n_grip = len(robot.grippers)
    dim = base_n + n_limb + n_grip
    return base_n, base_n, dim - n_grip, n_grip, dim


class DemoConverter:
    """Class to convert demonstrations."""

    @staticmethod
    def absolute_to_delta(demo: Demo) -> Demo:
        """Converts a demonstration from absolute to delta actions.

        :param demo: The demonstration to convert (in absolute joint positions).
        :return: The converted demonstration (in delta joint positions).
        """

        timesteps = deepcopy(demo.timesteps)
        if demo.metadata.environment_data.action_mode_absolute:
            demo.metadata.environment_data.action_mode_absolute = False

        # Cache environment info
        robot = demo.metadata.get_robot()
        action_space = robot.action_mode.action_space(1)
        floating_dof_count = len(robot.action_mode.floating_dofs)
        grippers_count = len(robot.grippers)

        overhead = np.zeros_like(action_space.sample())

        # Get initial position of robot
        initial_robot_position = robot._initial_qpos
        last_action = initial_robot_position
        for timestep in timesteps:
            absolute_action = timestep.executed_action + overhead
            delta_action = get_delta_action(
                last_action, absolute_action, floating_dof_count, grippers_count
            )
            clipped_action = np.clip(delta_action, action_space.low, action_space.high)
            overhead = delta_action - clipped_action
            if not np.allclose(overhead, 0):
                timestep.set_executed_action(clipped_action)
                last_action = absolute_action - overhead
            else:
                overhead *= 0
                timestep.set_executed_action(delta_action)
                last_action = absolute_action
        
        # Handle any remaining overhead after the last timestep
        if not np.allclose(overhead, 0):
            # Create an additional timestep to handle remaining overhead
            last_timestep = deepcopy(timesteps[-1])
            # Set the action to the remaining overhead
            clipped_overhead = np.clip(overhead, action_space.low, action_space.high)
            last_timestep.set_executed_action(clipped_overhead)
            timesteps.append(last_timestep)
        
        if demo.metadata.environment_data.action_mode_absolute:
            demo.metadata.environment_data.action_mode_absolute = False
        return Demo(demo.metadata, timesteps)
    
    @staticmethod
    def joint_to_ee(demo: Demo) -> Demo:
        """Converts a demonstration from joint positions to end-effector positions.

        :param demo: The demonstration to convert (in joint positions).
        :return: The converted demonstration (in end-effector positions).
        """
        timesteps = deepcopy(demo.timesteps)
        
        # Cache environment info
        robot = demo.metadata.get_robot()
        original_action_space = robot.action_mode.action_space(1)
        floating_dof_count = len(robot.action_mode.floating_dofs)
        grippers_count = len(robot.grippers)
        
        # Store the original action mode
        original_action_mode = robot.action_mode
        
        # Create a new metadata with updated action mode for end-effector control
        new_metadata = deepcopy(demo.metadata)
        new_metadata.environment_data.end_effector_mode = True
        if hasattr(new_metadata.environment_data, "action_mode_absolute"):
            new_metadata.environment_data.action_mode_absolute = True
        
        # Convert each timestep's joint action to EE action
        for i, timestep in enumerate(timesteps):
            # Get the current joint action
            joint_action = timestep.executed_action
            
            # Extract floating base actions and gripper actions which should remain unchanged
            floating_base_actions = joint_action[:floating_dof_count] if floating_dof_count > 0 else np.array([])
            gripper_actions = joint_action[-grippers_count:] if grippers_count > 0 else np.array([])
            
            # Extract actual joint positions (excluding floating base and grippers)
            joint_positions = joint_action[floating_dof_count:len(joint_action)-grippers_count]
            
            # Use forward kinematics to convert joint positions to EE positions
            ee_positions = robot.forward_kinematics(joint_positions)
            
            # Combine EE positions with floating base and gripper actions
            ee_action = np.concatenate([
                floating_base_actions,
                ee_positions.flatten(),  # Assuming ee_positions contains position and orientation
                gripper_actions
            ])
            
            # Set the new action
            timestep.set_executed_action(ee_action)
        
        return Demo(new_metadata, timesteps)
    
    @staticmethod
    def joint_absolute_to_ee_delta(demo: Demo) -> Demo:
        """Converts a demonstration from absolute joint positions to delta end-effector positions.

        Uses rotation vectors (axis-angle) for singularity-free orientation deltas.

        :param demo: The demonstration to convert (in absolute joint positions).
        :return: The converted demonstration (in delta end-effector positions).
        """
        
        # First convert absolute joint to absolute EE (FK now returns rotvec orientations)
        abs_ee_demo = DemoConverter.joint_to_ee(deepcopy(demo))
        
        # Create a new metadata for delta EE control
        new_metadata = deepcopy(abs_ee_demo.metadata)
        new_metadata.environment_data.action_mode_absolute = False
                
        # Get environment info
        robot = new_metadata.get_robot()
        action_space = robot.action_mode.action_space(1)
        floating_dof_count = len(robot.action_mode.floating_dofs)
        grippers_count = len(robot.grippers)
        
        # Create new timesteps with delta EE actions
        timesteps = deepcopy(abs_ee_demo.timesteps)
        overhead = np.zeros_like(action_space.sample())
        last_action = robot._initial_ee_pos
        
        for timestep in timesteps:
            absolute_action = timestep.executed_action + overhead

            # Extract per-arm segments: each arm is [x, y, z, rvx, rvy, rvz]
            absolute_action_arms = absolute_action[floating_dof_count:-grippers_count]
            half = len(absolute_action_arms) // 2
            abs_left = absolute_action_arms[:half]
            abs_right = absolute_action_arms[half:]

            last_action_arms = last_action[floating_dof_count:-grippers_count]
            last_left = last_action_arms[:half]
            last_right = last_action_arms[half:]

            # Position deltas
            delta_left_pos = abs_left[:3] - last_left[:3]
            delta_right_pos = abs_right[:3] - last_right[:3]

            # Orientation deltas as rotation vectors (no gimbal lock, no wrapping needed)
            r_left_curr = R.from_rotvec(last_left[3:6])
            r_left_target = R.from_rotvec(abs_left[3:6])
            delta_left_rotvec = (r_left_target * r_left_curr.inv()).as_rotvec()

            r_right_curr = R.from_rotvec(last_right[3:6])
            r_right_target = R.from_rotvec(abs_right[3:6])
            delta_right_rotvec = (r_right_target * r_right_curr.inv()).as_rotvec()

            # Assemble delta action
            delta_action = np.zeros_like(absolute_action)
            delta_action[floating_dof_count:-grippers_count] = np.concatenate([
                delta_left_pos, delta_left_rotvec,
                delta_right_pos, delta_right_rotvec,
            ])
            
            # Floating base and grippers are passed through as-is
            delta_action[:floating_dof_count] = absolute_action[:floating_dof_count]
            delta_action[-grippers_count:] = absolute_action[-grippers_count:]
            
            clipped_action = np.clip(delta_action, action_space.low, action_space.high)
            overhead = delta_action - clipped_action
            
            if not np.allclose(overhead, 0):
                print('clipping: ', overhead)
                timestep.set_executed_action(clipped_action)
                last_action = absolute_action - overhead
            else:
                overhead *= 0
                timestep.set_executed_action(delta_action)
                last_action = absolute_action
        
        return Demo(new_metadata, timesteps)

    @staticmethod
    def clip_actions(demo: Demo, action_scale: float = 1) -> Demo:
        """Clip demo actions to action space."""
        timesteps = deepcopy(demo.timesteps)
        action_space = demo.metadata.get_action_space(action_scale)
        overhead = np.zeros_like(action_space.sample())
        for timestep in timesteps:
            action = timestep.executed_action + overhead
            clipped_action = np.clip(action, action_space.low, action_space.high)
            overhead = action - clipped_action
            timestep.set_executed_action(clipped_action)
        return Demo(demo.metadata, timesteps)

    @staticmethod
    def _validate_absolute_joint_demo(demo: Demo, fn_name: str) -> None:
        """Raise if ``demo`` is not absolute joint-space (required for clamping)."""
        env_data = demo.metadata.environment_data
        if env_data.end_effector_mode:
            raise ValueError(
                f"{fn_name} requires joint-space actions; demo is end-effector."
            )
        if not env_data.action_mode_absolute:
            raise ValueError(
                f"{fn_name} requires absolute joint actions; demo is delta."
            )

    @staticmethod
    def _resolve_max_step(control_frequency_hz: float, v_max: Optional[float]) -> float:
        """Max per-step limb change (rad) = v_max / control_frequency."""
        from roboeval.action_modes import JointPositionActionMode

        if control_frequency_hz <= 0:
            raise ValueError("control_frequency_hz must be positive.")
        if v_max is None:
            v_max = JointPositionActionMode.MAX_JOINT_VEL
        return v_max / control_frequency_hz

    @staticmethod
    def clamp_absolute_joint_velocity(
        demo: Demo,
        control_frequency_hz: float,
        max_joint_velocity_rad_s: Optional[float] = None,
    ) -> Demo:
        """Slew-limit absolute joint targets to respect the velocity limit.

        Rewrites each timestep's limb-joint targets so consecutive targets differ
        by at most ``v_max / control_frequency_hz`` per joint (L-inf), starting
        from the robot's initial qpos. Floating-base and gripper components are
        left unchanged. The number of timesteps is preserved — this is the
        offline equivalent of the runtime clamp, *not* a time-stretch (a clamped
        trajectory may lag and fail to reach its targets; see
        :meth:`retime_absolute_joint_velocity` to also re-time).

        :param demo: Absolute joint-space demonstration.
        :param control_frequency_hz: Control frequency the demo will run at.
        :param max_joint_velocity_rad_s: Velocity cap (default
            ``JointPositionActionMode.MAX_JOINT_VEL``).
        :raises ValueError: If the demo is end-effector or delta.
        :return: A new demo with slew-limited limb targets.
        """
        DemoConverter._validate_absolute_joint_demo(
            demo, "clamp_absolute_joint_velocity"
        )
        max_step = DemoConverter._resolve_max_step(
            control_frequency_hz, max_joint_velocity_rad_s
        )

        robot = demo.metadata.get_robot()
        _, limb_lo, limb_hi, _, _ = _limb_layout(robot)
        q0 = np.asarray(robot._initial_qpos, dtype=np.float64)

        timesteps = deepcopy(demo.timesteps)
        if not timesteps:
            return Demo(deepcopy(demo.metadata), timesteps)

        raw_limb = np.array(
            [
                np.asarray(ts.executed_action, dtype=np.float64)[limb_lo:limb_hi]
                for ts in timesteps
            ]
        )
        feasible_limb = _feasible_limb_targets_rowwise(
            q0[limb_lo:limb_hi], raw_limb, max_step
        )
        for ts, limb in zip(timesteps, feasible_limb):
            full = np.asarray(ts.executed_action, dtype=np.float64).copy()
            full[limb_lo:limb_hi] = limb
            ts.set_executed_action(full)
        return Demo(deepcopy(demo.metadata), timesteps)

    @staticmethod
    def max_limb_discrepancy_after_velocity_clamp(
        demo: Demo,
        control_frequency_hz: float,
        max_joint_velocity_rad_s: Optional[float] = None,
    ) -> float:
        """Max per-joint gap (rad) between raw and slew-clamped limb targets.

        A measure of how infeasible a demo is under the velocity limit: ``0.0``
        means every target already satisfies the slew rule; larger values mean
        the clamp had to hold the robot back further behind its commanded target.

        :return: Max absolute limb-target discrepancy across all timesteps.
        """
        clamped = DemoConverter.clamp_absolute_joint_velocity(
            demo, control_frequency_hz, max_joint_velocity_rad_s
        )
        robot = demo.metadata.get_robot()
        _, limb_lo, limb_hi, _, _ = _limb_layout(robot)

        max_d = 0.0
        for raw_ts, cl_ts in zip(demo.timesteps, clamped.timesteps):
            raw_limb = np.asarray(raw_ts.executed_action, dtype=np.float64)[
                limb_lo:limb_hi
            ]
            cl_limb = np.asarray(cl_ts.executed_action, dtype=np.float64)[
                limb_lo:limb_hi
            ]
            if raw_limb.size:
                max_d = max(max_d, float(np.max(np.abs(cl_limb - raw_limb))))
        return max_d

    @staticmethod
    def retime_absolute_joint_velocity(
        demo: Demo,
        control_frequency_hz: float,
        max_joint_velocity_rad_s: Optional[float] = None,
        interpolate_gripper: bool = False,
    ) -> Demo:
        """Time-stretch a demo so every waypoint is reachable under the limit.

        Unlike :meth:`clamp_absolute_joint_velocity` (same length, lags), this
        inserts linearly-interpolated intermediate timesteps between consecutive
        targets so the limb never moves more than ``v_max / control_frequency_hz``
        per step *and* still reaches each original waypoint. Both arms share the
        same substep count per segment, so bimanual motion stays synchronized.
        Re-simulate the result (e.g. via :meth:`create_demo_in_new_env` with an
        ``enforce_joint_velocity_limits=True`` env) to capture consistent obs.

        :param demo: Absolute joint-space demonstration.
        :param control_frequency_hz: Control frequency the retimed demo runs at.
        :param max_joint_velocity_rad_s: Velocity cap (default
            ``JointPositionActionMode.MAX_JOINT_VEL``).
        :param interpolate_gripper: If True, linearly interpolate the gripper
            command across inserted substeps; if False (default), hold the
            previous gripper value and switch only on reaching the waypoint, so
            grasps fire at arrival rather than mid-approach.
        :raises ValueError: If the demo is end-effector or delta.
        :return: A new, time-stretched demo (>= original length).
        """
        DemoConverter._validate_absolute_joint_demo(
            demo, "retime_absolute_joint_velocity"
        )
        max_step = DemoConverter._resolve_max_step(
            control_frequency_hz, max_joint_velocity_rad_s
        )

        robot = demo.metadata.get_robot()
        _, limb_lo, limb_hi, n_grip, dim = _limb_layout(robot)
        prev_full = np.asarray(robot._initial_qpos, dtype=np.float64)

        new_steps: list[DemoStep] = []
        for ts in demo.timesteps:
            target_full = np.asarray(ts.executed_action, dtype=np.float64)
            limb_move = np.abs(
                target_full[limb_lo:limb_hi] - prev_full[limb_lo:limb_hi]
            )
            max_move = float(np.max(limb_move)) if limb_move.size else 0.0
            n_sub = max(1, int(np.ceil(max_move / max_step - 1e-9)))
            for k in range(1, n_sub + 1):
                interp = prev_full + (target_full - prev_full) * (k / n_sub)
                if not interpolate_gripper and n_grip > 0:
                    src = target_full if k == n_sub else prev_full
                    interp[dim - n_grip:] = src[dim - n_grip:]
                sub = deepcopy(ts)
                sub.set_executed_action(interp)
                new_steps.append(sub)
            prev_full = target_full
        return Demo(deepcopy(demo.metadata), new_steps)

    @staticmethod
    def decimate(
        demo: Demo,
        target_freq: int,
        original_freq: int = CONTROL_FREQUENCY_MAX,
        robot: Optional[Robot] = None,
    ) -> Demo:
        """Decimate provided demo at certain rate.

        :param demo: Original demonstration.
        :param target_freq: Control frequency of the new demo.
        :param original_freq: Control frequency of the original demo.
        :param robot: Optional existing robot instance to speed-up decimation.
        """
        if original_freq != CONTROL_FREQUENCY_MAX:
            raise RuntimeError(
                f"Demonstrations with frequency != {CONTROL_FREQUENCY_MAX} "
                f"can't be decimated."
            )

        decimation_rate = int(np.round(original_freq / target_freq))
        robot = robot or demo.metadata.get_robot()
        action_space = robot.action_mode.action_space(decimation_rate)
        grippers_count = len(robot.grippers)

        original_timesteps = deepcopy(demo.timesteps)
        decimated_timesteps: list[DemoStep] = []

        action = np.zeros_like(action_space.sample())
        overhead = np.zeros_like(action_space.sample())

        # Repeat final actions to ensure success
        if 0 < len(original_timesteps) % decimation_rate < decimation_rate:
            steps_count = decimation_rate - len(original_timesteps) % decimation_rate
            original_timesteps.extend([deepcopy(original_timesteps[-1])] * steps_count)

        actions_counter = 0
        for timestep in original_timesteps:
            timestep = deepcopy(timestep)
            original_action = timestep.executed_action.copy()
            action += original_action + overhead
            overhead *= 0
            actions_counter += 1
            if actions_counter % decimation_rate == 0:
                if demo.metadata.environment_data.action_mode_absolute:
                    floating_base_actions = demo.metadata.floating_dof_count
                    action[floating_base_actions:] = (
                        action[floating_base_actions:] / decimation_rate
                    )
                action[-grippers_count:] = original_action[-grippers_count:]
                clipped_action = np.clip(action, action_space.low, action_space.high)
                timestep.set_executed_action(clipped_action)
                decimated_timesteps.append(timestep)
                overhead = action - clipped_action
                action = np.zeros_like(action)
        return Demo(demo.metadata, decimated_timesteps)

    @staticmethod
    def create_demo_in_new_env(
        demo: Demo,
        env: RoboEvalEnv,
    ) -> Demo:
        """Create a new demonstration in a new environment.

        :param demo: The demonstration to convert.
        :param env: The environment to collect the new demonstration in (action
            mode must match the demonstration).

        :return: The new demonstration.
        """
        env.reset(seed=demo.seed)
        metadata = Metadata.from_env(env)
        metadata.uuid = demo.metadata.uuid
        new_demo = Demo(metadata)

        if (demo.metadata.environment_data.action_mode_absolute != env.action_mode.absolute) or (demo.metadata.environment_data.end_effector_mode != env.action_mode.ee):
            assert demo.metadata.environment_data.action_mode_absolute == True and demo.metadata.environment_data.end_effector_mode == False, "Only absolute joint positions is supported"

            if env.action_mode.ee and env.action_mode.absolute: # Absolute EE
                demo = DemoConverter.joint_to_ee(demo)
            elif env.action_mode.ee and not env.action_mode.absolute: # Delta EE
                demo = DemoConverter.joint_absolute_to_ee_delta(demo)
            elif not env.action_mode.ee and not env.action_mode.absolute: # Delta Joint
                demo = DemoConverter.absolute_to_delta(demo)
            else:
                raise ValueError(
                    "The required action mode is not supported. "
                )
        
        with tqdm(
            total=len(demo.timesteps),
            desc="Creating Demo",
            unit="step",
            leave=False,
        ) as pbar:
            for timestep in demo.timesteps:
                action = timestep.executed_action
                observation, reward, term, trunc, info = env.step(action)
                new_demo.add_timestep(
                    observation,
                    reward,
                    term,
                    trunc,
                    info,
                    action,
                )
                pbar.update()

        return new_demo
