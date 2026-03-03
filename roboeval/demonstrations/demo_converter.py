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
