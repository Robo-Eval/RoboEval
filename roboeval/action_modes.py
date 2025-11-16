"""Action modes for H1."""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import warnings
from abc import abstractmethod, ABC
from typing import Optional
from gymnasium import spaces
from mojo import Mojo

from roboeval.const import TOLERANCE_ANGULAR
from roboeval.utils.physics_utils import (
    is_target_reached,
)

import numpy as np
import copy 
if TYPE_CHECKING:
    from roboeval.robots.robot import Robot

from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

class TargetStateNotReachedWarning(Warning):
    """Warning raised when the target state is not reached within the maximum steps."""
    pass


class PelvisDof(Enum):
    """Set of floating base DOFs."""

    X = "pelvis_x"
    Y = "pelvis_y"
    Z = "pelvis_z"
    RZ = "pelvis_rz"


def wrap_to_pi(euler_angles: np.ndarray) -> np.ndarray:
    return (euler_angles + np.pi) % (2 * np.pi) - np.pi

def compose_pose_delta_stable(current_pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Compose delta pose with current EE pose using stable rotation math.
    
    Supports both single-arm (6D) and bimanual (12D) poses.
    
    Args:
        current_pose: (6,) or (12,) array of [pos + euler] for 1 or 2 arms
        delta: (6,) or (12,) array of [pos_delta + euler_delta] for 1 or 2 arms

    Returns:
        composed_pose: (6,) or (12,) array of [new_pos + new_euler] for 1 or 2 arms
    """
    assert current_pose.shape == delta.shape, "current_pose and delta must have the same shape"
    assert current_pose.shape[0] in [6, 12], "Pose must be 6D (single-arm) or 12D (bimanual)"
    
    num_arms = current_pose.shape[0] // 6
    composed = np.zeros_like(current_pose)

    for i in range(num_arms):
        pos = current_pose[i*6:i*6+3] + delta[i*6:i*6+3]

        r_current = R.from_euler("xyz", current_pose[i*6+3:i*6+6])
        r_delta = R.from_euler("xyz", delta[i*6+3:i*6+6])
        r_composed = r_delta * r_current

        composed[i*6:i*6+3] = pos
        composed[i*6+3:i*6+6] = r_composed.as_euler("xyz")

    return composed

def euler_to_normalized_quaternion(euler_xyz: np.ndarray) -> Quaternion:
    """Convert Euler angles to a normalized quaternion."""
    quat = R.from_euler("xyz", euler_xyz).as_quat()  # xyzw
    norm = np.linalg.norm(quat)
    if norm == 0 or np.any(np.isnan(quat)):
        raise ValueError("Invalid quaternion generated from Euler angles")
    quat /= norm
    return Quaternion(quat[3], quat[0], quat[1], quat[2])  # Convert to wxyz (scalar first)

def compose_pose_delta(current_pose: np.ndarray, delta_pose: np.ndarray) -> np.ndarray:
    """Composes a delta pose with a current pose using proper orientation math (Euler angles)."""
    result = np.zeros_like(current_pose)

    for i in [0, 6]:  # Left arm: 0-5, Right arm: 6-11
        # Position addition
        curr_pos = current_pose[i:i+3]
        delta_pos = delta_pose[i:i+3]
        result[i:i+3] = curr_pos + delta_pos

        # Orientation composition
        curr_euler = current_pose[i+3:i+6]
        delta_euler = delta_pose[i+3:i+6]

        # Create Rotation objects
        r_current = R.from_euler('xyz', curr_euler)
        r_delta = R.from_euler('xyz', delta_euler)

        # Compose the rotations (delta applied first)
        r_new = r_delta * r_current

        # Convert back to euler angles (in same convention)
        result[i+3:i+6] = r_new.as_euler('xyz')

    return result


DEFAULT_DOFS = [PelvisDof.X, PelvisDof.Y, PelvisDof.RZ]


class ActionMode(ABC):
    """Base action mode class used for controlling H1."""

    def __init__(
        self,
        floating_base: bool = True,
        floating_dofs: Optional[list[PelvisDof]] = None,
    ):
        """Init.

        :param floating_base: If True, then legs are frozen, and the robot base
            controlled by positional actuators.
            If False, then user has full control of legs (i.e. for whole-body control).
        :param floating_dofs: Set of floating DOFs. By default, it is: [X, Y, RZ].
        """
        self._floating_base = floating_base
        self._floating_dofs = DEFAULT_DOFS if floating_dofs is None else floating_dofs

        # Will be assigned later
        self._mojo: Optional[Mojo] = None
        self._robot: Optional[Robot] = None

    def bind_robot(self, robot: Robot, mojo: Mojo):
        """Bind action mode to robot."""
        self._robot = robot
        self._mojo = mojo

    @property
    def floating_base(self) -> bool:
        """Is floating base enabled."""
        return self._floating_base

    @property
    def floating_dofs(self) -> list[PelvisDof]:
        """Set of floating DOFs."""
        return self._floating_dofs

    @abstractmethod
    def action_space(
        self, action_scale: float, seed: Optional[int] = None
    ) -> spaces.Box:
        """The action space for this action mode."""
        pass

    @abstractmethod
    def step(self, action: np.ndarray):
        """Apply the control command and step the physics.

        Note: This function has the responsibility of calling `mujoco.mj_step`.

        :param action: The entire action passed to the action mode.
        """
        pass


class JointPositionActionMode(ActionMode):
    """Control all joints through joint position.

    Allows to control joint positions, supporting both absolute and delta positions.
    For absolute control, set 'absolute' to True. If the floating base is enabled,
    only delta position control is applied to it.

    Notes:
        - `block_until_reached` does not guarantee reaching the target position because
          the target position could be unreachable due to collisions.
        - Joints of the `floating_base` are always controlled in delta position mode.
    """

    MAX_STEPS = 200

    def __init__(
        self,
        absolute: bool = False,
        block_until_reached: bool = False,
        floating_base: bool = True,
        ee: bool = False,
        floating_dofs: list[PelvisDof] = None,
    ):
        """See base.

        :param absolute: Use absolute or delta joint positions.
        :param block_until_reached: Continue stepping until the target
            position is reached or the step threshold is exceeded.
        """
        super().__init__(
            floating_base=floating_base,
            floating_dofs=floating_dofs,
        )
        self.ee = ee
        self.absolute = absolute
        self.block_until_reached = block_until_reached

    def _action_space_ee(self, action_scale: float, seed: Optional[int] = None) -> spaces.Box:
        """Get the action space for end-effector control."""
        bounds = []
        for wrist_site in self._robot._wrist_sites:
            # Position bounds (x, y, z)
            pos_bounds = [[-action_scale, action_scale]] * 3
            if self.absolute:
                # For absolute mode, use inf 
                pos_bounds = [[-np.inf, np.inf]] * 3 #TODO: check if this is correct
            else:
                # For delta mode, use the action scale
                pos_bounds = [[-np.inf, np.inf]] * 3 #TODO: check if this is correct
            
            bounds.extend(pos_bounds)
            
            # Orientation bounds (represented as euler angles)
            rot_bounds = [[-action_scale, action_scale]] * 3
            if self.absolute:
                # For absolute mode, use 360 degrees
                rot_bounds = [[-np.pi, np.pi]] * 3 #TODO: check if this is correct
            else:
                # For delta mode, use the action scale
                rot_bounds = [[-np.pi, np.pi]] * 3 #TODO: check if this is correct
            bounds.extend(rot_bounds)

        # Return as spaces.Box
        bounds = np.array(bounds).copy().astype(np.float32)
        return bounds

    def action_space(
        self, action_scale: float, seed: Optional[int] = None
    ) -> spaces.Box:
        """See base."""
        bounds = []
        if self.floating_base:
            action_bounds = self._robot.floating_base.get_action_bounds()
            action_bounds = [np.array(b) * action_scale for b in action_bounds]
            bounds.extend(action_bounds)

        if self.ee:
            action_bounds = self._action_space_ee(action_scale, seed)
            # Convert to list
            action_bounds = [np.array(b) for b in action_bounds]
            bounds.extend(action_bounds)
        else:
            for actuator in self._robot.limb_actuators:
                action_bounds = np.array(
                    self._robot.get_limb_control_range(actuator, self.absolute)
                )
                action_bounds *= 1 if self.absolute else action_scale
                bounds.append(action_bounds)
        for _, gripper in self._robot.grippers.items():
            bounds.append(gripper.range)
        bounds = np.array(bounds).copy().astype(np.float32)
        low, high = bounds.T
        return spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
            seed=seed,
        )

    def step(self, action: np.ndarray, inverse_kinematics = None):
        """See base."""
        if not inverse_kinematics:
            assert not self.ee, "Cannot use ee mode with ik set to None"
        if self.floating_base:
            base_action = action[: self._robot.floating_base.dof_amount]
            action = action[self._robot.floating_base.dof_amount :]
            self._robot.floating_base.set_control(base_action)

        gripper_actions = action[-len(self._robot.grippers) :]
        for side, gripper_action in zip(self._robot.grippers, gripper_actions):
            self._robot.grippers[side].set_control(gripper_action)

        action = action[: -len(self._robot.grippers)]

        # Check if the action is for end-effector control
        if self.ee:
            # Get floating base DOF count (0 if no floating base)
            floating_base_dof = self._robot.floating_base.dof_amount if self._robot.floating_base else 0
            
            # Check if robot is bimanual or single-arm
            is_bimanual = self._robot._multiarm

            # Check if the action is absolute or delta
            if self.absolute:
                # Convert to joint positions via IK
                if is_bimanual:
                    # Bimanual robot: use both arms
                    num_qpos_arm = (len(self._robot.qpos_actuated) - len(self._robot.grippers) - floating_base_dof) // 2
                    qpos_arm_left = self._robot.qpos_actuated[floating_base_dof:floating_base_dof + num_qpos_arm]
                    qpos_arm_right = self._robot.qpos_actuated[floating_base_dof + num_qpos_arm:floating_base_dof + 2*num_qpos_arm]
                    
                    target_pose_left = action[:6]
                    target_pose_right = action[6:12]
                else:
                    # Single-arm robot: only left arm
                    qpos_arm_left = self._robot.qpos_actuated[floating_base_dof:-len(self._robot.grippers)]
                    qpos_arm_right = np.zeros(len(qpos_arm_left))  # Dummy, will be ignored by IK
                    
                    target_pose_left = action[:6]
                    target_pose_right = np.zeros(6)  # Dummy, will be ignored by IK
                
                joint_positions = inverse_kinematics.solve_arms_only(
                    qpos_arm_left=qpos_arm_left,
                    qpos_arm_right=qpos_arm_right,
                    target_pose_left=target_pose_left,
                    target_pose_right=target_pose_right,
                )
                current_ee_pose = self._robot.forward_kinematics(joint_positions = joint_positions)

                action = joint_positions

            else:
                # Delta EE mode
                joint_positions = self._robot.get_initial_joint_positions()

                # Get current end-effector pose
                current_ee_pose = self._robot.forward_kinematics(joint_positions=joint_positions)

                # Compose delta EE pose using stable math
                combined = compose_pose_delta_stable(current_ee_pose, action)
                
                if is_bimanual:
                    # Bimanual: split joint positions for each arm
                    half_len = len(joint_positions) // 2
                    qpos_arm_left = joint_positions[:half_len]
                    qpos_arm_right = joint_positions[half_len:]
                    
                    target_pose_left = combined[:6]
                    target_pose_right = combined[6:12]
                else:
                    # Single-arm: only left arm
                    qpos_arm_left = joint_positions
                    qpos_arm_right = np.zeros(len(joint_positions))  # Dummy
                    
                    target_pose_left = combined[:6]
                    target_pose_right = np.zeros(6)  # Dummy
                
                joint_positions_new = inverse_kinematics.solve_arms_only(
                    qpos_arm_left=qpos_arm_left,
                    qpos_arm_right=qpos_arm_right,
                    target_pose_left=target_pose_left,
                    target_pose_right=target_pose_right,
                    inplace=True,
                    forward=False
                )

                action = copy.deepcopy(joint_positions_new)

        for i, actuator in enumerate(self._robot.limb_actuators):
            actuator = self._mojo.physics.bind(actuator)
            actuator.ctrl = action[i] if (self.absolute or self.ee) else actuator.ctrl + action[i]
        if self.block_until_reached:
            self._step_until_reached()
        else:
            self._mojo.step()

    def _step_until_reached(self):
        """Step physics until the target position is reached."""
        steps_counter = 0
        while True:
            self._mojo.step()
            steps_counter += 1
            if self._is_target_state_reached() or steps_counter >= self.MAX_STEPS:
                if steps_counter >= self.MAX_STEPS:
                    warnings.warn(
                        f"Failed to reach target state in " f"{self.MAX_STEPS} steps!",
                        TargetStateNotReachedWarning,
                    )
                break

    def _is_target_state_reached(self):
        if self.floating_base:
            if not self._robot.floating_base.is_target_reached:
                return False
        for actuator in self._robot.limb_actuators:
            if not is_target_reached(actuator, self._mojo.physics, TOLERANCE_ANGULAR):
                return False
        return True