"""Robot configuration dataclasses."""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

from roboeval.action_modes import PelvisDof
from roboeval.const import HandSide
from roboeval.utils.dof import Dof


@dataclass
class GripperConfig:
    """Configuration for a gripper embedded into robot model.

    Attributes:
        actuators: List of gripper's actuator names
        range: Range of gripper control values
        body: Name of the gripper body for embedded grippers
        model: Path to the gripper XML model for standalone grippers
        pad_bodies: List of root pad bodies for contact detection
        pinch_site: Site name for pinch grasping, if applicable
        discrete: Whether to round control signal to min or max range value
        reverse_control: Whether to reverse the control signal direction
    """
    actuators: List[str]
    range: np.ndarray
    body: Optional[str] = None
    model: Optional[Path] = None
    pad_bodies: List[str] = field(default_factory=list)
    pinch_site: Optional[str] = None
    discrete: bool = True
    reverse_control: bool = False

    def __post_init__(self):
        """Validate configuration requirements."""
        if not self.body and not self.model:
            raise ValueError("Either 'body' or 'model' must be specified.")


@dataclass
class ArmConfig:
    """Configuration for a robot arm.

    Attributes:
        site: Site on the robot where the gripper could be attached
        links: List of body links that make up the arm
        wrist_dof: Optional wrist DOF that can be added to the arm
        offset_position: Mounting positional offset in XYZ coordinates
        offset_euler: Mounting orientation offset as Euler angles
        model: Path to the arm model file, if separate
        actuators: List of actuator names for the arm
        joints: List of joint names for the arm
    """
    site: str
    links: List[str]
    wrist_dof: Optional[Dof] = None
    offset_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    offset_euler: np.ndarray = field(default_factory=lambda: np.zeros(3))
    model: Optional[Path] = None
    actuators: List[str] = field(default_factory=list)
    joints: List[str] = field(default_factory=list)


@dataclass
class FloatingBaseConfig:
    """Configuration for a robot's floating base.

    Attributes:
        dofs: Dictionary mapping pelvis DOF types to their configurations
        delta_range_position: Min/max range for position control deltas
        delta_range_rotation: Min/max range for rotation control deltas
        offset_position: Base positional offset in XYZ coordinates
    """
    dofs: Dict[PelvisDof, Dof]
    delta_range_position: Tuple[float, float]
    delta_range_rotation: Tuple[float, float]
    offset_position: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class RobotIKConfig:
    """Configuration for robot inverse kinematics solver.
    
    Attributes:
        robot_prefix: Prefix used in robot model names
        root_body_name: Name of the robot's root body
        torso_name: Name of the robot's torso body
        arm_roots: List of root body names for each arm
        arm_sites: List of site names for each arm's end-effector
        kp: Position proportional gain for IK control
        kv: Velocity gain for IK control (calculated if None)
        joint_damping: Joint damping coefficient (calculated if None)
        solver_max_steps: Maximum iterations for IK solver
        wrist_angle_scale: Scaling factor for wrist angle control
        timestep_factor: Factor to adjust simulation timestep during IK
        joint_limits: Dictionary of joint limits by name
        end_effector_exclude_word: String to exclude certain joints from end effector control
        kp_position: Position proportional gain
        kv_position: Position velocity gain
        kp_orientation: Orientation proportional gain
        kv_orientation: Orientation velocity gain
        range_ee_position: Control range for end-effector position
        convergence_threshold: Error threshold for early stopping
    """
    # Robot structure
    robot_prefix: str
    root_body_name: str
    torso_name: str
    arm_roots: List[str]
    arm_sites: List[str]
    
    # Performance parameters
    kp: float = 1000
    kv: float = None
    joint_damping: float = None
    solver_max_steps: int = 40
    wrist_angle_scale: float = 2
    timestep_factor: int = 10
    
    # Joint configuration
    joint_limits: Dict = field(default_factory=dict)
    end_effector_exclude_word: str = None

    # Control parameters
    kp_position: float = 100.0
    kv_position: float = 10.0
    kp_orientation: float = 1.0
    kv_orientation: float = 0.10
    range_ee_position: Tuple[float, float] = (-10.0, 10.0)
    convergence_threshold: float = 0.01
    
    def __post_init__(self):
        """Calculate default values for optional parameters."""
        if self.kv is None:
            self.kv = 2 * np.sqrt(self.kp)
        if self.joint_damping is None:
            self.joint_damping = self.kp / 200


@dataclass
class RobotConfig:
    """Configuration for a complete robot.

    Attributes:
        delta_range: Action range for delta position action mode
        position_kp: Stiffness of actuators for absolute position action mode
        pelvis_body: Name of the pelvis body element
        floating_base: Configuration for the robot's floating base
        gripper: Configuration for the robot's grippers
        arms: Configuration for the robot's arms by hand side
        actuators: Dictionary of actuator names and whether they're used in floating mode
        cameras: List of available camera names
        namespaces_to_remove: List of XML namespaces to remove from the model
        model: Path to the robot's XML model file
        model_name: Name of the robot model
        arm_offset: Dictionary of arm position offsets by hand side
        arm_offset_euler: Dictionary of arm orientation offsets by hand side
    """
    delta_range: Tuple[float, float]
    position_kp: float
    pelvis_body: str
    floating_base: FloatingBaseConfig
    gripper: GripperConfig
    arms: Dict[HandSide, ArmConfig]
    actuators: Optional[Dict[str, bool]] = None
    cameras: List[str] = field(default_factory=list)
    namespaces_to_remove: List[str] = field(default_factory=list)
    model: Optional[Path] = None 
    model_name: Optional[str] = None
    arm_offset: Optional[Dict[HandSide, np.ndarray]] = None
    arm_offset_euler: Optional[Dict[HandSide, np.ndarray]] = None
