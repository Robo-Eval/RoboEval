import numpy as np
from mojo.elements.consts import JointType
from roboeval.const import ASSETS_PATH, THIRD_PARTY_PATH, HandSide
from roboeval.robots.config import ArmConfig, FloatingBaseConfig, RobotConfig, GripperConfig, RobotIKConfig
from roboeval.robots.robot import Robot
from roboeval.utils.dof import Dof
from roboeval.action_modes import PelvisDof

# Constants
PANDA_WRIST_DOF = None
STIFFNESS_XY = 1e4
STIFFNESS_Z = 1e6
RANGE_DOF_Z = (0.4, 1.0)

# Common configurations
PANDA_ARM = ArmConfig(
    site="attachment_site",
    links=["link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7"],
    wrist_dof=PANDA_WRIST_DOF,
    model=THIRD_PARTY_PATH / "mujoco_menagerie/franka_emika_panda/panda_nohand_modified.xml",
    joints=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
    actuators=["actuator1", "actuator2", "actuator3", "actuator4", "actuator5", "actuator6", "actuator7"],
)

PANDA_ACTUATORS = {
    "actuator1": True, "actuator2": True, "actuator3": True,
    "actuator4": True, "actuator5": True, "actuator6": True,
    "actuator7": True, "left_wrist": True, "right_wrist": True,
}

# Camera configurations
CAMERA_CONFIG = {
    "head": {
        "manual": True,
        "parent": "invisible_base",
        "position": (-0.4, 0, 1.35),
        "quaternion": (0.6124, 0.3536, -0.3536, -0.6124),
        "fov": 60,
    },
    "left_wrist": {
        "manual": True,
        "parent": "panda nohand_left/left_wrist_link",
        "position": (0.05, 0, 0.07),
        "quaternion": (0, 1, 0, 0),
        "fov": 60,
    },
    "right_wrist": {
        "manual": True,
        "parent": "panda nohand_right/right_wrist_link",
        "position": (0.05, 0, 0.07),
        "quaternion": (0, 1, 0, 0),
        "fov": 60,
    },
}

CAMERA_CONFIG_SINGLE = {
    "top": {
        "manual": True,
        "parent": "invisible_base",
        "position": (-0.4, 0, 1.35),
        "quaternion": (0.6124, 0.3536, -0.3536, -0.6124),
        "fov": 60,
    },
    "left_side": {
        "manual": True,
        "parent": "invisible_base",
        "position": (0, -1.0, 0.5),
        "quaternion": (0.7071, 0.7071, 0, 0),
        "fov": 60,
    },
    "right_side": {
        "manual": True,
        "parent": "invisible_base",
        "position": (0, 1.0, 0.5),
        "quaternion": (0.7071, -0.7071, 0, 0),
        "fov": 60,
    },

    "front": {
        "manual": True,
        "parent": "invisible_base",
        # Place the camera on the far side of the table (positive X), a bit further back and
        # slightly higher so it looks down at the robot/table. The quaternion below orients
        # the camera to face toward the origin (robot) with a small downward tilt.
        "position": (1.4, 0, 1.6),
        # Quaternion (w, x, y, z). This gives a frontal view pointing toward -X with a
        # modest downward pitch. Tweak the values or the Z position to raise/lower tilt.
        "quaternion": (0.6124, 0.3536, 0.3536, 0.6124),
        "fov": 80,
    },
    "left_wrist": {
        "manual": True,
        "parent": "panda nohand_left/left_wrist_link",
        "position": (-0.05, 0.0, 0.00),
        "quaternion": (0, 1, 0, 0),
        "fov": 60,
    },
}

# Base configurations
PANDA_FLOATING_BASE_XYZRZ = FloatingBaseConfig(
    dofs={
        PelvisDof.X: Dof(joint_type=JointType.SLIDE, axis=(1, 0, 0), stiffness=STIFFNESS_XY),
        PelvisDof.Y: Dof(joint_type=JointType.SLIDE, axis=(0, 1, 0), stiffness=STIFFNESS_XY),
        PelvisDof.Z: Dof(
            joint_type=JointType.SLIDE, 
            axis=(0, 0, 1),
            joint_range=RANGE_DOF_Z,
            action_range=RANGE_DOF_Z,
            stiffness=STIFFNESS_Z,
        ),
        PelvisDof.RZ: Dof(joint_type=JointType.HINGE, axis=(0, 0, 1), stiffness=STIFFNESS_XY),
    },
    delta_range_position=(-0.01, 0.01),
    delta_range_rotation=(-0.05, 0.05),
    offset_position=np.array([0, 0, 0]),
)

PANDA_FLOATING_BASE = FloatingBaseConfig(
    dofs=[],  # Fixed base, no floating dofs
    delta_range_position=(-0.01, 0.01),
    delta_range_rotation=(-0.05, 0.05),
    offset_position=np.array([0, 0, 0]),
)

# Gripper configurations
PANDA_GRIPPER = GripperConfig(
    model=THIRD_PARTY_PATH / 'mujoco_menagerie/franka_emika_panda/hand_modified.xml',
    actuators=["actuator8"],
    range=np.array([0, 1]),
    reverse_control=True,
    pad_bodies=["left_finger", "right_finger"],
)

# Robot configurations
PANDA_CONFIG_WITH_PANDA_GRIPPER = RobotConfig(
    delta_range=(-0.1, 0.1),
    position_kp=4500,
    pelvis_body=None,
    floating_base=PANDA_FLOATING_BASE,
    gripper=PANDA_GRIPPER,
    arms={HandSide.LEFT: PANDA_ARM, HandSide.RIGHT: PANDA_ARM},
    arm_offset={HandSide.LEFT: np.array([-0., 0.3, 0.6]), HandSide.RIGHT: np.array([-0., -0.3, 0.6])},
    arm_offset_euler={HandSide.LEFT: np.array([0, 0, 0]), HandSide.RIGHT: np.array([0, 0, 0])},
    actuators=PANDA_ACTUATORS,
    cameras=CAMERA_CONFIG,
    namespaces_to_remove=['key'],
    model_name="Bimanual Panda",
)

PANDA_CONFIG_WITH_PANDA_GRIPPER_LOWER = RobotConfig(
    **{k: v for k, v in PANDA_CONFIG_WITH_PANDA_GRIPPER.__dict__.items() if k != 'arm_offset'},
    arm_offset={HandSide.LEFT: np.array([0, 0.3, 0.2]), HandSide.RIGHT: np.array([0, -0.3, 0.2])},
)

PANDA_CONFIG_WITH_PANDA_GRIPPER_SINGLE = RobotConfig(
    delta_range=(-0.1, 0.1),
    position_kp=4500,
    pelvis_body=None,
    floating_base=PANDA_FLOATING_BASE,
    gripper=PANDA_GRIPPER,
    arms={HandSide.LEFT: PANDA_ARM, HandSide.RIGHT: None},
    arm_offset={HandSide.LEFT: np.array([-0., 0.0, 0.6])},
    arm_offset_euler={HandSide.LEFT: np.array([0, 0, 0])},
    actuators=PANDA_ACTUATORS,
    cameras=CAMERA_CONFIG_SINGLE,
    namespaces_to_remove=['key'],
    model_name="Single Panda",
)


def create_bimanual_panda_config() -> RobotIKConfig:
    """Create config for a bimanual Panda robot."""
    joint_limits = {
        "panda nohand_left/joint1": (-2.8973, 2.8973),
        "panda nohand_left/joint2": (-1.7628, 1.7628),
        "panda nohand_left/joint3": (-2.8973, 2.8973),
        "panda nohand_left/joint4": (-3.0718, -0.0698),
        "panda nohand_left/joint5": (-2.8973, 2.8973),
        "panda nohand_left/joint6": (-0.0175, 3.7525),
        "panda nohand_left/joint7": (-2.8973, 2.8973),
    }
    
    # Add right arm joint limits by copying left arm limits
    right_limits = {k.replace("left", "right"): v for k, v in joint_limits.items()}
    joint_limits.update(right_limits)
    
    return RobotIKConfig(
        robot_prefix="",
        root_body_name="invisible_base",
        torso_name="invisible_base",
        arm_roots=[
            "panda nohand_left\\link0",
            "panda nohand_right\\link0",
        ],
        arm_sites=[
            "panda nohand_left\\attachment_site",
            "panda nohand_right\\attachment_site",
        ],
        joint_limits=joint_limits,
        end_effector_exclude_word="finger",
    )


def create_single_panda_config() -> RobotIKConfig:
    """Create config for a single Panda robot."""
    return RobotIKConfig(
        robot_prefix="",
        root_body_name="invisible_base",
        torso_name="invisible_base",
        arm_roots=["panda nohand_left\\link0"],
        arm_sites=["panda nohand_left\\attachment_site"],
        joint_limits={
            "panda nohand_left/joint1": (-2.8973, 2.8973),
            "panda nohand_left/joint2": (-1.7628, 1.7628),
            "panda nohand_left/joint3": (-2.8973, 2.8973),
            "panda nohand_left/joint4": (-3.0718, -0.0698),
            "panda nohand_left/joint5": (-2.8973, 2.8973),
            "panda nohand_left/joint6": (-0.0175, 3.7525),
            "panda nohand_left/joint7": (-2.8973, 2.8973),
        },
        end_effector_exclude_word="finger",
    )


class BimanualPanda(Robot):
    """Panda Robot with two arms."""

    @property
    def ik_config(self) -> RobotIKConfig:
        """Get robot IK config."""
        return create_bimanual_panda_config()

    @property
    def config(self) -> RobotConfig:
        """Get robot config."""
        return PANDA_CONFIG_WITH_PANDA_GRIPPER
    

class SinglePanda(Robot):
    """Panda Robot with a single arm."""

    @property
    def ik_config(self) -> RobotIKConfig:
        """Get robot IK config."""
        return create_single_panda_config()

    @property
    def config(self) -> RobotConfig:
        """Get robot config."""
        return PANDA_CONFIG_WITH_PANDA_GRIPPER_SINGLE
