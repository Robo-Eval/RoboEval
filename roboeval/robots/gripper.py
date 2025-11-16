"""Robot Gripper module for controlling robotic grippers."""
from typing import Union, Iterable, Optional

import numpy as np
from dm_control import mjcf
from mojo import Mojo
from mojo.elements import Geom, Site, MujocoElement, Body
from mujoco_utils import mjcf_utils

from roboeval.const import HandSide
from roboeval.envs.props.prop import Prop
from roboeval.robots.config import GripperConfig, ArmConfig
from roboeval.utils.physics_utils import get_colliders, has_collided_collections
from scipy.spatial.transform import Rotation

from pyquaternion import Quaternion


class Gripper:
    """Robot Gripper class for controlling a simulated robotic gripper.
    
    This class provides an interface for controlling a gripper including position control,
    collision detection, and state monitoring.
    """

    _NORMAL_RANGE = (0, 1)
    _ROUND_DECIMALS = 1

    def __init__(
        self,
        side: HandSide,
        wrist_site: Site,
        arm_config: ArmConfig,
        gripper_config: GripperConfig,
        mojo: Mojo,
    ):
        """Initialize a gripper instance.
        
        Args:
            side: Which hand side this gripper represents (LEFT/RIGHT)
            wrist_site: The site to attach the gripper to
            arm_config: Configuration for the arm this gripper attaches to
            gripper_config: Configuration for this gripper
            mojo: The Mojo simulation instance
        """
        self._side = side
        self._wrist_site = wrist_site
        self._config = gripper_config
        self._mojo = mojo

        self._pad_bodies: list[Body] = []
        self._pad_geoms: list[Geom] = []
        self._actuators: list[mjcf.Element] = []
        self._actuated_joints: list[mjcf.Element] = []
        self._pinch_site: Optional[Site] = None

        if self._config.model:
            self._body: Body = self._mojo.load_model(
                str(gripper_config.model), self._wrist_site, on_loaded=self._on_loaded
            )
            self._body.mjcf.pos = arm_config.offset_position
            self._body.mjcf.euler = arm_config.offset_euler
            self._mojo.mark_dirty()
        elif self._config.body:
            self._body: Body = Body.get(
                self._mojo, self._config.body, mojo.root_element
            )
            self._process_gripper(mojo.root_element.mjcf, self._body)
        self._pad_geoms = self._get_pad_geoms()

    @property
    def body(self) -> Body:
        """Get the main body of the gripper.
        
        Returns:
            The gripper's body object
        """
        return self._body

    @property
    def wrist_site(self) -> Site:
        """Get the wrist attachment site.
        
        Returns:
            The site where the gripper is attached to the arm
        """
        return self._wrist_site

    @property
    def pinch_position(self) -> np.ndarray:
        """Get the position of the pinch site.
        
        Returns:
            3D position vector of the pinch site
        """
        return self._mojo.physics.bind(self._pinch_site.mjcf).xpos.copy()

    @property
    def wrist_position(self) -> np.ndarray:
        """Get the position of the wrist site.
        
        Returns:
            3D position vector of the wrist site
        """
        return self._mojo.physics.bind(self._wrist_site.mjcf).xpos.copy()
    
    @property
    def wrist_orientation(self) -> Quaternion:
        """Get the wrist site orientation as a quaternion.
        
        Returns:
            Quaternion representing the wrist orientation in [w, x, y, z] format
        """
        xmat = self._mojo.physics.bind(self._wrist_site.mjcf).xmat.copy()
        rot_matrix = np.array(xmat).reshape(3, 3)  # Reshape xmat into 3x3 rotation matrix
        quaternion = Rotation.from_matrix(rot_matrix).as_quat(canonical=True)  # Returns [x, y, z, w] quaternion
        # Return as Quaternion object in the form [w, x, y, z]
        return Quaternion(np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]]))
        
    @property
    def range(self) -> np.ndarray:
        """Get the gripper control range.
        
        Returns:
            Min/max range values for the gripper
        """
        return self._config.range

    @property
    def actuators(self) -> list[mjcf.Element]:
        """Get all gripper actuators.
        
        Returns:
            List of MJCF actuator elements for this gripper
        """
        return self._actuators

    @property
    def qpos(self) -> float:
        """Get the average position of actuated joints.
        
        Returns:
            Normalized position value between the gripper's range limits
        """
        positions = []
        for joint in self._actuated_joints:
            joint = self._mojo.physics.bind(joint)
            positions.append(
                np.interp(joint.qpos.item(), joint.range, self._config.range)
            )
        return np.round(np.average(positions), decimals=self._ROUND_DECIMALS)

    @property
    def qvel(self) -> float:
        """Get the average velocity of gripper actuators.
        
        Returns:
            Average velocity value of the actuated joints
        """
        velocities = []
        for joint in self._actuated_joints:
            joint = self._mojo.physics.bind(joint)
            velocities.append(joint.qvel.item())
        return np.round(np.average(velocities), decimals=self._ROUND_DECIMALS)

    def is_holding_object(self, other: Union[Geom, Iterable[Geom], Prop]) -> bool:
        """Check if gripper is holding an object.
        
        Args:
            other: The object (geometry or prop) to check for collision with the gripper
            
        Returns:
            True if the gripper pads are colliding with the object, False otherwise
        """
        return has_collided_collections(
            self._mojo.physics, self._pad_geoms, get_colliders(other)
        )

    def set_control(self, ctrl: float):
        """Set the control state of the gripper.
        
        Args:
            ctrl: Control value in the gripper's range to set actuator positions
        """
        if self._config.reverse_control:
            ctrl = 1 - ctrl
            
        ctrl = np.interp(ctrl, self._config.range, self._NORMAL_RANGE)
        if self._config.discrete:
            ctrl = np.round(ctrl)
        ctrl = np.interp(ctrl, self._NORMAL_RANGE, self._config.range)
        for actuator in self._actuators:
            actuator = self._mojo.physics.bind(actuator)
            ctrl = np.interp(ctrl, self._config.range, actuator.ctrlrange)
            actuator.ctrl = ctrl
                        
    def _on_loaded(self, model: mjcf.RootElement):
        """Callback when model is loaded.
        
        Args:
            model: The loaded MJCF model
        """
        model.model += f"_{self._side.value.lower()}"
        self._process_gripper(model, MujocoElement(self._mojo, model))

    def _process_gripper(
        self, model: mjcf.RootElement, element: Union[MujocoElement, Body]
    ):
        """Process the gripper model to set up sites, actuators and joints.
        
        Args:
            model: The MJCF root element
            element: The MujocoElement or Body representing the gripper
        """
        if self._config.pinch_site:
            self._pinch_site: Site = Site.get(
                self._mojo, self._config.pinch_site, element
            )
        else:
            # Create a new end effector site if none specified
            if hasattr(element, 'mjcf') and hasattr(element.mjcf, 'tag') and element.mjcf.tag == 'mujoco':
                if hasattr(self, '_body') and self._body is not None:
                    parent = self._body
                else:
                    parent = MujocoElement(self._mojo, model.worldbody)
            else:
                parent = element
            
            self._pinch_site: Site = Site.create(
                self._mojo, parent=parent, size=np.repeat(0.01, 3), group=5
            )

        # Cache gripper pads
        if self._config.pad_bodies:
            for pad_name in self._config.pad_bodies:
                self._pad_bodies.append(Body.get(self._mojo, pad_name, element))

        # Configure actuators and joints
        all_actuators: list[mjcf.Element] = mjcf_utils.safe_find_all(model, "actuator")
        for actuator in all_actuators:
            actuator_name = actuator.name or actuator.joint.name
            if actuator_name not in self._config.actuators:
                continue
            # Caching actuators
            self._actuators.append(actuator)
            # Caching actuated gripper joints
            if actuator.tendon:
                for tendon_joint in actuator.tendon.joint:
                    self._actuated_joints.append(tendon_joint.joint)
            elif actuator.joint:
                self._actuated_joints.append(actuator.joint)

    def _get_pad_geoms(self) -> list[Geom]:
        """Get all collidable geometries in the gripper pads.
        
        Returns:
            List of collidable geometry objects that make up the gripper pads
        """
        geoms = []
        for body in self._pad_bodies:
            geoms.extend([g for g in body.geoms if g.is_collidable()])
        return geoms
