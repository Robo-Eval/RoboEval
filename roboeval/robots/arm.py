"""Robot Arm."""
from typing import Union, Iterable, Optional

import numpy as np
from dm_control import mjcf
from mojo import Mojo
from mojo.elements import Geom, Site, MujocoElement, Body
from mujoco_utils import mjcf_utils

from roboeval.const import HandSide
from roboeval.robots.config import ArmConfig, GripperConfig
from roboeval.robots.gripper import Gripper
from scipy.spatial.transform import Rotation

from pyquaternion import Quaternion

class Arm:
    """Robot Arm."""

    def __init__(
        self,
        side: HandSide,
        base_site: Site,  # Connection point for the arm
        arm_config: ArmConfig,
        mojo: Mojo,
    ):
        """Init."""
        self._side = side
        self._base_site = base_site
        self._config = arm_config
        self._mojo = mojo

        # Components to track
        self._joints: list[mjcf.Element] = []  # All arm joints
        self._actuators: list[mjcf.Element] = []  # All arm actuators
        self._bodies: list[Body] = []  # All arm bodies
        self._end_effector_site: Optional[Site] = None  # End effector (wrist) site

        # Load the arm model
        if self._config.model:
            self._body: Body = self._mojo.load_model(
                str(arm_config.model), self._base_site, on_loaded=self._on_loaded
            )
            self._mojo.mark_dirty()
        elif self._config.body:
            self._body: Body = Body.get(
                self._mojo, self._config.body, mojo.root_element
            )
            self._process_arm(mojo.root_element.mjcf, self._body)
        
        # Attached gripper
        self._gripper: Optional[Gripper] = None

    @property
    def body(self) -> Body:
        """Get arm base body."""
        return self._body

    @property
    def base_site(self) -> Site:
        """Get base site."""
        return self._base_site

    @property
    def end_effector_site(self) -> Site:
        """Get end effector site."""
        return self._end_effector_site

    @property
    def end_effector_position(self) -> np.ndarray:
        """Get position of the end effector site."""
        return self._mojo.physics.bind(self._end_effector_site.mjcf).xpos.copy()

    @property
    def end_effector_orientation(self) -> np.ndarray:
        """Get end effector orientation as a quaternion."""
        xmat = self._mojo.physics.bind(self._end_effector_site.mjcf).xmat.copy()
        rot_matrix = np.array(xmat).reshape(3, 3)
        quaternion = Rotation.from_matrix(rot_matrix).as_quat()
        return Quaternion(np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]]))

    @property
    def actuators(self) -> list[mjcf.Element]:
        """Get list of arm actuators."""
        return self._actuators
    
    @property
    def joints(self) -> list[mjcf.Element]:
        """Get list of arm joints."""
        return self._joints

    @property
    def qpos(self) -> np.ndarray:
        """Get joint positions of the arm."""
        positions = []
        for joint in self._joints:
            joint = self._mojo.physics.bind(joint)
            positions.append(joint.qpos.item())
        return np.array(positions)

    @property
    def qvel(self) -> np.ndarray:
        """Get joint velocities of the arm."""
        velocities = []
        for joint in self._joints:
            joint = self._mojo.physics.bind(joint)
            velocities.append(joint.qvel.item())
        return np.array(velocities)
    
    @property
    def gripper(self) -> Optional[Gripper]:
        """Get attached gripper if any."""
        return self._gripper
    
    def attach_gripper(self, gripper_config: GripperConfig) -> Gripper:
        """Attach a gripper to the end effector."""
        self._gripper = Gripper(
            side=self._side,
            wrist_site=self._end_effector_site,
            arm_config=self._config,
            gripper_config=gripper_config,
            mojo=self._mojo
        )
        return self._gripper

    def set_control(self, ctrl: np.ndarray):
        """Set control signals for arm joints."""
        # Apply control to each actuator
        for i, actuator in enumerate(self._actuators):
            if i < len(ctrl):
                actuator = self._mojo.physics.bind(actuator)
                actuator.ctrl = ctrl[i]

    def get_keyframes(self) -> dict[str, np.ndarray]:
        """Get keyframes for arm joints."""
        keyframes = {}
        
        return keyframes


    def _on_loaded(self, model: mjcf.RootElement):
        model.model += f"_{self._side.value.lower()}"
        self._process_arm(model, MujocoElement(self._mojo, model))

    def _process_arm(self, model: mjcf.RootElement, element: Union[MujocoElement | Body]):
        # Set up end effector site
        if self._config.site:
            self._end_effector_site = Site.get(
                self._mojo, self._config.site, element
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
                
            self._end_effector_site = Site.create(
                self._mojo, parent=parent, size=np.repeat(0.01, 3), group=5
            )

        # Collect all bodies
        bodies = mjcf_utils.safe_find_all(model, "body")
        for body in bodies:
            if body.name in self._config.links:
                self._bodies.append(Body.get(self._mojo, body.name, element))

        # Configure actuators and joints
        all_actuators = mjcf_utils.safe_find_all(model, "actuator")
        for actuator in all_actuators:
            actuator_name = actuator.name or actuator.joint.name
            if actuator_name in self._config.actuators:
                # Cache actuator
                self._actuators.append(actuator)
        
        # Find all arm joints
        all_joints = mjcf_utils.safe_find_all(model, "joint")
        for joint in all_joints:
            if joint.name in self._config.joints:
                self._joints.append(joint)
                
    def forward_kinematics(self, joint_angles: np.ndarray) -> tuple[np.ndarray, Quaternion]:
        """
        Compute forward kinematics to get end-effector position and orientation.
        
        Args:
            joint_angles: Array of joint angles
            
        Returns:
            Tuple of (position, orientation)
        """
        # Save current state
        old_qpos = self.qpos.copy()
        
        # Set the provided joint angles
        physics = self._mojo.physics
        for i, joint in enumerate(self._joints):
            if i < len(joint_angles):
                physics.bind(joint).qpos = joint_angles[i]
        
        # Forward the simulation one step without dynamics
        physics.forward()
        
        # Get the resulting end-effector pose
        position = self.end_effector_position
        orientation = self.end_effector_orientation
        
        # Restore original state
        for i, joint in enumerate(self._joints):
            physics.bind(joint).qpos = old_qpos[i]
        physics.forward()
        
        return position, orientation

if __name__ == "__main__":

    from roboeval.robots.configs.panda import PANDA_ARM, PANDA_GRIPPER, PandaWithPandaGripper, PANDA_CONFIG_WITH_PANDA_GRIPPER
    from mojo import Mojo
    from roboeval.const import (
    HandSide,
    WORLD_MODEL,
    )
    from roboeval.action_modes import (
    JointPositionActionMode,
)
    
    # Initialize the physics with our model
    mojo = Mojo(WORLD_MODEL)

    base_site = Site.create(mojo, group=5)

    # Get the site object from the element
    # base_site = Site.get(mojo, 'arm_attachment', mojo.root_element)
    
    arm_config = PANDA_ARM

    gripper_config = PANDA_GRIPPER

    # Instantiate the arm
    arm = Arm(
        side=HandSide.RIGHT,  # Right arm
        base_site=base_site,   # Attach to the base site
        arm_config=arm_config,
        mojo=mojo
    )

     # Attach a gripper to the arm
    gripper = arm.attach_gripper(gripper_config)

    from roboeval.envs.kitchen import KitchenEnv

    env = KitchenEnv(
        action_mode=JointPositionActionMode(floating_base=False),        
        render_mode="human",
        robot_cls=arm,
    )

    _ = env.reset()
    for _ in range(1000):
        _, _, _, _, _ = env.step(env.action_space.sample())
        env.render()

    env.close()