from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Union, Iterable

import numpy as np
from dm_control import mjcf
from mojo import Mojo
from mojo.elements import Body, MujocoElement, Geom, Camera, Joint, Site
from mujoco_utils import mjcf_utils

from roboeval.action_modes import (
    ActionMode,
    JointPositionActionMode,
)
from roboeval.robots.config import ArmConfig, RobotConfig
from roboeval.const import (
    HandSide,
    WORLD_MODEL,
)
from roboeval.envs.props.prop import Prop
from roboeval.robots.floating_base import RobotFloatingBase
from roboeval.robots.gripper import Gripper
from roboeval.utils.physics_utils import (
    get_critical_damping_from_stiffness,
    get_actuator_qpos,
    get_actuator_qvel,
)

import copy

from roboeval.robots.arm import Arm

from scipy.spatial.transform import Rotation as R


class ActuatorType(Enum):
    """Supported body actuator types."""

    POSITION = "position"
    MOTOR = "motor"
    GENERAL = "general"


class Robot(ABC):
    """Abstract base class for robot models in the simulation environment.
    
    This class provides the foundational functionality for all robot types, handling
    robot creation, kinematics, actuator setup, and control. It supports both
    pre-built robot models and compositional robot building.
    
    Key features:
    - Forward and inverse kinematics
    - Floating base control
    - Arm and gripper management
    - Position/torque control modes
    - Keyframe handling for poses
    """

    def __init__(
        self,
        action_mode: ActionMode,
        mojo: Optional[Mojo] = None,
    ):
        """Initialize the robot.
        
        Args:
            action_mode: Defines how the robot can be controlled
            mojo: Optional Mojo instance. If not provided, a new one is created
        """
        self._action_mode = action_mode
        self._mojo = mojo or Mojo(WORLD_MODEL)
        self.keyframes = None
        
        if not self.config.model:  # Composition mode
            self._mjcf_model = self._mojo.root_element.mjcf
            
            attached_model_mjcf = self._setup_pelvis(self._mjcf_model)
            self._body = self._pelvis

            # Set up floating base
            self.set_up_floating_base(self._mjcf_model)

            self.load_arms(parent=self._body)
            self.set_up_grippers()
            
            # Set up actuators
            self.get_keyframes()

            self.cleanup_namespaces(self._mjcf_model)
            self._setup_actuators(self._mjcf_model)
            self._setup_camera(self._body)
        else:
            self._body = self._mojo.load_model(
                str(self.config.model), on_loaded=self._on_loaded
            )

        self._multiarm = len(self._arms) >= 2
        self._grippers = self._get_grippers()
        self._joints = self._get_joints()

        if not self._action_mode.floating_base:
            self._body.set_kinematic(True)

        # Bind robot to action mode
        self._action_mode.bind_robot(self, self._mojo)

        # Get the initial joint position of the robot
        self._initial_qpos = copy.deepcopy(self.qpos_actuated) if not self.keyframes else self.qpos_initial

        # Set up the initial ee positions of the robot
        floating_base_dofs = len(self._action_mode.floating_dofs) if self._action_mode.floating_base else 0
        gripper_dofs = len(self._grippers)
        
        # Combine together the floating base, arm and gripper positions
        floating_base_part = self._initial_qpos[:floating_base_dofs].tolist() if floating_base_dofs > 0 else []
        arm_part = self.forward_kinematics(self._initial_qpos[floating_base_dofs:-gripper_dofs]).tolist()
        gripper_part = self._initial_qpos[-gripper_dofs:].tolist() if gripper_dofs > 0 else []
        self._initial_ee_pos = np.array(floating_base_part + arm_part + gripper_part, dtype=np.float32)

    def get_initial_qpos(self):
        """Get the initial joint positions for the robot.
        
        Returns:
            np.ndarray: Initial joint positions
        """
        return copy.deepcopy(self.qpos_actuated)
    
    @property
    def qpos_initial(self) -> np.ndarray:
        """Get positions of actuated joints based on keyframes.
        
        Returns:
            np.ndarray: Initial joint positions from keyframes
        """
        qpos = []
        if self.floating_base:
            qpos.extend(self._floating_base.qpos)
            
        for actuator in self._limb_actuators:
            # Get name of actuator
            actuator_full_name = actuator.full_identifier
            side = HandSide.LEFT if 'left' in actuator_full_name else HandSide.RIGHT
            arm = self._arms[side]
            arm_actuators = arm._config.actuators
            actuator_name = actuator.name

            if actuator_name in arm_actuators: 
                actuator_keyframe_idx = arm_actuators.index(actuator_name)
                keyframe_value = self.keyframes[side][actuator_keyframe_idx]
            else:
                print(f"Actuator {actuator_name} not found in arm {side}")
                keyframe_value = 0
            qpos.append(keyframe_value)

        qpos.extend(self.qpos_grippers)
        assert len(qpos) == len(self.qpos_actuated)

        return np.array(qpos, np.float32)
    
    def get_keyframes(self, name_identifier='home'):
        """Extract keyframes from the model.
        
        Args:
            name_identifier: Name of the keyframe to extract
        """
        # Get all the keyframes
        scoped_elements_dict = self._mjcf_model.keyframe.key._scoped_elements
        self.keyframes = {}
        name_model = self._mjcf_model.full_identifier

        # Get the right and left arm
        for side, arm in self._arms.items():
            # Find the corresponding keyframe to the arm
            arm_name = arm.body.mjcf.full_identifier
            arm_keys = scoped_elements_dict[arm_name.replace('/', '')]
            home_key = arm_keys[name_identifier]
            qpos = home_key.qpos
            self.keyframes[side] = qpos

    def cleanup_namespaces(self, model: mjcf.RootElement):
        """Remove redundant elements from the model.
        
        Args:
            model: The MJCF model to clean up
        """
        for namespace in self.config.namespaces_to_remove:
            elements = model.find_all(namespace)
            for element in elements:
                element.remove()

    def load_keyframe_from_xml(self, xml_path=None, keyframe_name=None, keyframe_index=0):
        """Load a keyframe from an XML file or extract it from the current model.
        
        Args:
            xml_path: Path to XML file containing keyframe definitions (optional)
            keyframe_name: Name of the keyframe to load (optional)
            keyframe_index: Index of the keyframe to load if name not provided (default=0)
            
        Returns:
            bool: True if keyframe was successfully loaded
        """
        # If XML path is provided, load the external keyframe file
        if xml_path:
            try:
                import pathlib
                # Parse the XML file to extract keyframe data
                keyframe_model = mjcf.from_path(str(pathlib.Path(xml_path).resolve()))
                keyframes = keyframe_model.keyframe
            except Exception as e:
                print(f"Failed to load keyframe from XML: {e}")
                return False
        else:
            # Use keyframes from the current model
            keyframes = self._mojo.model.keyframe
        
        # Find the requested keyframe
        target_keyframe = None
        if keyframe_name:
            for kf in keyframes:
                if kf.name == keyframe_name:
                    target_keyframe = kf
                    break
            if target_keyframe is None:
                print(f"Keyframe '{keyframe_name}' not found")
                return False
        elif keyframes and len(keyframes) > keyframe_index:
            target_keyframe = keyframes[keyframe_index]
        else:
            print(f"No keyframe found at index {keyframe_index}")
            return False
        
        # Extract the joint positions from the keyframe
        keyframe_qpos = np.array(target_keyframe.qpos)
        
        # Map the keyframe joint positions to robot actuators
        actuator_indices = []
        for i, actuator in enumerate(self._limb_actuators):
            if actuator.joint:
                joint_id = self._mojo.physics.bind(actuator.joint).id
                actuator_indices.append(joint_id)
            elif actuator.tendon:
                tendon_id = self._mojo.physics.bind(actuator.tendon).id
                actuator_indices.append(tendon_id)
        
        # Apply keyframe positions to actuated joints
        for i, actuator in enumerate(self._limb_actuators):
            if actuator.joint:
                self._mojo.physics.bind(actuator.joint).qpos = keyframe_qpos[actuator_indices[i]]
            elif actuator.tendon:
                self._mojo.physics.bind(actuator).ctrl = keyframe_qpos[actuator_indices[i]]
        
        # Update physics after setting keyframe
        self._mojo.physics.step()
        
        return True

    def _setup_camera(self, model: mjcf.RootElement):
        """Configure cameras for the robot.
        
        Args:
            model: The MJCF model to add cameras to
        """
        mojo_model = MujocoElement(self._mojo, model)
        # Configure cameras
        self._cameras = []
        root_mjcf = self._mojo.root_element.mjcf

        for camera_name, camera_properties in self.config.cameras.items():
            if camera_properties.get("manual"):
                parent_name = camera_properties.get("parent")
                parent = mjcf_utils.safe_find(root_mjcf, 'body', parent_name)
                if not parent:
                    raise ValueError(f"Parent element {parent_name} not found for camera {camera_name}")
                
                parent = Body(self._mojo, parent)

                position = camera_properties.get("position", [0, 0, 0])
                quaternion = camera_properties.get("quaternion", [1, 0, 0, 0])
                fovy = camera_properties.get("fovy", None)
                focal = camera_properties.get("focal", None)
                sensor_size = camera_properties.get("sensor_size", None)
                
                # Manual camera creation
                camera = Camera.create(
                    self._mojo,
                    parent,
                    position=position,
                    quaternion=quaternion,
                    fovy=fovy,
                    focal=focal,
                    sensor_size=sensor_size,
                    camera_name=camera_name,
                )
                self._cameras.append(camera)
            else:
                self._cameras.append(Camera.get(self._mojo, camera_name, mojo_model))
        
    def _setup_actuators(self, model: mjcf.RootElement):
        """Configure actuators for the robot.
        
        Args:
            model: The MJCF model to configure actuators in
        """
        # List of new actuators with damping to be tuned
        new_actuators = []

        # Setup floating base
        if self._action_mode.floating_base:
            new_actuators.extend(self._floating_base.all_actuators)

        # Assign limb actuators
        self._limb_actuators = []
        all_actuators = mjcf_utils.safe_find_all(model, "actuator")
        actuators_from_arms = set(self.config.actuators.keys())
        
        for actuator in all_actuators:
            actuator_name = actuator.name or actuator.joint.name
            if actuator_name not in actuators_from_arms:
                continue
                
            # Remove actuators not used in floating mode
            if self._floating_base and not self.config.actuators[actuator_name]:
                if actuator.joint:
                    actuator.joint.remove()
                if actuator.tendon:
                    actuator.tendon.remove()
                actuator.remove()
                continue

            if isinstance(self._action_mode, JointPositionActionMode):
                if actuator.tag in [ActuatorType.POSITION.value, ActuatorType.GENERAL.value]:
                    self._limb_actuators.append(actuator)
                else:
                    actuator_name = actuator.name
                    actuator_joint = actuator.joint
                    actuator_tendon = actuator.tendon
                    actuator.remove()

                    actuator = model.actuator.add(
                        "position",
                        name=actuator_name,
                        joint=actuator_joint,
                        tendon=actuator_tendon,
                        kp=self.config.position_kp,
                        ctrlrange=actuator_joint.range if actuator_joint else None,
                    )
                    self._limb_actuators.append(actuator)
                    new_actuators.append(actuator)

        # Sort limb actuators according to the joints tree
        joints = mjcf_utils.safe_find_all(model, "joint")
        self._limb_actuators.sort(key=lambda a: joints.index(a.joint) if a.joint else 0)

        # Temporary instance of physics to simplify model editing
        physics_tmp = mjcf.Physics.from_mjcf_model(model)

        # Fix joint damping
        for actuator in new_actuators:
            damping = get_critical_damping_from_stiffness(
                actuator.kp, actuator.joint.full_identifier, physics_tmp
            )
            actuator.joint.damping = damping

    def _setup_pelvis(self, model: mjcf.RootElement):
        """Create the robot pelvis.
        
        Args:
            model: The MJCF model to add the pelvis to
            
        Returns:
            mjcf.Element: The created pelvis element
        """
        # Create the invisible base first
        pelvis = model.worldbody.add(
            "body", 
            name="invisible_base",
            pos=[0, 0, 0]
        )
        pelvis.add(
            'geom', name='torso', type='box', size=[0.2, 0.6, 0.6])
        
        # Add minimal inertia to the base
        pelvis.add(
            "inertial",
            pos=[0, 0, 0],
            mass="0.01",  # Very light mass
            diaginertia="0.01 0.01 0.01"  # Minimal inertia
        )

        self._mojo.mark_dirty()
        self._pelvis = Body(self._mojo, pelvis)

        # Always remove free joints
        if self._pelvis.is_kinematic():
            self._pelvis.set_kinematic(False)

        return pelvis
    
    def load_arms(self, parent=None):
        """Load and configure robot arms.
        
        Args:
            parent: Parent body to attach the arms to
        """
        # Set up arm sites
        self._arms = {}
        for side, arm_config in self.config.arms.items():
            if arm_config is None:
                continue

            position = self.config.arm_offset[side]
            euler = self.config.arm_offset_euler[side]

            position = np.array(position) if not isinstance(position, np.ndarray) else position
            euler = np.array(euler) if not isinstance(euler, np.ndarray) else euler

            # Create a base site for the arm if parent is provided
            base_site = None
            if parent is not None:
                # First, make sure we have a body to attach the site to
                if hasattr(parent.mjcf, 'worldbody'):
                    # If parent is the root element, use worldbody
                    parent_body = parent.mjcf.worldbody
                elif hasattr(parent.mjcf, 'body'):
                    # Try to use a body element if available
                    parent_body = parent.mjcf
                else:
                    # Assume parent is already a body element
                    parent_body = parent.mjcf
                    
                # Create a site at the specified offset position/orientation to attach the arm
                site_name = f"{side.name.lower()}_arm_mount"
                base_site = parent_body.add('site', name=site_name, pos=position, euler=euler)
                base_site = Site.get(self._mojo, site_name, parent)  # Get mojo wrapper for the site

            arm = Arm(side=side, base_site=base_site, arm_config=arm_config, mojo=self._mojo)
            self._arms[side] = arm

    def set_up_floating_base(self, model: mjcf.RootElement):
        """Configure the floating base for the robot.
        
        Args:
            model: The MJCF model to add the floating base to
        """
        self._floating_base = None
        if self._action_mode.floating_base:
            self._floating_base = RobotFloatingBase(
                self.config.floating_base,
                self._pelvis,
                self._action_mode.floating_dofs,
                model,
                self._mojo,
            )

    def set_up_grippers(self):
        """Configure grippers for the robot arms."""
        # Configure wrist sites
        self._wrist_sites = {}
        for side, arm in self._arms.items():
            arm_config = self.config.arms[side]

            model = self._mjcf_model
            mojo_model = MujocoElement(self._mojo, model)
            site_name = f"{arm.body.mjcf.full_identifier}{arm_config.site}"
            
            # Configure wrist joints
            self._add_wrist(model, side, arm_config, site_name=site_name)

            self._wrist_sites[side] = Site.get(
                self._mojo, site_name, mojo_model
            )
            
    def get_initial_joint_positions(self):
        """Get the initial position for all actuator joints.
        
        Returns:
            list: Initial joint positions
        """
        initial_joint_positions = []

        for actuator in self.limb_actuators:
            actuator = self._mojo.physics.bind(actuator)
            initial_joint_positions.append(copy.deepcopy(actuator.ctrl))

        return initial_joint_positions

    def inverse_kinematics(
        self, 
        target_poses: np.ndarray, 
        initial_joint_positions: Optional[np.ndarray] = None,
        max_iterations: int = 50,
        tolerance: float = 1e-3,
        damping: float = 0.5,
        debug: bool = False,
        weight_factor: float = 1.0
    ) -> np.ndarray:
        """Compute inverse kinematics to find joint positions for target end-effector poses.
        
        Args:
            target_poses: Target end-effector poses [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, ...]
            initial_joint_positions: Starting joint positions for IK (optional)
            max_iterations: Maximum iterations for IK algorithm
            tolerance: Convergence tolerance
            damping: Damping factor for numerical stability
            debug: Whether to print debug information
            weight_factor: Weight factor for orientation error
            
        Returns:
            np.ndarray: Joint positions that achieve the target poses
        """
        # Set up initial joint positions if not provided
        if initial_joint_positions is None:
            initial_joint_positions = np.array([get_actuator_qpos(actuator, self._mojo.physics) 
                                          for actuator in self._limb_actuators], dtype=np.float32)
        else:
            initial_joint_positions = np.array(initial_joint_positions, dtype=np.float32)
        
        # Ensure target_poses has the right format
        target_poses = np.array(target_poses, dtype=np.float32)
        num_ee = len(self._wrist_sites)
        expected_length = num_ee * 6  # 6 values per end-effector (pos + euler)
        assert len(target_poses) == expected_length, f"Expected {expected_length} values in target_poses"
        
        # Store original joint positions to restore later
        original_positions = {}
        for i, actuator in enumerate(self._limb_actuators):
            if actuator.joint:
                original_positions[actuator.joint] = self._mojo.physics.bind(actuator.joint).qpos
        
        try:
            # Make a copy of the initial joint positions
            joint_positions = initial_joint_positions.copy()
            
            # Pre-allocate arrays to avoid repeated memory allocations
            jacobian = np.zeros((num_ee * 6, len(joint_positions)))
            pos_error = np.zeros(num_ee * 3)
            orientation_error = np.zeros(num_ee * 3)
            delta = 1e-3
            
            # This lets us quickly revert to a clean state when computing the Jacobian
            physics_state = self._mojo.physics.get_state()
            
            # Iteratively update joint positions to minimize pose error
            for iteration in range(max_iterations):
                # Get current end-effector poses for these joint positions
                current_poses = self.forward_kinematics(joint_positions)
                
                # Calculate position error for all end-effectors
                for i in range(num_ee):
                    pos_error[i*3:(i+1)*3] = target_poses[i*6:i*6+3] - current_poses[i*6:i*6+3]
                
                # Handle orientation using rotation matrices
                for i in range(num_ee):
                    target_euler = target_poses[i*6+3:(i+1)*6]
                    current_euler = current_poses[i*6+3:(i+1)*6]
                    
                    # Convert to rotation matrices
                    target_rot = R.from_euler('xyz', target_euler).as_matrix()
                    current_rot = R.from_euler('xyz', current_euler).as_matrix()
                    
                    # Calculate the difference rotation matrix
                    diff_rot = target_rot @ current_rot.T
                    
                    # Convert to axis-angle for the error vector
                    orientation_error[i*3:(i+1)*3] = R.from_matrix(diff_rot).as_rotvec() * weight_factor
                
                # Combine position and orientation errors
                pose_error = np.concatenate([pos_error, orientation_error])
                error_norm = np.linalg.norm(pose_error)
                
                # Check convergence
                if error_norm < tolerance:
                    if debug:
                        print(f"Converged after {iteration} iterations with error {error_norm}")
                    break
                
                if debug and iteration % 10 == 0:
                    print(f"Iteration {iteration}, error: {error_norm}")
                
                # Compute Jacobian with finite differences
                for j in range(len(joint_positions)):
                    perturbed_joints_plus = joint_positions.copy()
                    perturbed_joints_plus[j] += delta
                    perturbed_joints_minus = joint_positions.copy()
                    perturbed_joints_minus[j] -= delta
                    
                    poses_plus = self.forward_kinematics(perturbed_joints_plus)
                    poses_minus = self.forward_kinematics(perturbed_joints_minus)
                    
                    # Vectorized computation for position Jacobian
                    for i in range(num_ee):
                        pos_idx = i*6
                        jacobian_pos_idx = i*3
                        jacobian_ori_idx = num_ee*3 + i*3
                        
                        # Position component
                        jacobian[jacobian_pos_idx:jacobian_pos_idx+3, j] = (
                            poses_plus[pos_idx:pos_idx+3] - poses_minus[pos_idx:pos_idx+3]
                        ) / (2 * delta)
                        
                        # Orientation component using matrices
                        ori_idx = i*6 + 3
                        rot_plus = R.from_euler('xyz', poses_plus[ori_idx:ori_idx+3]).as_matrix()
                        rot_minus = R.from_euler('xyz', poses_minus[ori_idx:ori_idx+3]).as_matrix()
                        diff_rot = rot_plus @ rot_minus.T
                        rot_vec = R.from_matrix(diff_rot).as_rotvec() / (2 * delta)
                        jacobian[jacobian_ori_idx:jacobian_ori_idx+3, j] = rot_vec
                
                # Use damped least squares with single SVD computation
                try:
                    u, s, vh = np.linalg.svd(jacobian, full_matrices=False)
                    s_damped = s / (s**2 + damping**2)
                    J_damped_inv = vh.T @ np.diag(s_damped) @ u.T
                    update = J_damped_inv @ pose_error
                    
                    # Adaptive step size to prevent large updates
                    step_size = min(0.5, 0.2 / (np.linalg.norm(update) + 1e-6))
                    joint_positions += step_size * update
                    
                    if debug and iteration % 5 == 0:
                        print(f"Iter {iteration} | Error: {error_norm:.6f} | Update: {np.linalg.norm(update):.6f}")
                
                except np.linalg.LinAlgError:
                    # Fallback to pseudoinverse
                    J_pinv = np.linalg.pinv(jacobian, rcond=1e-3)
                    update = J_pinv @ pose_error
                    joint_positions += 0.3 * update
                
                # Apply joint limits if available
                for i, actuator in enumerate(self._limb_actuators):
                    if actuator.joint and hasattr(actuator.joint, 'range') and actuator.joint.range is not None:
                        joint_limits = actuator.joint.range
                        if joint_limits is not None:
                            joint_positions[i] = np.clip(joint_positions[i], joint_limits[0], joint_limits[1])
            
            if iteration == max_iterations - 1 and debug:
                print(f"Failed to converge after {max_iterations} iterations. Final error: {error_norm}")
                
            return joint_positions
        
        finally:
            # Restore original joint positions
            for joint, pos in original_positions.items():
                self._mojo.physics.bind(joint).qpos = pos
            
            # Update physics to restore state
            self._mojo.physics.forward()

    def forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        """Compute forward kinematics to get end-effector poses from joint positions.
        
        Args:
            joint_positions: Joint positions for the robot arms (excluding floating base and grippers)
            
        Returns:
            np.ndarray: End-effector poses for all arms, containing positions and orientations
        """
        # Ensure joint positions are in the correct format
        joint_positions = np.array(joint_positions, dtype=np.float32)
        assert len(joint_positions) == len(self._limb_actuators), (
            f"Invalid number of joint positions. {len(joint_positions)} "
            f"Expected {len(self._limb_actuators)} positions."
        )
        
        # Store original joint positions to restore later
        original_positions = {}
        for i, actuator in enumerate(self._limb_actuators):
            if actuator.joint:
                original_positions[actuator.joint] = copy.deepcopy(self._mojo.physics.bind(actuator.joint).qpos)
        
        try:
            # Set the joint positions
            for i, actuator in enumerate(self._limb_actuators):
                if actuator.joint:
                    self._mojo.physics.bind(actuator.joint).qpos = joint_positions[i]
                else:
                    raise ValueError(
                        f"Joint position {i} is out of range. Expected {len(self._limb_actuators)} positions."
                    )
            # Update forward kinematics
            self._mojo.physics.forward()
            
            # Get end-effector poses
            ee_poses = []
            for side in self._wrist_sites.keys():
                wrist_site = self._wrist_sites[side]
                position = wrist_site.get_position()
                orientation = wrist_site.get_quaternion()  # wxyz array

                # Get position and orientation relative to the pelvis
                if self._pelvis is not None:
                    pelvis_pos = self._pelvis.get_position()
                    pelvis_quat = self._pelvis.get_quaternion()
                    position -= pelvis_pos
                    rot_pelvis = R.from_quat(pelvis_quat, scalar_first=True)
                    rot_ee = R.from_quat(orientation, scalar_first=True)
                    rot_relative = rot_ee * rot_pelvis.inv()
                    orientation = rot_relative.as_euler('xyz')
                else:
                    # Convert wxyz orientation to rpy
                    orientation = R.from_quat(orientation, scalar_first=True).as_euler('xyz')
                orientation = (orientation + np.pi) % (2 * np.pi) - np.pi
                ee_poses.extend([*position, *orientation])  # [x,y,z, rx,ry,rz]
            
            return np.array(ee_poses, dtype=np.float32)
        finally:
            # Restore original joint positions
            for joint, pos in original_positions.items():
                self._mojo.physics.bind(joint).qpos = pos
            
            # Update physics to restore state
            self._mojo.physics.forward()
    
    def reset_pose(self):
        """Reset the robot's pose to its initial state."""
        assert len(self._initial_qpos) == (
            len(self._floating_base.qpos if self._floating_base else []) + 
            len(self._limb_actuators) + 
            len(self.qpos_grippers)
        ), "Initial qpos does not match the number of actuators."
        
        # Reset actuated joint positions
        floating_base_len = len(self._floating_base.qpos) if self._floating_base else 0
        for i, actuator in enumerate(self._limb_actuators):
            index_corrected = i + floating_base_len
            
            if actuator.joint:
                self._mojo.physics.bind(actuator.joint).qpos = self._initial_qpos[index_corrected]
            elif actuator.tendon:
                self._mojo.physics.bind(actuator).ctrl = self._initial_qpos[index_corrected]
            else:
                print(f"Actuator {actuator} is not supported for resetting.")
            
            actuator = self._mojo.physics.bind(actuator)
            actuator.ctrl = self._initial_qpos[index_corrected]
        
        # Ensure physics update
        self._mojo.physics.step()

    @property
    @abstractmethod
    def config(self) -> RobotConfig:
        """Get robot configuration.
        
        Returns:
            RobotConfig: The robot's configuration
        """
        pass

    @property
    def action_mode(self) -> ActionMode:
        """Get the robot's action mode.
        
        Returns:
            ActionMode: The robot's action mode
        """
        return self._action_mode

    @property
    def pelvis(self) -> Body:
        """Get the robot's pelvis body.
        
        Returns:
            Body: The pelvis body
        """
        return self._pelvis

    @property
    def limb_actuators(self) -> list[mjcf.Element]:
        """Get all limb actuators.
        
        Returns:
            list[mjcf.Element]: List of limb actuators
        """
        return self._limb_actuators

    @property
    def grippers(self) -> dict[HandSide, Gripper]:
        """Get robot grippers.
        
        Returns:
            dict[HandSide, Gripper]: Dictionary of grippers by hand side
        """
        return self._grippers

    @property
    def floating_base(self) -> Optional[RobotFloatingBase]:
        """Get the robot's floating base.
        
        Returns:
            Optional[RobotFloatingBase]: The floating base, if available
        """
        return self._floating_base

    @property
    def cameras(self) -> list[Camera]:
        """Get robot cameras.
        
        Returns:
            list[Camera]: List of cameras
        """
        return self._cameras

    @property
    def qpos(self) -> np.ndarray:
        """Get positions of all joints.
        
        Returns:
            np.ndarray: Joint positions
        """
        return np.array(
            [joint.get_joint_position() for joint in self._joints], np.float32
        )

    @property
    def qpos_grippers(self) -> np.ndarray:
        """Get current state of gripper actuators.
        
        Returns:
            np.ndarray: Gripper positions
        """
        qpos = []
        for _, gripper in self.grippers.items():
            qpos.append(gripper.qpos)
        return np.array(qpos, np.float32)

    @property
    def qpos_actuated(self) -> np.ndarray:
        """Get positions of actuated joints.
        
        Returns:
            np.ndarray: Actuated joint positions
        """
        qpos = []
        if self.floating_base:
            qpos.extend(self._floating_base.qpos)
        for actuator in self._limb_actuators:
            qpos.append(get_actuator_qpos(actuator, self._mojo.physics))
        qpos.extend(self.qpos_grippers)
        return np.array(qpos, np.float32)

    @property
    def qvel(self) -> np.ndarray:
        """Get velocities of all joints.
        
        Returns:
            np.ndarray: Joint velocities
        """
        return np.array(
            [joint.get_joint_velocity() for joint in self._joints], np.float32
        )

    @property
    def qvel_actuated(self) -> np.ndarray:
        """Get velocities of actuated joints.
        
        Returns:
            np.ndarray: Actuated joint velocities
        """
        qvel = []
        if self.floating_base:
            qvel.extend(self._floating_base.qvel)
        for actuator in self._limb_actuators:
            qvel.append(get_actuator_qvel(actuator, self._mojo.physics))
        for _, gripper in self._grippers.items():
            qvel.append(gripper.qvel)
        return np.array(qvel, np.float32)

    def get_hand_pos(self, side: HandSide) -> np.ndarray:
        """Get position of robot hand site.
        
        Args:
            side: Which hand to get the position of
            
        Returns:
            np.ndarray: Hand position
        """
        if side not in self.config.arms.keys():
            return np.zeros(3)
        return self._grippers[side].wrist_position

    def is_gripper_holding_object(
        self, other: Union[Geom, Iterable[Geom], Prop], side: HandSide
    ) -> bool:
        """Check if the gripper is holding an object.
        
        Args:
            other: Object to check
            side: Which gripper to check
            
        Returns:
            bool: True if the gripper is holding the object
        """
        if side not in self.config.arms.keys():
            return False
        return self._grippers[side].is_holding_object(other)

    def set_pose(self, position: np.ndarray, orientation: np.ndarray):
        """Instantly set pose of the robot pelvis.
        
        Args:
            position: Position to set
            orientation: Orientation to set (quaternion)
        """
        if self._action_mode.floating_base:
            self._floating_base.reset(position, orientation)
        else:
            self._pelvis.set_position(position)
            self._pelvis.set_quaternion(orientation)

    def get_limb_control_range(
        self, actuator: mjcf.Element, absolute: bool
    ) -> np.ndarray:
        """Get control range of the limb actuator.
        
        Args:
            actuator: The actuator to get the range for
            absolute: Whether to return absolute range or delta range
            
        Returns:
            np.ndarray: Control range
        """
        if not absolute:
            return np.array(self.config.delta_range)
        else:
            return self._mojo.physics.bind(actuator).ctrlrange
        
    def _get_arms(self) -> dict[HandSide, Arm]:
        """Get robot arms.
        
        Returns:
            dict[HandSide, Arm]: Dictionary of arms by hand side
        """
        arms = {}
        # Check if self._arms is already set
        if self._arms:
            return self._arms
        for side, arm_config in self.config.arms.items():
            pass  # This method isn't fully implemented
        return arms

    def _get_grippers(self) -> dict[HandSide, Gripper]:
        """Get robot grippers.
        
        Returns:
            dict[HandSide, Gripper]: Dictionary of grippers by hand side
        """
        grippers = {}
        for side, arm_config in self.config.arms.items():
            if arm_config is None:
                continue
            grippers[side] = Gripper(
                side,
                self._wrist_sites[side],
                arm_config,
                self.config.gripper,
                self._mojo,
            )
        return grippers

    def _get_joints(self) -> list[Joint]:
        """Get robot joints.
        
        Returns:
            list[Joint]: List of robot joints
        """
        return list(self._body.joints)

    def _on_loaded(self, model: mjcf.RootElement):
        """Callback for when the model is loaded.
        
        Args:
            model: The loaded MJCF model
        """
        mojo_model = MujocoElement(self._mojo, model)

        # Remove all redundant elements from the model
        for namespace in self.config.namespaces_to_remove:
            elements = model.find_all(namespace)
            for element in elements:
                element.remove()

        # Configure cameras
        self._cameras = [
            Camera.get(self._mojo, camera, mojo_model) for camera in self.config.cameras
        ]

        # Configure wrist joints
        self._wrist_sites = {}
        for side, arm_config in self.config.arms.items():
            if arm_config.wrist_dof:
                self._add_wrist(model, side, arm_config)
            self._wrist_sites[side] = Site.get(
                self._mojo, arm_config.site, mojo_model
            )

        # Configure pelvis
        self._pelvis = Body.get(self._mojo, self.config.pelvis_body, mojo_model)

        # Always remove free joints
        if self._pelvis.is_kinematic():
            self._pelvis.set_kinematic(False)

        # Reset initial position of pelvis
        self._pelvis.mjcf.pos = np.zeros(3)
        self._pelvis.mjcf.euler = np.zeros(3)

        # Setup floating base and actuators (using the same logic as in _setup_actuators)
        new_actuators = []
        self._floating_base = None
        if self._action_mode.floating_base:
            self._floating_base = RobotFloatingBase(
                self.config.floating_base,
                self._pelvis,
                self._action_mode.floating_dofs,
                model,
                self._mojo,
            )
            new_actuators.extend(self._floating_base.all_actuators)

        # Set up actuators (same as in _setup_actuators)
        self._limb_actuators = []
        all_actuators = mjcf_utils.safe_find_all(model, "actuator")
        for actuator in all_actuators:
            actuator_name = actuator.name or actuator.joint.name
            if actuator_name not in self.config.actuators.keys():
                continue
            
            # Remove actuators not used in floating mode
            if self._floating_base and not self.config.actuators[actuator_name]:
                if actuator.joint:
                    actuator.joint.remove()
                if actuator.tendon:
                    actuator.tendon.remove()
                actuator.remove()
                continue
                
            if isinstance(self._action_mode, JointPositionActionMode):
                if actuator.tag in [ActuatorType.POSITION.value, ActuatorType.GENERAL.value]:
                    self._limb_actuators.append(actuator)
                else:
                    actuator_name = actuator.name
                    actuator_joint = actuator.joint
                    actuator_tendon = actuator.tendon
                    actuator.remove()

                    actuator = model.actuator.add(
                        "position",
                        name=actuator_name,
                        joint=actuator_joint,
                        tendon=actuator_tendon,
                        kp=self.config.position_kp,
                        ctrlrange=actuator_joint.range if actuator_joint else None,
                    )
                    self._limb_actuators.append(actuator)
                    new_actuators.append(actuator)

        # Sort limb actuators according to the joints tree
        joints = mjcf_utils.safe_find_all(model, "joint")
        self._limb_actuators.sort(key=lambda a: joints.index(a.joint) if a.joint else 0)

        # Temporary instance of physics to simplify model editing
        physics_tmp = mjcf.Physics.from_mjcf_model(model)

        # Fix joint damping
        for actuator in new_actuators:
            damping = get_critical_damping_from_stiffness(
                actuator.kp, actuator.joint.full_identifier, physics_tmp
            )
            actuator.joint.damping = damping

    @staticmethod
    def _add_wrist(model: mjcf.RootElement, side: HandSide, arm_config: ArmConfig, site_name: str = None):
        """Add a wrist joint to the robot arm.
        
        Args:
            model: The MJCF model
            side: Which side the wrist is on
            arm_config: Arm configuration
            site_name: Name of the site to use (optional)
        """
        join_name = f"{side.name.lower()}_wrist"

        if not site_name:
            site_name = arm_config.site
            site_name_short = arm_config.site
        else:
            site_name_short = arm_config.site

        site = mjcf_utils.safe_find(model, "site", site_name)
        if site.pos is None or len(site.pos) != 3:
            site_pos = np.array([0, 0, 0])
        else:
            site_pos = np.array(site.pos)

        site_parent = site.parent
        site.remove()

        wrist = site_parent.add("body", name=f"{join_name}_link", pos=site_pos)
        wrist.add(
            "inertial", pos="0 0 0", mass="1e-15", diaginertia="1e-15 1e-15 1e-15"
        )
        wrist.add("site", name=site_name_short)

        if arm_config.wrist_dof:
            joint = wrist.add(
                "joint",
                type=arm_config.wrist_dof.joint_type.value,
                name=join_name,
                axis=arm_config.wrist_dof.axis,
                range=arm_config.wrist_dof.joint_range,
            )
            model.actuator.add(
                "motor",
                name=join_name,
                joint=joint,
                ctrlrange=arm_config.wrist_dof.action_range,
            )
