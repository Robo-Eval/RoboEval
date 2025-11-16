# Standard library imports
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import warnings

# Third-party imports
from dm_control import mjcf
from dm_control.utils.inverse_kinematics import qpos_from_site_pose
import mujoco
import numpy as np
from lxml import etree
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

# Project-specific imports
from mujoco_utils import collision_utils, mjcf_utils, physics_utils
from roboeval.robots.config import RobotIKConfig

# Constants
WORLDBODY = "worldbody"


@dataclass
class Pose:
    """Pose represented by position vector and quaternion orientation."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: Quaternion = field(default_factory=lambda: Quaternion())


class UpperBodyIK(ABC):
    """Abstract base class for upper body IK solvers."""

    @abstractmethod
    def solve(
        self,
        root_pose: Pose,
        qpos_arms: list[np.ndarray],
        target_poses: list[Pose],
    ) -> np.ndarray:
        """
        Solve IK for arms given target poses.
        
        Args:
            root_pose: Pose of the robot's root
            qpos_arms: List of joint positions for each arm
            target_poses: List of target poses for each arm
            
        Returns:
            Joint positions that achieve the target poses
        """
        pass

    @abstractmethod
    def _generate_ee_actuators(self, site: str, origin: str):
        """
        Generate end-effector actuators for the given site.
        
        Args:
            site: Site name for the actuator
            origin: Reference site name
            
        Returns:
            List of actuators for the end effector
        """
        pass


class GenericUpperBodyIK(UpperBodyIK):
    """Generic upper body IK solver."""

    def __init__(self, env, config: RobotIKConfig):
        """
        Initialize the IK solver with a robot configuration.
        
        Args:
            env: Environment containing the robot model
            config: Robot IK configuration
        """
        self._config = config
        base_model = env.mojo.root_element.mjcf
        base_xml = base_model.to_xml()

        # Removing all except robot parts
        for elem in base_xml.find(WORLDBODY):
            if not elem.attrib["name"].startswith(config.robot_prefix):
                elem.getparent().remove(elem)

        self._model = mjcf.from_xml_string(
            xml_string=etree.tostring(base_xml),
            escape_separators=True,
            assets=base_model.get_assets(),
        )

        # Remove unnecessary features
        features_to_remove = {"key", "actuator", "tendon", "contact", "equality"}
        for feature in features_to_remove:
            try:
                elements = mjcf_utils.safe_find_all(self._model, feature)
                for element in elements:
                    element.remove()
            except ValueError:
                pass

        # Fix root body
        self._root_body = mjcf_utils.safe_find(self._model, "body", config.root_body_name)
        if self._root_body.freejoint:
            self._root_body.freejoint.remove()

        # Find limb roots
        self._torso = mjcf_utils.safe_find(self._model, "body", config.torso_name)

        self._shoulders = []
        for arm_root in config.arm_roots:
            shoulder = mjcf_utils.safe_find(self._model, "body", arm_root)
            if shoulder:
                self._shoulders.append(shoulder)

        self._arm_sites = []
        for arm_site in config.arm_sites:
            site = mjcf_utils.safe_find(self._model, "site", arm_site)
            if site:
                self._arm_sites.append(site)

        self._setup_joints()
        self._setup_actuators()
        self._setup_physics()
    
    def solve(
        self,
        root_pose: Pose,
        qpos_arms: list[np.ndarray],
        target_poses: list[Pose],
    ) -> np.ndarray:
        """
        Solve IK for both arms using the MuJoCo IK utility.
        
        Args:
            root_pose: Pose of the robot's root
            qpos_arms: List of joint positions for each arm
            target_poses: List of target poses for each arm
            
        Returns:
            Joint positions that achieve the target poses
        """
        assert len(qpos_arms) == len(self._arm_sites), "Number of arm joint positions must match number of arm sites."
        assert len(target_poses) == len(self._arm_sites), "Number of target poses must match number of arm sites."
        assert len(qpos_arms) == len(self._shoulders), "Number of arm joint positions must match number of shoulders."

        # Set root pose
        self._physics.bind(self._root_body).pos = root_pose.position
        self._physics.bind(self._root_body).quat = root_pose.orientation.elements

        # Combine initial joint positions
        arm_joints = self._physics.bind(self._arm_joints)
        qpos = np.concatenate(qpos_arms)
        arm_joints.qpos = qpos
        arm_joints.qvel = np.zeros_like(qpos)
        arm_joints.qacc = np.zeros_like(qpos)

        self._physics.forward()

        # Solve for arms
        for idx, (site, target_pose) in enumerate(zip(self._arm_sites, target_poses)):
            if not isinstance(target_pose, Pose):
                raise ValueError("Target pose must be a Pose object.")

            # Get the shoulder associated with the site
            shoulder = self._shoulders[idx]  # Assuming the order matches 
            if shoulder is None:
                raise ValueError(f"No shoulder found for site {site.name}")

            # Solve IK for the current arm
            ik_result = qpos_from_site_pose(
                physics=self._physics,
                site_name=site.name,
                target_pos=target_pose.position,
                target_quat=target_pose.orientation.elements,
                joint_names=[j.name for j in mjcf_utils.safe_find_all(shoulder, "joint")],
                tol=0.000001,
                max_steps=self._config.solver_max_steps,
                inplace=True,
            )

            if not ik_result.success:
                print(f"[WARN] IK did not converge for arm at site {site.name}")

        # Return the updated joint positions
        solution = self._physics.data.qpos[:len(qpos)].copy()
        return solution

    def solve_arms_only(
        self,
        qpos_arm_left: np.ndarray,
        qpos_arm_right: np.ndarray,
        target_pose_left: Pose,
        target_pose_right: Pose,
        forward: bool = False,
        enable_warning: bool = False,
        inplace: bool = True,
    ) -> np.ndarray:
        """
        Solve IK for arm(s) without setting the root pose.
        
        Supports both single-arm and bimanual robots. For single-arm robots,
        only the left arm parameters are used.
        
        Args:
            qpos_arm_left: Joint positions for left arm
            qpos_arm_right: Joint positions for right arm (ignored for single-arm)
            target_pose_left: Target pose for left arm
            target_pose_right: Target pose for right arm (ignored for single-arm)
            forward: Whether to revert changes after solving
            enable_warning: Whether to show warnings
            inplace: Whether to modify the physics state in-place
            
        Returns:
            Joint positions that achieve the target poses
        """
        # Determine if this is a single-arm or bimanual robot
        num_arms = len(self._arm_sites)
        is_single_arm = num_arms == 1
        
        # Convert array targets to Pose objects if needed
        if not isinstance(target_pose_left, Pose):
            if target_pose_left.shape[0] == 6:
                quaternion_left = R.from_euler("xyz", target_pose_left[3:]).as_quat(scalar_first=True)
                target_pose_left = Pose(position=target_pose_left[:3], orientation=Quaternion(quaternion_left))
            elif target_pose_left.shape[0] == 7:
                quaternion_left = Quaternion(target_pose_left[3:])
                target_pose_left = Pose(position=target_pose_left[:3], orientation=quaternion_left)
        
        if not is_single_arm and not isinstance(target_pose_right, Pose):
            if target_pose_right.shape[0] == 6:
                quaternion_right = R.from_euler("xyz", target_pose_right[3:]).as_quat(scalar_first=True)
                target_pose_right = Pose(position=target_pose_right[:3], orientation=Quaternion(quaternion_right))
            elif target_pose_right.shape[0] == 7:
                quaternion_right = Quaternion(target_pose_right[3:])
                target_pose_right = Pose(position=target_pose_right[:3], orientation=quaternion_right)

        # Combine initial joint positions
        arm_joints = self._physics.bind(self._arm_joints)
        if is_single_arm:
            qpos = qpos_arm_left.copy()
        else:
            qpos = np.concatenate((qpos_arm_left, qpos_arm_right))
        
        qpos_backup = qpos.copy()
        arm_joints.qpos = qpos
        arm_joints.qvel = np.zeros_like(qpos)
        arm_joints.qacc = np.zeros_like(qpos)

        self._physics.forward()

        with warnings.catch_warnings():
            if not enable_warning:
                warnings.filterwarnings("ignore")
                
            # Solve for left arm
            ik_left = qpos_from_site_pose(
                physics=self._physics,
                site_name=self._arm_sites[0].name,
                target_pos=target_pose_left.position,
                target_quat=target_pose_left.orientation.elements,
                joint_names=[j.name for j in mjcf_utils.safe_find_all(self._shoulders[0], "joint")],
                tol=0.00001,
                max_steps=self._config.solver_max_steps,
                inplace=inplace,
            )

            # Solve for right arm (only if bimanual)
            if is_single_arm:
                ik_right = None
            else:
                ik_right = qpos_from_site_pose(
                    physics=self._physics,
                    site_name=self._arm_sites[1].name,
                    target_pos=target_pose_right.position,
                    target_quat=target_pose_right.orientation.elements,
                    joint_names=[j.name for j in mjcf_utils.safe_find_all(self._shoulders[1], "joint")],
                    tol=0.00001,
                    max_steps=self._config.solver_max_steps,
                    inplace=inplace,
                )

        # Check convergence
        if is_single_arm:
            if not ik_left.success and enable_warning:
                print("[WARN] IK did not converge for left arm")
        else:
            if not (ik_left.success and ik_right.success) and enable_warning:
                print("[WARN] IK did not converge for both arms")

        # Return the updated joint positions
        solution = self._physics.data.qpos[:len(qpos)].copy()

        # Revert the changes made to the arm joints if requested
        if forward:
            arm_joints.qpos = qpos_backup
            arm_joints.qvel = np.zeros_like(qpos)
            arm_joints.qacc = np.zeros_like(qpos)
            self._physics.forward()
            
        return solution

    def _setup_joints(self):
        """Set up the joints for IK solving."""
        all_joints = mjcf_utils.safe_find_all(self._root_body, "joint")
        
        arm_joints = set()
        for shoulder in self._shoulders:
            arm_joint = mjcf_utils.safe_find_all(shoulder, "joint")[:-1]  # Exclude the last joint
            arm_joints.update(arm_joint)

        # Filter joints
        for joint in all_joints:
            if "wrist" not in joint.name.lower() and \
               (joint not in arm_joints or self._config.end_effector_exclude_word in joint.name.lower()):
                joint.remove()
            else:
                joint.damping = self._config.joint_damping
                if joint.name in self._config.joint_limits:
                    joint.range = self._config.joint_limits[joint.name]

        self._arm_joints = mjcf_utils.safe_find_all(self._root_body, "joint")
    
    def _setup_actuators(self):
        """Set up actuators for IK control."""
        self._origin_site = self._model.worldbody.add("site", name="ee_origin_site")
        
        self._actuators = []

        for site in self._arm_sites:
            actuator = self._generate_ee_actuators(site.name, self._origin_site.name)
            self._actuators.extend(actuator)

        # Disable collisions between arms and torso
        for shoulder in self._shoulders:
            self._model.contact.add("exclude", body1=self._torso.name, body2=shoulder.name)
    
    def _setup_physics(self):
        """Set up the physics simulation."""
        # Enable gravity compensation
        physics_utils.compensate_gravity(self._model)

        # Disable collisions
        for body in mjcf_utils.safe_find_all(self._model, "body"):
            collision_utils.disable_body_collisions(body)

        self._physics = mjcf.Physics.from_mjcf_model(self._model)
        self._physics.model.opt.timestep *= self._config.timestep_factor

        for body in mjcf_utils.safe_find_all(self._model, "body"):
            body = self._physics.bind(body)
            np.maximum(body.inertia, 1e-5)  # Use small non-zero inertia values
    
    def _generate_ee_actuators(self, site: str, origin: str):
        """
        Generate end-effector actuators for the given site.
        
        Args:
            site: Site name for the actuator
            origin: Reference site name
            
        Returns:
            List of actuators for the end effector
        """
        x = self._model.actuator.add(
            "position",
            kp=self._config.kp_position,
            kv=self._config.kv_position,
            ctrlrange=self._config.range_ee_position,
            name=f"{site}_ee_x",
        )
        y = self._model.actuator.add(
            "position",
            kp=self._config.kp_position,
            kv=self._config.kv_position,
            ctrlrange=self._config.range_ee_position,
            name=f"{site}_ee_y",
        )
        z = self._model.actuator.add(
            "position",
            kp=self._config.kp_position,
            kv=self._config.kv_position,
            ctrlrange=self._config.range_ee_position,
            name=f"{site}_ee_z",
        )

        rx = self._model.actuator.add(
            "position",
            kp=self._config.kp_orientation,
            kv=self._config.kv_orientation,
            ctrlrange="-1.57 1.57",
            name=f"{site}_ee_rx",
        )
        ry = self._model.actuator.add(
            "position",
            kp=self._config.kp_orientation,
            kv=self._config.kv_orientation,
            ctrlrange="-1.57 1.57",
            name=f"{site}_ee_ry",
        )
        rz = self._model.actuator.add(
            "position",
            kp=self._config.kp_orientation,
            kv=self._config.kv_orientation,
            ctrlrange="-1.57 1.57",
            name=f"{site}_ee_rz",
        )

        actuators = [x, y, z, rx, ry, rz]
        for index, actuator in enumerate(actuators):
            actuator.gear = np.zeros(6)
            actuator.gear[index] = 1
            actuator.site = site
            actuator.refsite = origin
        return actuators

    def _get_site_quaternion(self, site: mjcf.Element) -> Quaternion:
        """
        Get the quaternion of a site.
        
        Args:
            site: The site element
            
        Returns:
            Quaternion representing the site's orientation
        """
        bound_site = self._physics.bind(site)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, bound_site.xmat)
        return Quaternion(quat)