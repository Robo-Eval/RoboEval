from collections import defaultdict
import time
import numpy as np

from typing import Optional, Union, Iterable, List, Dict, Tuple, Any, Set

import numpy as np
from dm_control import mjcf
from mojo import Mojo
from mojo.elements import Body, MujocoElement, Geom, Camera, Joint, Site
from roboeval.const import HandSide  # LEFT / RIGHT enums
from roboeval.utils.physics_utils import get_colliders   # returns list[Geom]
from roboeval.robots.robot import Robot

_DEFAULT_COLLISION_MARGIN: float = 1e-8


class MetricRolloutEval:
    """
    Generic rollout‑metric recorder.

    1)  self._metric_init(...)      (once in _initialize_env or _on_reset)
    2)  self._metric_step()         (call every _on_step)
    3)  self._metric_finalize(...)  (at episode end, get dict of metrics)
    """

    # ──────────────────────── INITIALISE ────────────────────────────
    def _metric_init(
        self,
        *,
        track_vel_sync: bool = False, # <- track gripper actuator speeds
        track_vertical_sync: bool = False, # <- track gripper wrist height
        track_slippage: bool = False, # <- track gripper slip
        slip_objects: Optional[list] = None,   # <- plain list or single obj,
        slip_sample_window: int = 20,          # <- number of frames to check
        track_collisions: bool = False,  # <- track collisions
        track_cartesian_jerk: bool = True,    # <- track cartesian jerk
        track_joint_jerk: bool = True,        # <- track joint jerk
        track_cartesian_path_length: bool = True,  # <- track cartesian path length
        track_joint_path_length: bool = True,      # <- track joint path length
        track_orientation_path_length: bool = True, # <- track orientation path length
        robot: Robot = None # <- robot instance
    ):
        """
        Initialize tracking for various rollout metrics.
        
        Parameters:
        -----------
        track_vel_sync : bool
            Track gripper actuator speeds synchronization
        track_vertical_sync : bool  
            Track gripper wrist height alignment
        track_slippage : bool
            Track gripper slip events
        slip_objects : Optional[list]
            List of Prop instances to watch for grip loss.
            If you pass a single object, it will be wrapped in a list.
        slip_sample_window : int
            Number of frames to check for slip detection
        track_collisions : bool
            Track collision events with environment and self-collisions
        track_cartesian_jerk : bool
            Track cartesian jerk (rate of change of acceleration) for end-effectors
        track_joint_jerk : bool
            Track joint jerk (rate of change of joint acceleration)
        track_cartesian_path_length : bool
            Track total cartesian path length traveled by end-effectors
        track_joint_path_length : bool
            Track total joint space path length
        track_orientation_path_length : bool
            Track total orientation change path length
        robot : Robot
            Robot instance for accessing kinematics
        """
        self._metrics = defaultdict(float)
        self.start_time = time.time()

        self._track_vel   = track_vel_sync
        self._track_vert  = track_vertical_sync
        self._track_slip  = track_slippage
        
        self._track_cartesian_jerk = track_cartesian_jerk
        self._track_joint_jerk = track_joint_jerk
        self._track_cartesian_path_length = track_cartesian_path_length
        self._track_joint_path_length = track_joint_path_length
        self._track_orientation_path_length = track_orientation_path_length
        
        self._slip_window  = max(1, slip_sample_window)
        
        self._track_collisions = track_collisions

        # normalise slip_objects into a list
        if slip_objects is None:
            self._slip_objs: list = []
        elif isinstance(slip_objects, list):
            self._slip_objs = slip_objects
        else:
            self._slip_objs = [slip_objects]
            
        # ------------ collision bookkeeping ------------------------
        if self._track_collisions and robot is not None:
            self._cache_geoms()                         # fills the ID sets below
            self._env_collision_cnt = 0
            self._self_collision_cnt = 0
            self._past_scene_contacts: Set[int] = set()  # tracked scene geoms
            self._past_robot_contacts: Set[Tuple[int,int]] = set()  # tracked robot geoms

        # ------------ new path and jerk tracking -------------------
        self._robot = robot  # store robot reference for new metrics
        
        # Initialize previous states for jerk and path calculations
        self._prev_cartesian_pos: Dict[HandSide, np.ndarray] = {}
        self._prev_cartesian_vel: Dict[HandSide, np.ndarray] = {}
        self._prev_joint_pos: Dict[HandSide, np.ndarray] = {}
        self._prev_joint_vel: Dict[HandSide, np.ndarray] = {}
        self._prev_orientation: Dict[HandSide, np.ndarray] = {}
        
        # Accumulated path lengths and jerks
        self._cartesian_path_length: Dict[HandSide, float] = {}
        self._joint_path_length: Dict[HandSide, float] = {}
        self._orientation_path_length: Dict[HandSide, float] = {}
        self._cartesian_jerk_sum: Dict[HandSide, float] = {}
        self._cartesian_jerk_squared_sum: Dict[HandSide, float] = {}
        self._joint_jerk_sum: Dict[HandSide, float] = {}
        self._joint_jerk_squared_sum: Dict[HandSide, float] = {}
        self._jerk_sample_count: Dict[HandSide, int] = {}
        
        # Initialize for each gripper/arm
        if robot is not None:
            for side in robot.grippers.keys():
                self._cartesian_path_length[side] = 0.0
                self._joint_path_length[side] = 0.0
                self._orientation_path_length[side] = 0.0
                self._cartesian_jerk_sum[side] = 0.0
                self._cartesian_jerk_squared_sum[side] = 0.0
                self._joint_jerk_sum[side] = 0.0
                self._joint_jerk_squared_sum[side] = 0.0
                self._jerk_sample_count[side] = 0
            
        # accumulators
        self._sum_vel_err = 0.0
        self._n_vel = 0
        self._sum_z_err = 0.0
        self._n_z = 0

        # sampled state for slip logic
        self._prev_state: Dict[Tuple[Any, HandSide], Tuple[bool, float]] = {}
        if self._robot is not None:
            for obj in self._slip_objs:
                for side, grip in self._robot.grippers.items():
                    self._prev_state[(obj, side)] = (
                        grip.is_holding_object(obj),
                        grip.qpos
                    )
        
        self._frame = 0
        self._metrics_slip_count = 0
        self._slip_count_per_object = {
            f"object_{idx + 1}": 0 for idx, o in enumerate(self._slip_objs)
        }
        
        # task-stage flags
        self._stage_flags : Dict[int, bool] = defaultdict(bool)


    # ────────────────────────── PER‑STEP ───────────────────────────
    def _metric_step(self):
        # velocity sync (compare arm actuator speeds)
        if self._track_vel and self._robot is not None and self._robot._multiarm:
            lv = self._arm_speed_norm(self._robot._arms[HandSide.LEFT].qvel)
            rv = self._arm_speed_norm(self._robot._arms[HandSide.RIGHT].qvel)
            self._sum_vel_err += abs(lv - rv)
            self._n_vel += 1

        # vertical alignment of wrists
        if self._track_vert and self._robot is not None and self._robot._multiarm:
            lz = self._robot.grippers[HandSide.LEFT].wrist_position[2]
            rz = self._robot.grippers[HandSide.RIGHT].wrist_position[2]
            self._sum_z_err += abs(lz - rz)
            self._n_z += 1
            
        # --------- collision counting (every step) -------------------
        if self._track_collisions and self._robot is not None and (self._frame % self._slip_window == 0):
            current_scene_contact: Set[int] = set()
            current_robot_contact: Set[Tuple[int,int]] = set()

            contacts = self._mojo.physics.data.contact
            external_env = self._scene_geoms - self._ignore_geoms # define the external environment
            for c in contacts:
                if c.dist > _DEFAULT_COLLISION_MARGIN:
                    continue
                g1, g2 = int(c.geom1), int(c.geom2)
                o1, o2 = self._geom_owner(g1), self._geom_owner(g2)
                
                if o1 == o2 == "robot":   
                    pair = tuple(sorted((g1, g2)))          # unique unordered pair
                    current_robot_contact.add(pair)          # add to current robot contact
                    continue                                # nothing else to check

                elif {"robot", "scene"} == {o1, o2}:
                    scene_gid = g1 if o1 == "scene" else g2
                    if scene_gid in external_env:
                        # add to current scene 
                        current_scene_contact.add(scene_gid)
                    continue                   
            # consider the robot geom collisions as a set of pairs     
            new_robot_contacts = current_robot_contact - self._past_robot_contacts
            self._self_collision_cnt += len(new_robot_contacts)
            self._past_robot_contacts = current_robot_contact
                        
            # update the set of active scene contacts
            new_contacts = current_scene_contact - self._past_scene_contacts
            self._env_collision_cnt += len(new_contacts)
            self._past_scene_contacts = current_scene_contact

        # slip detection every N frames
        if self._track_slip and self._robot is not None and (self._frame % self._slip_window == 0):
            # print(f"slip detection: {self._frame}")
            for obj_idx, obj in enumerate(self._slip_objs):
                key = f"object_{obj_idx + 1}"
                for side, grip in self._robot.grippers.items():
                    idx = (obj, side)
                    prev_hold, prev_qpos = self._prev_state[idx]
                    cur_hold, cur_qpos = grip.is_holding_object(obj), grip.qpos

                    if prev_hold and not cur_hold and (cur_qpos <= prev_qpos):
                        self._metrics_slip_count += 1
                        self._slip_count_per_object[key] += 1

                    self._prev_state[idx] = (cur_hold, cur_qpos)

        # --------- path length and jerk tracking -------------------
        if self._robot is not None:
            for side, gripper in self._robot.grippers.items():
                # Get current states
                current_cartesian_pos = np.array(gripper.wrist_position)
                current_joint_pos = np.array(self._robot._arms[side].qpos)
                
                # Get orientation (quaternion with w, x, y, z attributes)
                orientation_data = gripper.wrist_orientation
                current_orientation = np.array([orientation_data.w, orientation_data.x, orientation_data.y, orientation_data.z])
                
                # Calculate path lengths and velocities when previous data is available
                if side in self._prev_cartesian_pos:
                    # Cartesian path length
                    if self._track_cartesian_path_length:
                        cartesian_displacement = np.linalg.norm(current_cartesian_pos - self._prev_cartesian_pos[side])
                        self._cartesian_path_length[side] += cartesian_displacement
                    
                    # Joint path length  
                    if self._track_joint_path_length:
                        joint_displacement = np.linalg.norm(current_joint_pos - self._prev_joint_pos[side])
                        self._joint_path_length[side] += joint_displacement
                    
                    # Orientation path length (using quaternion angular distance)
                    if self._track_orientation_path_length:
                        # Calculate quaternion angular distance
                        # Angular distance = 2 * arccos(|dot_product|) where dot_product is clamped to [-1, 1]
                        dot_product = np.dot(current_orientation, self._prev_orientation[side])
                        # Clamp to handle numerical precision issues
                        dot_product = np.clip(dot_product, -1.0, 1.0)

                        # Angular distance
                        orientation_diff = 2.0 * np.arccos(np.abs(dot_product))
                        self._orientation_path_length[side] += orientation_diff

                    # Calculate current velocities
                    dt = self._get_timestep()
                    current_cartesian_vel = (current_cartesian_pos - self._prev_cartesian_pos[side]) / dt
                    current_joint_vel = (current_joint_pos - self._prev_joint_pos[side]) / dt
                    
                    # Calculate jerk when previous velocity data is available
                    if side in self._prev_cartesian_vel:
                        jerk_calculated = False
                        
                        # Cartesian jerk
                        if self._track_cartesian_jerk:
                            cartesian_accel = (current_cartesian_vel - self._prev_cartesian_vel[side]) / dt
                            # Initialize previous acceleration storage if needed
                            if not hasattr(self, '_prev_cartesian_accel'):
                                self._prev_cartesian_accel: Dict[HandSide, np.ndarray] = {}
                            
                            if side in self._prev_cartesian_accel:
                                # Calculate jerk vector: derivative of acceleration
                                jerk_vector = (cartesian_accel - self._prev_cartesian_accel[side]) / dt
                                cartesian_jerk = np.linalg.norm(jerk_vector)
                                # Accumulate both regular jerk (for avg) and squared jerk (for RMS)
                                self._cartesian_jerk_sum[side] += cartesian_jerk
                                self._cartesian_jerk_squared_sum[side] += cartesian_jerk ** 2
                                if not jerk_calculated:
                                    self._jerk_sample_count[side] += 1
                                    jerk_calculated = True
                            
                            self._prev_cartesian_accel[side] = cartesian_accel.copy()
                        
                        # Joint jerk
                        if self._track_joint_jerk:
                            joint_accel = (current_joint_vel - self._prev_joint_vel[side]) / dt
                            # Initialize previous acceleration storage if needed
                            if not hasattr(self, '_prev_joint_accel'):
                                self._prev_joint_accel: Dict[HandSide, np.ndarray] = {}
                            
                            if side in self._prev_joint_accel:
                                # Calculate jerk vector: derivative of acceleration
                                jerk_vector = (joint_accel - self._prev_joint_accel[side]) / dt
                                joint_jerk = np.linalg.norm(jerk_vector)
                                # Accumulate both regular jerk (for avg) and squared jerk (for RMS)
                                self._joint_jerk_sum[side] += joint_jerk
                                self._joint_jerk_squared_sum[side] += joint_jerk ** 2
                                if not jerk_calculated:
                                    self._jerk_sample_count[side] += 1
                                    jerk_calculated = True
                            
                            self._prev_joint_accel[side] = joint_accel.copy()
                    
                    # Store current velocities for next iteration
                    self._prev_cartesian_vel[side] = current_cartesian_vel.copy()
                    self._prev_joint_vel[side] = current_joint_vel.copy()
                
                # Store current positions for next iteration
                self._prev_cartesian_pos[side] = current_cartesian_pos.copy()
                self._prev_joint_pos[side] = current_joint_pos.copy()
                # Store orientation as numpy array copy
                if isinstance(current_orientation, np.ndarray):
                    self._prev_orientation[side] = current_orientation.copy()
                else:
                    self._prev_orientation[side] = np.array(current_orientation)

        self._frame += 1
        # print(f'frame_count: {self._frame}')


    # ───────────────────────── FINALISE ────────────────────────────
    def _metric_finalize(
        self,
        success_flag: bool,
        target_distance: Optional[Union[float, Dict[Any, float]]] = None,
        pose_error:    Optional[Union[float, Dict[Any, float]]] = None,
    ):
        m = self._metrics
        m["completion_time"] = time.time() - self.start_time

        # slip results collected during _metric_step
        m["slip_count"]             = self._metrics_slip_count
        m["slip_count_per_object"]  = self._slip_count_per_object
        
        # -------- collisions ----
        if self._track_collisions:
            m["env_collision_count"]  = self._env_collision_cnt
            m["self_collision_count"] = self._self_collision_cnt
        # -------- task stages ---   (True / False flags)
        if self._stage_flags:
            m["task_stage_reached"] = dict(self._stage_flags)
            
        # -------- subtask progress --- (percentage of stages completed)
        if self._stage_flags:
            # Automatically determine total stages from the maximum stage index
            max_stage = max(self._stage_flags.keys())
            completed_stages = sum(1 for success in self._stage_flags.values() if success)
            m["subtask_progress"] = float(completed_stages / max_stage)


        # -------- store distance(s) --------------------------------
        if target_distance is not None:
            if isinstance(target_distance, dict):
                m["target_distance"] = {
                    k: float(v) for k, v in target_distance.items()
                }
            else:                              # single number
                m["target_distance"] = float(target_distance)

        # -------- store pose‑error(s) ------------------------------
        if pose_error is not None:
            if isinstance(pose_error, dict):
                m["object_pose_error"] = {
                    k: float(v) for k, v in pose_error.items()
                }
            else:
                m["object_pose_error"] = float(pose_error)

        m["success"] = float(success_flag)

        # velocity‑sync score   (lower avg error ⇒ higher score)
        if self._track_vel and self._n_vel and self._robot is not None and self._robot._multiarm:
            avg_err = self._sum_vel_err / self._n_vel
            m["bimanual_arm_velocity_difference"] = float(avg_err)

        # vertical‑sync score
        if self._track_vert and self._n_z and self._robot is not None and self._robot._multiarm:
            avg_z = self._sum_z_err / self._n_z
            m["bimanual_gripper_vertical_difference"] = float(avg_z)

        # -------- new path and jerk metrics ------------------------
        if self._robot is not None:
            # Path length metrics
            if self._track_cartesian_path_length:
                if self._robot._multiarm:
                    m["cartesian_path_length"] = {
                        side.name.lower(): float(length) 
                        for side, length in self._cartesian_path_length.items()
                    }
                    # Total and average for bimanual
                    total_cartesian = sum(self._cartesian_path_length.values())
                    m["total_cartesian_path_length"] = float(total_cartesian)
                    m["avg_cartesian_path_length"] = float(total_cartesian / len(self._cartesian_path_length))
                else:
                    # Single arm case
                    side = list(self._cartesian_path_length.keys())[0]
                    m["cartesian_path_length"] = float(self._cartesian_path_length[side])
            
            if self._track_joint_path_length:
                if self._robot._multiarm:
                    m["joint_path_length"] = {
                        side.name.lower(): float(length) 
                        for side, length in self._joint_path_length.items()
                    }
                    # Total and average for bimanual
                    total_joint = sum(self._joint_path_length.values())
                    m["total_joint_path_length"] = float(total_joint)
                    m["avg_joint_path_length"] = float(total_joint / len(self._joint_path_length))
                else:
                    # Single arm case
                    side = list(self._joint_path_length.keys())[0]
                    m["joint_path_length"] = float(self._joint_path_length[side])
            
            if self._track_orientation_path_length:
                if self._robot._multiarm:
                    m["orientation_path_length"] = {
                        side.name.lower(): float(length) 
                        for side, length in self._orientation_path_length.items()
                    }
                    # Total and average for bimanual
                    total_orientation = sum(self._orientation_path_length.values())
                    m["total_orientation_path_length"] = float(total_orientation)
                    m["avg_orientation_path_length"] = float(total_orientation / len(self._orientation_path_length))
                else:
                    # Single arm case
                    side = list(self._orientation_path_length.keys())[0]
                    m["orientation_path_length"] = float(self._orientation_path_length[side])
            
            # Jerk metrics (both average and RMS)
            if self._track_cartesian_jerk:
                if self._robot._multiarm:
                    # Average jerk
                    m["avg_cartesian_jerk"] = {
                        side.name.lower(): float(jerk_sum / max(1, self._jerk_sample_count[side])) 
                        for side, jerk_sum in self._cartesian_jerk_sum.items()
                    }
                    # RMS jerk
                    m["rms_cartesian_jerk"] = {
                        side.name.lower(): float(np.sqrt(jerk_squared_sum / max(1, self._jerk_sample_count[side]))) 
                        for side, jerk_squared_sum in self._cartesian_jerk_squared_sum.items()
                    }
                    # Overall average for bimanual
                    total_samples = sum(self._jerk_sample_count.values())
                    if total_samples > 0:
                        total_jerk = sum(self._cartesian_jerk_sum.values())
                        m["overall_avg_cartesian_jerk"] = float(total_jerk / total_samples)
                        total_jerk_squared = sum(self._cartesian_jerk_squared_sum.values())
                        m["overall_rms_cartesian_jerk"] = float(np.sqrt(total_jerk_squared / total_samples))
                else:
                    # Single arm case
                    side = list(self._cartesian_jerk_sum.keys())[0]
                    samples = max(1, self._jerk_sample_count[side])
                    m["avg_cartesian_jerk"] = float(self._cartesian_jerk_sum[side] / samples)
                    m["rms_cartesian_jerk"] = float(np.sqrt(self._cartesian_jerk_squared_sum[side] / samples))
            
            if self._track_joint_jerk:
                if self._robot._multiarm:
                    # Average jerk
                    m["avg_joint_jerk"] = {
                        side.name.lower(): float(jerk_sum / max(1, self._jerk_sample_count[side])) 
                        for side, jerk_sum in self._joint_jerk_sum.items()
                    }
                    # RMS jerk
                    m["rms_joint_jerk"] = {
                        side.name.lower(): float(np.sqrt(jerk_squared_sum / max(1, self._jerk_sample_count[side]))) 
                        for side, jerk_squared_sum in self._joint_jerk_squared_sum.items()
                    }
                    # Overall average for bimanual
                    total_samples = sum(self._jerk_sample_count.values())
                    if total_samples > 0:
                        total_jerk = sum(self._joint_jerk_sum.values())
                        m["overall_avg_joint_jerk"] = float(total_jerk / total_samples)
                        total_jerk_squared = sum(self._joint_jerk_squared_sum.values())
                        m["overall_rms_joint_jerk"] = float(np.sqrt(total_jerk_squared / total_samples))
                else:
                    # Single arm case
                    side = list(self._joint_jerk_sum.keys())[0]
                    samples = max(1, self._jerk_sample_count[side])
                    m["avg_joint_jerk"] = float(self._joint_jerk_sum[side] / samples)
                    m["rms_joint_jerk"] = float(np.sqrt(self._joint_jerk_squared_sum[side] / samples))

        return dict(m)

    # ─────────────────────── collision helpers ───────────────────────
    def _cache_geoms(self) -> None:
        """Build three sets: robot geoms, scene geoms, ignored (slip) geoms."""
        self._robot_geoms:  Set[int] = set()
        self._scene_geoms:  Set[int] = set()
        self._ignore_geoms: Set[int] = set()

        # all robot bodies (pelvis + every arm)
        def collect(body: Body):
            for g in body.geoms:
                gid = int(self._mojo.physics.bind(g.mjcf).element_id)
                self._robot_geoms.add(gid)

        collect(self._robot.pelvis)
        for arm in self._robot._arms.values():
            collect(arm.body)

        # slip-object geoms we want to ignore in env collision count
        for obj in self._slip_objs:
            for g in get_colliders(obj):
                gid = int(self._mojo.physics.bind(g.mjcf).element_id)
                self._ignore_geoms.add(gid)

        # everything not robot gets added to scene geoms
        for gid in range(self._mojo.physics.model.ngeom):
            if gid not in self._robot_geoms:
                self._scene_geoms.add(gid)
                
        # print(f"robot geoms: {self._robot_geoms}")
        # print(f"scene geoms: {self._scene_geoms}")
        # print(f"target geoms: {self._ignore_geoms}")

    def _geom_owner(self, geom_id: int) -> str:
        if geom_id in self._robot_geoms:
            return "robot"
        if geom_id in self._scene_geoms:
            return "scene"
        return "other"
    
    def _metric_stage(self, stage_idx: int, success: Optional[bool] = None):
        """
        Call this from the *environment* whenever a new sub-goal is achieved.
        e.g  self._metric_stage(1) after the first mug is grasped.
        """
        self._stage_flags[stage_idx] = success if success is not None else True
        
    def get_metric_stage(self, stage_idx: int=1) -> bool:
        """Get the stage status  for each progression"""
        return bool(self._stage_flags.get(stage_idx, False))
    
    @staticmethod
    def _arm_speed_norm(arm_qvel) -> float:
        """Scalar joint‑space speed (L2‑norm of qvel)."""
        return float(np.linalg.norm(arm_qvel, ord=2))
    
    def _get_timestep(self) -> float:
        """Get environment control timestep"""
        # Calculate timestep from control frequency
        # env.control_frequency is in Hz, so timestep = 1 / frequency
        if hasattr(self, 'control_frequency'):
            return 1.0 / float(self.control_frequency)
        elif hasattr(self, '_control_frequency'):
            return 1.0 / float(self._control_frequency)
        # Default to 50Hz control frequency (0.02s timestep)
        return 0.02
    
    @staticmethod
    def _obj_key(obj):
        """Return a stable dict key for an object."""
        return getattr(obj, "name", str(id(obj)))