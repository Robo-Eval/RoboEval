"""
MuJoCo adapter for the simulator-agnostic MetricEvaluator.

``MetricRolloutEval`` remains the mixin used by every task environment.
Its public interface (``_metric_init``, ``_metric_step``, ``_metric_stage``,
``get_metric_stage``, ``_metric_finalize``) is **unchanged** — zero edits
required in any task file.

Internally it now packs MuJoCo / Robot data into ``StepData`` dataclasses
and delegates all computation to a ``MetricEvaluator`` instance.
"""

from collections import defaultdict
import time
from typing import Any, Dict, Optional, Set, Tuple, Union

import numpy as np
from dm_control import mjcf
from mojo import Mojo
from mojo.elements import Body, Geom

from roboeval.const import HandSide
from roboeval.utils.physics_utils import get_colliders
from roboeval.robots.robot import Robot

from roboeval.utils.metric_evaluator import (
    ArmStepData,
    CollisionContext,
    ContactPair,
    MetricConfig,
    MetricEvaluator,
    StepData,
)


class MetricRolloutEval:
    """
    Generic rollout-metric recorder.

    1)  self._metric_init(...)      (once in _initialize_env or _on_reset)
    2)  self._metric_step()         (call every _on_step)
    3)  self._metric_finalize(...)  (at episode end, get dict of metrics)
    """

    # ──────────────────────── INITIALISE ────────────────────────────
    def _metric_init(
        self,
        *,
        track_vel_sync: bool = False,
        track_vertical_sync: bool = False,
        track_slippage: bool = False,
        slip_objects: Optional[list] = None,
        slip_sample_window: int = 20,
        track_collisions: bool = False,
        track_cartesian_jerk: bool = True,
        track_joint_jerk: bool = True,
        track_cartesian_path_length: bool = True,
        track_joint_path_length: bool = True,
        track_orientation_path_length: bool = True,
        robot: Robot = None,
    ):
        """
        Initialize tracking for various rollout metrics.

        Parameters
        ----------
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
        self._robot = robot

        # Normalise slip_objects into a list
        if slip_objects is None:
            self._slip_objs: list = []
        elif isinstance(slip_objects, list):
            self._slip_objs = slip_objects
        else:
            self._slip_objs = [slip_objects]

        # Build stable string keys for each slip object
        self._slip_obj_keys = [self._obj_key(o) for o in self._slip_objs]

        # Determine arm IDs from robot
        if robot is not None:
            arm_ids = [side.name.lower() for side in robot.grippers.keys()]
            is_bimanual = robot._multiarm
        else:
            arm_ids = []
            is_bimanual = False

        # Store the side→arm_id mapping for extraction
        self._side_to_id: Dict[HandSide, str] = {}
        if robot is not None:
            for side in robot.grippers.keys():
                self._side_to_id[side] = side.name.lower()

        # Build config
        config = MetricConfig(
            track_vel_sync=track_vel_sync,
            track_vertical_sync=track_vertical_sync,
            track_slippage=track_slippage,
            slip_object_ids=self._slip_obj_keys,
            slip_sample_window=slip_sample_window,
            track_collisions=track_collisions,
            track_cartesian_jerk=track_cartesian_jerk,
            track_joint_jerk=track_joint_jerk,
            track_cartesian_path_length=track_cartesian_path_length,
            track_joint_path_length=track_joint_path_length,
            track_orientation_path_length=track_orientation_path_length,
            is_bimanual=is_bimanual,
            arm_ids=arm_ids,
        )

        self._evaluator = MetricEvaluator(config)

        # Build collision context if needed
        collision_ctx = None
        if track_collisions and robot is not None:
            collision_ctx = self._build_collision_context()

        # Build initial slip state from the robot
        initial_slip: Dict[Tuple[str, str], Tuple[bool, float]] = {}
        if robot is not None:
            for obj, obj_key in zip(self._slip_objs, self._slip_obj_keys):
                for side, grip in robot.grippers.items():
                    arm_id = self._side_to_id[side]
                    initial_slip[(obj_key, arm_id)] = (
                        grip.is_holding_object(obj),
                        grip.qpos,
                    )

        self._evaluator.init(
            collision_context=collision_ctx,
            initial_slip_state=initial_slip if initial_slip else None,
        )

    # ────────────────────────── PER-STEP ───────────────────────────
    def _metric_step(self):
        if self._robot is None:
            self._evaluator.step(StepData(arms={}, timestep=self._get_timestep()))
            return

        arms: Dict[str, ArmStepData] = {}
        for side, gripper in self._robot.grippers.items():
            arm_id = self._side_to_id[side]
            arm = self._robot._arms[side]

            orientation_data = gripper.wrist_orientation
            ori = np.array(
                [orientation_data.w, orientation_data.x,
                 orientation_data.y, orientation_data.z]
            )

            holding = {}
            for obj, obj_key in zip(self._slip_objs, self._slip_obj_keys):
                holding[obj_key] = gripper.is_holding_object(obj)

            arms[arm_id] = ArmStepData(
                arm_id=arm_id,
                joint_positions=np.array(arm.qpos),
                joint_velocities=np.array(arm.qvel),
                wrist_position=np.array(gripper.wrist_position),
                wrist_orientation=ori,
                gripper_opening=gripper.qpos,
                holding_object=holding,
            )

        # Extract contacts if tracking collisions
        contacts = []
        if self._evaluator._cfg.track_collisions:
            for c in self._mojo.physics.data.contact:
                contacts.append(
                    ContactPair(
                        geom1=int(c.geom1),
                        geom2=int(c.geom2),
                        distance=float(c.dist),
                    )
                )

        data = StepData(
            arms=arms,
            contacts=contacts,
            timestep=self._get_timestep(),
        )
        self._evaluator.step(data)

    # ───────────────────────── FINALISE ────────────────────────────
    def _metric_finalize(
        self,
        success_flag: bool,
        target_distance: Optional[Union[float, Dict[Any, float]]] = None,
        pose_error: Optional[Union[float, Dict[Any, float]]] = None,
    ):
        return self._evaluator.finalize(
            success_flag=success_flag,
            target_distance=target_distance,
            pose_error=pose_error,
        )

    # ─────────────────────── stage helpers ───────────────────────
    def _metric_stage(self, stage_idx: int, success: Optional[bool] = None):
        """
        Call this from the *environment* whenever a new sub-goal is achieved.
        e.g  self._metric_stage(1) after the first mug is grasped.
        """
        self._evaluator.mark_stage(stage_idx, success)

    def get_metric_stage(self, stage_idx: int = 1) -> bool:
        """Get the stage status for each progression"""
        return self._evaluator.get_stage(stage_idx)

    # ─────────────────────── collision helpers ───────────────────────
    def _build_collision_context(self) -> CollisionContext:
        """Build CollisionContext from MuJoCo geom data."""
        robot_geoms: Set[int] = set()
        scene_geoms: Set[int] = set()
        ignore_geoms: Set[int] = set()

        def collect(body: Body):
            for g in body.geoms:
                gid = int(self._mojo.physics.bind(g.mjcf).element_id)
                robot_geoms.add(gid)

        collect(self._robot.pelvis)
        for arm in self._robot._arms.values():
            collect(arm.body)

        for obj in self._slip_objs:
            for g in get_colliders(obj):
                gid = int(self._mojo.physics.bind(g.mjcf).element_id)
                ignore_geoms.add(gid)

        for gid in range(self._mojo.physics.model.ngeom):
            if gid not in robot_geoms:
                scene_geoms.add(gid)

        return CollisionContext(
            robot_geom_ids=robot_geoms,
            scene_geom_ids=scene_geoms,
            ignore_geom_ids=ignore_geoms,
        )

    # ─────────────────────── utility ─────────────────────────────
    def _get_timestep(self) -> float:
        """Get environment control timestep"""
        if hasattr(self, "control_frequency"):
            return 1.0 / float(self.control_frequency)
        elif hasattr(self, "_control_frequency"):
            return 1.0 / float(self._control_frequency)
        return 0.02

    @staticmethod
    def _obj_key(obj):
        """Return a stable dict key for an object."""
        return getattr(obj, "name", str(id(obj)))

    @staticmethod
    def _arm_speed_norm(arm_qvel) -> float:
        """Scalar joint-space speed (L2-norm of qvel)."""
        return float(np.linalg.norm(arm_qvel, ord=2))
