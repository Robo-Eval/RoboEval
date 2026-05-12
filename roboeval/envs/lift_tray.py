from abc import ABC

import numpy as np
from pyquaternion import Quaternion

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH, HandSide
from roboeval.envs.props.items import BreakfastTray, LighterBreakfastTray
from roboeval.envs.props.cabintets import BaseCabinet
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.utils.physics_utils import distance
from roboeval.utils.metric_rollout import MetricRolloutEval

# Vertical offset from the tray's body origin to the bottom of the tray. This
# is the half-thickness of the breakfast-tray collider and is used so that
# ``init_z_pos`` represents the z-coordinate of the table-top surface that the
# tray rests on (so ``tray_lift_distance = body_z - init_z_pos`` starts at the
# half-thickness and grows by however far the tray is lifted off the table).
_TRAY_HALF_HEIGHT = 0.02694

# Steps of free physics integration used after teleporting the tray, so it
# freefalls onto the table and the recorded resting height reflects the
# actual on-table position rather than the spawn height.
_TRAY_SETTLE_STEPS = 60


class _TrayEnv(RoboEvalEnv, ABC, MetricRolloutEval):
    """Base env for lifting and moving tray"""
    _PRESET_PATH = PRESETS_PATH / "lift_tray.yaml"
    # Concrete tray prop type used by this env's preset. ``Preset.get_props``
    # matches by exact type (``BreakfastTray`` and ``LighterBreakfastTray`` are
    # sibling classes), so subclasses whose preset uses a different tray must
    # override this.
    _TRAY_CLS: type = BreakfastTray
    _success_check = True
    _final_metrics = {}


    def _initialize_env(self):
        self.breakfast_tray = self._preset.get_props(self._TRAY_CLS)[0]
        self.table = self._preset.get_props(BaseCabinet)[0]
        self.target_table = self._preset.get_props(BaseCabinet)[-1]

    # ---- helpers shared by every tray variant ----
    def _zero_tray_velocity(self):
        """Zero linear/angular velocities of the (kinematic) tray body.

        Teleporting a free body with ``set_position`` doesn't reset its
        ``qvel``; without this the tray can carry over momentum from the
        previous episode and bounce on the table during settling.
        """
        if self.breakfast_tray.kinematic:
            freejoint = self._mojo.physics.bind(
                self.breakfast_tray.body.mjcf.freejoint
            )
            freejoint.qvel = np.zeros_like(freejoint.qvel)

    def _settle_tray(self, steps: int = _TRAY_SETTLE_STEPS):
        """Step physics so the tray can freefall onto the table.

        The robot is left at its reset actuator targets, so it remains
        stationary while gravity pulls the tray down. Returns once the tray
        is roughly static or ``steps`` physics steps have elapsed.
        """
        self._zero_tray_velocity()
        for _ in range(steps):
            self._mojo.step()
            if self.breakfast_tray.is_static():
                break
    

# ----------------------------------- Lift Tray Tasks -----------------------------------
class LiftTray(_TrayEnv):
    """
    Lift the tray with both effectors (static)
    """

    _TOLERANCE = .1
    _OBJ_ROT = np.array([0.5 ,0.5 ,0.5 ,0.5])
    _OBJ_STEP = 0.15
    _OBJ_POS_EXTENTS = np.array([0.1, 0.25])
    _OBJ_POS_BOUNDS = np.array([0.005, 0.005, 0])
    _OBJ_ROT_BOUNDS = np.deg2rad(30)
    _SUCCESSFUL_DIST = 0.1

    _OBJ_POS = np.array([0.65, -0.325, 1.0])
    _OBJ_POS_EXTENTS = np.array([0.1, 0.25])  

    

    def _initialize_env(self):
        super()._initialize_env()
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.breakfast_tray,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True        
        )

    def _sync_lift_tray_metrics_after_settle(self):
        """Snapshot resting height and re-init rollout metrics (shared by subclasses)."""
        self.init_z_pos = (
            self.breakfast_tray.body.get_position()[2] - _TRAY_HALF_HEIGHT
        )
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.breakfast_tray,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )
        for idx in range(1, 5):
            self._metric_stage(idx, False)

    def _on_reset(self):
        # Tray pose: same as before this file's settle work — only canonical quat here.
        self.breakfast_tray.body.set_quaternion(self._OBJ_ROT)
        self._settle_tray()
        self._sync_lift_tray_metrics_after_settle()

    def _on_step(self):
        self._metric_step()

    def _success(self) -> bool:
        self._success_check = True
        
        for side in self.robot.grippers:
            if not self.robot.is_gripper_holding_object(self.breakfast_tray, side):
                self._success_check = False
            
        if self.breakfast_tray.is_colliding(self.table):
            self._success_check = False
        
        # –––––––––––––– Distance Checking –––––––––––––––––
        tray_table_distance = distance(self.breakfast_tray.body, self.table.body)
        tray_lift_distance = self.breakfast_tray.body.get_position()[2] - self.init_z_pos

        if tray_lift_distance < 0.05:
            self._success_check = False
        right_gripper_tray_dist = distance(self.breakfast_tray.body, self.robot.grippers[HandSide.RIGHT].body)
        left_gripper_tray_dist = distance(self.breakfast_tray.body, self.robot.grippers[HandSide.LEFT].body)

        # –––––––––––––– Stage Monitoring ––––––––––––––––––
        colliding = self.breakfast_tray.is_colliding(self.table)
        floor_collision = self.breakfast_tray.is_colliding(self._floor)
        holding_obj = all(self.robot.is_gripper_holding_object(self.breakfast_tray, side) for side in self.robot.grippers)
        grasping_left = self.robot.is_gripper_holding_object(self.breakfast_tray, HandSide.LEFT) 
        grasping_right = self.robot.is_gripper_holding_object(self.breakfast_tray, HandSide.RIGHT) 
        
        if grasping_left: self._metric_stage(1) # split into left and right grasp check
        if grasping_right: self._metric_stage(2)
        if holding_obj and not colliding and not floor_collision: self._metric_stage(3) # Check if tray is not colliding with the table or floor
        if tray_lift_distance >= 0.05: self._metric_stage(4) # tray lifted at least 0.05 above the table

        
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "lift_distance": tray_lift_distance,
                "tray-table distance": tray_table_distance,
                "right gripper-tray distance":right_gripper_tray_dist,
                "left gripper-tray distance": left_gripper_tray_dist
            }
        )
        return self._success_check

    def _fail(self) -> bool:
        return super()._fail()
    
    def _get_task_info(self):
        """Expose metrics every step (optional) or only at episode end."""
        return getattr(self, "_final_metrics", {})
    
class LiftTrayPosition(LiftTray):
    """
    Lift the tray with both effectors and randomization on tray's position
    """

    def _on_reset(self):
        super()._on_reset()
        points = get_random_points_on_plane(
            1,
            self._OBJ_POS,
            self._OBJ_POS_EXTENTS,
            self._OBJ_STEP,
            self._OBJ_POS_BOUNDS,
        )
        self.breakfast_tray.body.set_position(points[0])
        self._settle_tray()
        self._sync_lift_tray_metrics_after_settle()


class LiftTrayOrientation(LiftTray):
    """
    Lift the tray with both effectors and randomization on tray's orientation
    """

    _OBJ_ROT_BOUNDS = np.deg2rad(20)

    _OBJ_POS = np.array([0.65, -0.325, 0.9])

    _success_check = True
    _final_metrics = {}

    def _on_reset(self):
        super()._on_reset()
        angle = np.random.uniform(-self._OBJ_ROT_BOUNDS, self._OBJ_ROT_BOUNDS)
        new_rot = Quaternion(axis=[0, 0, 1], angle=angle) * Quaternion(self._OBJ_ROT)
        self.breakfast_tray.body.set_quaternion(new_rot.elements)
        self._settle_tray()
        self._sync_lift_tray_metrics_after_settle()


class LiftTrayPositionAndOrientation(LiftTray):
    """
    Lift the tray with both effectors and randomization of position AND orientation
    """

    def _on_reset(self):
        super()._on_reset()
        points = get_random_points_on_plane(
            1,
            self._OBJ_POS,
            self._OBJ_POS_EXTENTS,
            self._OBJ_STEP,
            self._OBJ_POS_BOUNDS,
        )
        self.breakfast_tray.body.set_position(points[0])
        angle = np.random.uniform(-self._OBJ_ROT_BOUNDS, self._OBJ_ROT_BOUNDS)
        new_rot = Quaternion(axis=[0, 0, 1], angle=angle) * Quaternion(self._OBJ_ROT)
        self.breakfast_tray.body.set_quaternion(new_rot.elements)
        self._settle_tray()
        self._sync_lift_tray_metrics_after_settle()


class DragOverAndLiftTray(_TrayEnv):
    """
    Tray begins on adjacent table
    Drag tray to target table and raise above target table
    """

    _PRESET_PATH = PRESETS_PATH / "drag_over_and_lift_tray.yaml"
    _TRAY_CLS = LighterBreakfastTray
    _OBJ_POS = np.array([0.65, 0.325, 1.0])


    def _initialize_env(self):
        super()._initialize_env()
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.breakfast_tray,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

    def _on_step(self):
        self._metric_step()

    def _on_reset(self):
        self.breakfast_tray.body.set_position(self._OBJ_POS)

        # Let the tray drop onto the source table so ``_init_z_pos`` reflects
        # the actual resting height rather than the spawn height in mid-air.
        self._settle_tray()

        self._init_z_pos = (
            self.breakfast_tray.body.get_position()[2] - _TRAY_HALF_HEIGHT
        )

        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.breakfast_tray,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )
        for idx in range(1, 5):
            self._metric_stage(idx, False)

    def _success(self) -> bool:
        self._success_check = True

        # –––––––––– success check ––––––––––––––––––
        for side in self.robot.grippers:
            if not self.robot.is_gripper_holding_object(self.breakfast_tray, side):
                self._success_check = False
        if self.breakfast_tray.is_colliding(self.table):
            self._success_check = False
        if self.breakfast_tray.is_colliding(self.target_table):
            self._success_check = False
        if self.breakfast_tray.body.get_position()[1] > 0:
            self._success_check = False

        # ––––––––––– distance check –––––––––––––––––
        tray_table_distance = distance(self.breakfast_tray.body, self.table.body)
        tray_lift_distance = self.breakfast_tray.body.get_position()[2] - self._init_z_pos
        right_gripper_tray_dist = distance(self.breakfast_tray.body, self.robot.grippers[HandSide.RIGHT].body)
        left_gripper_tray_dist = distance(self.breakfast_tray.body, self.robot.grippers[HandSide.LEFT].body)

         # –––––––––––––– Stage Monitoring ––––––––––––––––––
        colliding = self.breakfast_tray.is_colliding(self.table) or self.breakfast_tray.is_colliding(self.target_table)
        floor_collision = self.breakfast_tray.is_colliding(self._floor)
        holding_obj = all(self.robot.is_gripper_holding_object(self.breakfast_tray, side) for side in self.robot.grippers)
        # Stage 3 fires once the tray reaches the *target* table; previously
        # this checked ``self.table.hob`` (the source table the tray starts
        # on), so the stage latched immediately at t=0.
        target_collision = self.breakfast_tray.is_colliding(self.target_table.hob)
        grasping_left = self.robot.is_gripper_holding_object(self.breakfast_tray, HandSide.LEFT)
        grasping_right = self.robot.is_gripper_holding_object(self.breakfast_tray, HandSide.RIGHT)
        above_table = self.breakfast_tray.body.get_position()[1] <= 0

        if grasping_left: self._metric_stage(1) # left effector grasp tray
        if grasping_right: self._metric_stage(2) # right effector grasp tray
        if target_collision: self._metric_stage(3) # tray reaches target table
        if holding_obj and not colliding and not floor_collision and above_table: self._metric_stage(4) # Check if tray is not colliding with the table or floor WHILE holding the object and hovering over target table

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "lift_distance": tray_lift_distance,
                "tray-table distance": tray_table_distance,
                "right gripper-tray distance":right_gripper_tray_dist,
                "left gripper-tray distance": left_gripper_tray_dist
            }
        )
        return self._success_check

    def _get_task_info(self):
        """Expose metrics every step (optional) or only at episode end."""
        return getattr(self, "_final_metrics", {})