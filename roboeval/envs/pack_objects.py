from abc import ABC

import numpy as np
from pyquaternion import Quaternion

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH, HandSide
from roboeval.envs.props.package import Package
from roboeval.envs.props.cabintets import BaseCabinet
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.utils.physics_utils import distance
from roboeval.utils.metric_rollout import MetricRolloutEval


class PackBox(RoboEvalEnv, ABC, MetricRolloutEval):
    """
    Base pack box environment (static)
    Success: Robot must use effectors to close box lid
    """
    _PRESET_PATH = PRESETS_PATH / "packaging_box.yaml"

    _BOX_POS = np.array([0.6, 0.3, 1.05])
    _BOX_POS_EXTENTS = np.array([0.1, 0.14])
    _BOX_POS_BOUNDS = np.array([0.001, 0.001, 0])
    _BOX_ROT_BOUNDS = np.deg2rad(30)
    _BOX_ROT = np.deg2rad(90)
    _YCB_STEP = 0.02

    _TOLERANCE = .1
    
    _success_check = True
    _final_metrics = {}

    def _initialize_env(self):
        self.packing_box = Package(self._mojo)
        self.table = self._preset.get_props(BaseCabinet)[0]
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.packing_box,
            track_collisions=True,
            robot=self.robot,         
        )

    def _on_step(self):
        self._metric_step()

    def _on_reset(self):
        self.packing_box.set_state(state=0)
        self.packing_box.set_state(target_state={
            "packaging_box/joint_0": 1,
            "packaging_box/joint_1": 1,
        })

        self.packing_box.body.set_position(self._BOX_POS)
        self.packing_box.body.set_quaternion(Quaternion(axis=[0, 0, 1], angle=self._BOX_ROT).elements)
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.packing_box,
            track_collisions=True,
            robot=self.robot,         
        )
        
        for idx in range(1, 6): self._metric_stage(idx , False)

    def _success(self) -> bool:
        self.success_check = True
        
        # Close all lid joints of the box 
        if not np.allclose(self.packing_box.get_state(), 0, atol=self._TOLERANCE):
            self.success_check = False
            
        # —————————— distance check ——————————————————
        right_gripper_right_flap = distance(self.robot.grippers[HandSide.RIGHT].body, self.packing_box.flap_1)
        left_gripper_left_flap = distance(self.robot.grippers[HandSide.LEFT].body, self.packing_box.flap_2)

        
        # ——————————— stage monitoring ———————————————
        grasping_right_flap = self.robot.is_gripper_holding_object(self.packing_box.flap_1, HandSide.RIGHT)
        grasping_left_flap = self.robot.is_gripper_holding_object(self.packing_box.flap_2, HandSide.LEFT)
        close_box = np.allclose(self.packing_box.get_state(), 0, atol=self._TOLERANCE)
        box_state = self.packing_box.get_state()
        close_right_flap = abs(box_state[0]) < self._TOLERANCE
        close_left_flap = abs(box_state[1]) < self._TOLERANCE

        if grasping_left_flap: self._metric_stage(1) 
        if grasping_right_flap: self._metric_stage(2)
        if close_right_flap: self._metric_stage(3)
        if close_left_flap: self._metric_stage(4)
        if close_box: self._metric_stage(5) # both flaps are closed 
        
        self._final_metrics = self._metric_finalize(
            success_flag=self.success_check,
            target_distance={
                "right gripper-right flap distance": right_gripper_right_flap,
                "left gripper-left flap distance": left_gripper_left_flap
            }
            )

        return self.success_check
    
    def _get_task_info(self):
        """Expose metrics every step (optional) or only at episode end."""
        return getattr(self, "_final_metrics", {})      
    
class PackBoxPositionAndOrientation(PackBox):
    """Pack box with position and orientation randomization"""

    _BOX_POS = np.array([0.7, 0.25, 1.05])

    def _on_reset(self):
        self.packing_box.set_state(state=0)
        self.packing_box.set_state(target_state={
            "packaging_box/joint_0": 1,
            "packaging_box/joint_1": 1,
        })

        points = get_random_points_on_plane(
            1,
            self._BOX_POS,
            self._BOX_POS_EXTENTS,
            self._YCB_STEP,
            self._BOX_POS_BOUNDS,
        )

        point = points[0]
        self.packing_box.body.set_position(point)
        angle = np.random.uniform(-self._BOX_ROT_BOUNDS, self._BOX_ROT_BOUNDS)
        self.packing_box.body.set_quaternion(
            Quaternion(axis=[0, 0, 1], angle=self._BOX_ROT + angle).elements
        )

        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.packing_box,
            track_collisions=True,
            robot=self.robot,         
        )
        
        for idx in range(1, 6): self._metric_stage(idx , False)

class PackBoxOrientation(PackBox):
    """Pack box with orientation randomization"""
    _PRESET_PATH = PRESETS_PATH / "packaging_box_rotation.yaml"

    _BOX_POS = np.array([0.72, 0, 1.05])

    def _on_reset(self):
        self.packing_box.set_state(state=0)
        self.packing_box.set_state(target_state={
            "packaging_box/joint_0": 1,
            "packaging_box/joint_1": 1,
        })

        self.packing_box.body.set_position(self._BOX_POS)
        angle = np.random.uniform(-self._BOX_ROT_BOUNDS, self._BOX_ROT_BOUNDS)
        self.packing_box.body.set_quaternion(
            Quaternion(axis=[0, 0, 1], angle=self._BOX_ROT + angle).elements
        )

        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.packing_box,
            track_collisions=True,
            robot=self.robot,         
        )
        
        for idx in range(1, 6): self._metric_stage(idx , False)
        

class PackBoxPosition(PackBox):
    """Pack box with position randomization"""
    _BOX_POS = np.array([0.7, 0.3, 1.05])
    
    def _on_reset(self):
        self.packing_box.set_state(state=0)
        self.packing_box.set_state(target_state={
            "packaging_box/joint_0": 1,
            "packaging_box/joint_1": 1,
        })

        points = get_random_points_on_plane(
            1,
            self._BOX_POS,
            self._BOX_POS_EXTENTS,
            self._YCB_STEP,
            self._BOX_POS_BOUNDS,
        )

        point = points[0]
        self.packing_box.body.set_position(point)
        self.packing_box.body.set_quaternion(Quaternion(axis=[0, 0, 1], angle=self._BOX_ROT).elements)
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.packing_box,
            track_collisions=True,
            robot=self.robot,         
        )
        
        for idx in range(1, 6): self._metric_stage(idx , False)