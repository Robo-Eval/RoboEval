from abc import ABC

import numpy as np
from pyquaternion import Quaternion

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH
from roboeval.envs.props.cabintets import BaseCabinet
from roboeval.envs.props.partnet import KitchenPot
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance
from roboeval.const import HandSide

INITIAL_HEIGHT = 0.996
SUCCESS_ROT = np.deg2rad(20)
SUCCESS_HEIGHT = INITIAL_HEIGHT + 0.1


class LiftPot(RoboEvalEnv, ABC, MetricRolloutEval):
    """Lift kitchenpot base environment (static)"""

    _POT_ROT_BOUNDS = np.deg2rad(30)

    _PRESET_PATH = PRESETS_PATH / "lift_pot.yaml"
    
    _success_check = True
    _final_metrics = {}

    def _initialize_env(self):
        self.cabinet_1 = self._preset.get_props(BaseCabinet)[0]
        self.cabinet_2 = self._preset.get_props(BaseCabinet)[1]
        self.kitchenpot = KitchenPot(self._mojo)
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=[self.kitchenpot],
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True
        )

    def _on_reset(self):
        self.kitchenpot.set_pose(position=np.array([0.7, 0.0, INITIAL_HEIGHT]))
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=[self.kitchenpot],
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True
        )
        
        
        for idx in range(1, 5): self._metric_stage(idx , False)

        
    def _on_step(self):
        self._metric_step()

    def _success(self) -> bool:
        self._success_check = True
        if self.kitchenpot.is_colliding(self.cabinet_1):
            self._success_check = False
        if self.kitchenpot.is_colliding(self.cabinet_2):
            self._success_check = False

        pos = self.kitchenpot.body.get_position()
        if pos[2] < SUCCESS_HEIGHT:
            self._success_check = False

        up = np.array([0, 0, 1])
        right = np.array([0, -1, 0])
        kitchenpot_up = Quaternion(self.kitchenpot.body.get_quaternion()).rotate(up)
        phi = np.arccos(np.clip(np.dot(kitchenpot_up, right), -1.0, 1.0))
        theta = np.arccos(np.clip(np.dot(kitchenpot_up, up), -1.0, 1.0))
        if np.abs(phi - np.pi / 2) > SUCCESS_ROT or (theta > np.pi / 2):
            self._success_check = False
            
            
        # ———————————– distance checking ——————————————
        z_now = self.kitchenpot.body.get_position()[2]
        lift_distance = z_now - INITIAL_HEIGHT
        left_gripper_dist = distance(self.robot.grippers[HandSide.LEFT].body, self.kitchenpot.body)
        right_gripper_dist = distance(self.robot.grippers[HandSide.RIGHT].body, self.kitchenpot.body)
        
        # ———————————— stage monitoring ————————————————
        grasping = all(self.robot.is_gripper_holding_object(self.kitchenpot, side) for side in self.robot.grippers)
        grasping_left = self.robot.is_gripper_holding_object(self.kitchenpot, HandSide.LEFT) 
        grasping_right = self.robot.is_gripper_holding_object(self.kitchenpot, HandSide.RIGHT) 
        lift_pot = pos[2] > SUCCESS_HEIGHT
        pot_rot = np.abs(phi - np.pi / 2) < SUCCESS_ROT or (theta < np.pi / 2)
        
        if grasping_left: self._metric_stage(1)  
        if grasping_right: self._metric_stage(2)
        if lift_pot: self._metric_stage(3)
        if pot_rot and lift_pot and grasping: self._metric_stage(4)

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            pose_error=np.abs(phi - np.pi / 2),
            target_distance={
                "lift distance": lift_distance,
                "left gripper-kitchenpot distance": left_gripper_dist,
                "right gripper-kitchenpot distance": right_gripper_dist
            }
        )

        return self._success_check

    def _fail(self) -> bool:
        if super()._fail():
            return True
        if self.kitchenpot.is_colliding(self.floor):
            return True
        return False
    
    def _get_task_info(self):
        """Expose metrics every step (optional) or only at episode end."""
        return getattr(self, "_final_metrics", {})
    
class LiftPotPosition(LiftPot):
    """Lift pot with position randomization"""
    def _on_reset(self):
        super()._on_reset()

        spawn = self.get_spawn_boundary("workspace")
        spawn.clear()
        spawn.sample(self.kitchenpot, ignore_collisions=False, min_distance=0.1)

class LiftPotOrientation(LiftPot):
    """Lift pot with orientation randomization"""
    def _on_reset(self):
        super()._on_reset()
        angle = np.random.uniform(-self._POT_ROT_BOUNDS, self._POT_ROT_BOUNDS)
        self.kitchenpot.body.set_quaternion(Quaternion(axis=[0, 0, 1], angle=angle).elements)


class LiftPotPositionAndOrientation(LiftPot):
    """Lift pot with position AND orientation randomization"""
    def _on_reset(self):
        super()._on_reset()

        spawn = self.get_spawn_boundary("workspace")
        spawn.clear()
        spawn.sample(self.kitchenpot, ignore_collisions=False, min_distance=0.1)
        angle = np.random.uniform(-self._POT_ROT_BOUNDS, self._POT_ROT_BOUNDS)
        self.kitchenpot.body.set_quaternion(Quaternion(axis=[0, 0, 1], angle=angle).elements)
