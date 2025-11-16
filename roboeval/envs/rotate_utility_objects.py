from abc import ABC

import numpy as np
from pyquaternion import Quaternion

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH, HandSide
from roboeval.envs.props.garage import Valve
from roboeval.envs.props.tables import Table
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.utils.physics_utils import distance
from roboeval.envs.props.obstacle import Obstacle
from roboeval.utils.metric_rollout import MetricRolloutEval

class _RotateObjectEnv(RoboEvalEnv, MetricRolloutEval):
    """Base env for rotating objects"""

    _PRESET_PATH = PRESETS_PATH / "rotate_obj.yaml"
    
    _success_check = True
    _final_metrics = {}

    def _initialize_env(self):
        self.table = self._preset.get_props(Table)[0]
        
    def _get_task_info(self):
        """Expose metrics every step (optional) or only at episode end."""
        return getattr(self, "_final_metrics", {})

        
class RotateValve(_RotateObjectEnv):
    """
    Base environment for rotating valve (static)
    Success: rotate both valves 5 degrees counterclockwise
    """
    _VALVE_COUNT = 2
    _VALVE_POS = np.array([0.3, -0.1, 0.94])
    _VALVE_ROT = np.deg2rad(90)
    _VALVE_STEP = 0.60
    _VALVE_POS_EXTENTS = np.array([0.35, 0.6])
    _VALVE_POS_BOUNDS = np.array([0.1, 0.13, 0])
    _VALVE_ROT_BOUNDS = np.deg2rad(30)

    def _initialize_env(self):
        super()._initialize_env()
        self.valves = [Valve(self._mojo) for _ in range(self._VALVE_COUNT)]

        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.valves,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True  
        )

    def _on_reset(self):

        points = [[0.3, -0.4, 0.94], [0.3, 0.2, 0.94]]

        for valve, point in zip(self.valves, points):
            valve.body.set_position(point)
            valve.set_state(0)
            valve.body.set_quaternion(Quaternion(axis=[0, 0, 1], angle=self._VALVE_ROT).elements)
            
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.valves,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True  
        )
        for idx in range(1, 5): self._metric_stage(idx , False)

    
    def _on_step(self):
        self._metric_step()
        
    def _success(self) -> bool: 
        self._success_check = True   
        dist_check = {}

        for idx, valve in enumerate(self.valves):
            # for side in self.robot.grippers:
            #     if self.robot.is_gripper_holding_object(valve, side):
            #         return False
            # if not self.table.is_colliding(valve) or valve.is_colliding(self._floor):
            #     return False
            if valve.get_state() <= 0.10: 
                self._success_check = False
                
            # ————————————— distance check —————————————
            dist_check[f"valve_{idx}-right gripper distance"] = distance(valve.valve, self.robot.grippers[HandSide.RIGHT].body)
            dist_check[f"valve_{idx}-left gripper distance"] = distance(valve.valve, self.robot.grippers[HandSide.LEFT].body)

        # –––––––––––– stage monitoring ––––––––––––––––
        valve_1_rot = self.valves[0].get_state()
        valve_1_grasp = any(self.robot.is_gripper_holding_object(self.valves[0].valve, side) for side in self.robot.grippers)
        valve_2_rot = self.valves[1].get_state()
        valve_2_grasp = any(self.robot.is_gripper_holding_object(self.valves[1].valve, side) for side in self.robot.grippers)
            
        if valve_1_grasp: self._metric_stage(1) 
        if valve_1_rot >= 0.1: self._metric_stage(2) 
        if valve_2_grasp: self._metric_stage(3)
        if valve_2_rot >= 0.1: self._metric_stage(4)

        
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=dist_check
        )
        return self._success_check
    
    def _fail(self) -> bool:
        if super()._fail():
            return True
        
        for valve in self.valves:
            if valve.is_colliding(self._floor):
                return True
        return False
    
class RotateValvePosition(RotateValve):
    """Rotate valve with randomization of the position of the valves"""
    def _on_reset(self):
        super()._on_reset()
        
        for valve in self.valves:

            points = get_random_points_on_plane(
                self._VALVE_COUNT,
                self._VALVE_POS,
                self._VALVE_POS_EXTENTS,
                self._VALVE_STEP,
                self._VALVE_POS_BOUNDS,
            )
            
            for valve, point in zip(self.valves, points):
                valve.body.set_position(point)

class RotateValvePositionAndOrientation(RotateValve):
    """Rotate valve with randomization of the position and orientation of the valves"""
    _VALVE_POS_BOUNDS = np.array([0.07, 0.13, 0])
    _VALVE_ROT_BOUNDS = np.deg2rad(30)
    def _on_reset(self):
        super()._on_reset()
        for valve in self.valves:

            points = get_random_points_on_plane(
                self._VALVE_COUNT,
                self._VALVE_POS,
                self._VALVE_POS_EXTENTS,
                self._VALVE_STEP,
                self._VALVE_POS_BOUNDS,
            )
            
            for valve, point in zip(self.valves, points):
                valve.body.set_position(point)
                valve.set_state(0)
                angle = np.random.uniform(-self._VALVE_ROT_BOUNDS, self._VALVE_ROT_BOUNDS)
                valve.body.set_quaternion(
                    Quaternion(axis=[0, 0, 1], angle=self._VALVE_ROT+angle).elements
                )

    

class RotateValveObstacle(RotateValvePositionAndOrientation):
    """Rotate valve with a verticle obstical on the table (includes position and orientation randomization)"""
    _OBSTACLE_POS = np.array([0.36, 0, 1.1])
    _OBSTACLE_POS_BOUNDS = np.array([0.04, 0.13, 0])
    _OBSTACLE_POS_EXTENTS = np.array([0.35, 0.6])
    _OBSTACLE_ROT = np.deg2rad(90)
    _OBSTACLE_ROT_BOUNDS = np.deg2rad(30)
    _OBSTACLE_STEP = 0.60
    _VALVE_POS = np.array([0.32, -0.1, 0.94])
    _VALVE_ROT_BOUNDS = np.deg2rad(28)

    def _initialize_env(self):
        super()._initialize_env()
        self.obstacle = Obstacle(self._mojo)

    def _on_reset(self):
        super()._on_reset()
        
        points = get_random_points_on_plane(
            1,
            self._OBSTACLE_POS,
            self._OBSTACLE_POS_EXTENTS,
            self._OBSTACLE_STEP,
            self._OBSTACLE_POS_BOUNDS,
        )
        angle = np.random.uniform(-self._OBSTACLE_ROT_BOUNDS, self._OBSTACLE_ROT_BOUNDS)
        
        self.obstacle.set_pose(
            position=points[0],
            quat=Quaternion(axis=[0, 0, 1], angle=self._OBSTACLE_ROT + angle).elements,
        )