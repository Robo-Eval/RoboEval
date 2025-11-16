"""Manipulation tasks."""
from abc import ABC

import numpy as np
from pyquaternion import Quaternion

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH, HandSide
from roboeval.envs.props.items import Cube, Rod
from roboeval.envs.props.tables import Table
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance


"""------------- Stack Two Blocks Tasks -------------"""
class _StackBlocksEnv(RoboEvalEnv):
    """Base environment stacking blocks."""
    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    
    _NUM_BLOCKS = 2
    _BLOCKS_POS = np.array([0.5, 0, 1])
    _BLOCKS_POS_EXTENTS = np.array([0.1, 0.25])
    _BLOCKS_STEP = 0.15
    _BLOCKS_POS_BOUNDS = np.array([0.05, 0.05, 0])
    _BLOCKS_ROT_BOUNDS = np.deg2rad([0, 0, 180])
    _BLOCKS_ROT_BOUNDS_SINGLE = np.deg2rad(180)

    def _initialize_env(self):
        self.blocks = [Cube(self._mojo) for _ in range(self._NUM_BLOCKS)]
        self.table = self._preset.get_props(Table)[0]

    def _on_reset(self):
        points = get_random_points_on_plane(
            len(self.blocks),
            self._BLOCKS_POS,
            self._BLOCKS_POS_EXTENTS,
            self._BLOCKS_STEP,
        )
        for block, point in zip(self.blocks, points):
            block.set_pose(
                point,
                position_bounds=self._BLOCKS_POS_BOUNDS,
                rotation_bounds=self._BLOCKS_ROT_BOUNDS,
            )

class StackTwoBlocks(_StackBlocksEnv, MetricRolloutEval):
    """Stack two static blocks on top of each other (static)"""
    
    _BLOCK_1_POS = np.array([0.5, -0.35, 1])
    _BLOCK_2_POS = np.array([0.5, 0.35, 1])
    _SUCCESS_DIST = 0.1
    
    _success_check = True
    _final_metrics = {}
    
    def _initialize_env(self):
        super()._initialize_env()
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.blocks,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True       
        )
        
    def _on_reset(self):
        self.blocks[0].set_pose(self._BLOCK_1_POS)
        self.blocks[1].set_pose(self._BLOCK_2_POS)
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.blocks,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True      
        )
        
        self._init_z_1_pos = self.blocks[0].body.get_position()[2] - 0.0301
        self._init_z_2_pos = self.blocks[1].body.get_position()[2] - 0.0301
        for idx in range(1, 4): self._metric_stage(idx, False)

        
    def _on_step(self):
        self._metric_step()
        
    def _success(self) -> bool:
        self._success_check = True
        blocks_sorted = sorted(self.blocks, key=lambda b: b.body.get_position()[2])
        dist_blocks = {}
        
        if not blocks_sorted[0].is_colliding(self.table):
            self._success_check = False
        if not blocks_sorted[1].is_colliding(blocks_sorted[0]) or blocks_sorted[1].is_colliding(self.table):
            self._success_check = False

        for idx, block in enumerate(self.blocks):
            for side in self.robot.grippers:
                dist_blocks[f"block_{idx}-gripper_{side.value}"] = distance(self.robot.grippers[side].body, block.body)
                if self.robot.is_gripper_holding_object(block, side):
                    self._success_check = False
                    
        # –––––––––––distance checking –––––––––––––
        cube_1_lift_distance = self.blocks[0].body.get_position()[2] - self._init_z_1_pos
        cube_2_lift_distance = self.blocks[1].body.get_position()[2] - self._init_z_2_pos
        
        dist_blocks.update(
            {
                "cube 1 lift distance": cube_1_lift_distance,
                "cube 2 lift distance": cube_2_lift_distance
            }
        )
        
        # ––––––––––– stage monitoring ––––––––––––––
        grasping_cube_1 = any(self.robot.is_gripper_holding_object(self.blocks[0], side) for side in self.robot.grippers)
        grasping_cube_2 = any(self.robot.is_gripper_holding_object(self.blocks[1], side) for side in self.robot.grippers)
        block_stack = blocks_sorted[0].is_colliding(self.table) and blocks_sorted[1].is_colliding(blocks_sorted[0])
        block_table_check = blocks_sorted[1].is_colliding(self.table)
        ungrasped_cube = self.blocks[0] if not grasping_cube_1 and grasping_cube_2 else self.blocks[1]
        
        if grasping_cube_1 or grasping_cube_2: self._metric_stage(1) # flag the cube that is ungrasped and add a stage to check the untouched cube is grasped next
        if any(self.robot.is_gripper_holding_object(ungrasped_cube, side) for side in self.robot.grippers): self._metric_stage(2)
        if block_stack and not block_table_check: self._metric_stage(3)
        
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=dist_blocks
        )
        return self._success_check

    def _fail(self) -> bool:
        if super()._fail():
            return True
        for block in self.blocks:
            if block.is_colliding(self.floor):
                return True
        return False
    
    def _get_task_info(self):   
        """Expose metrics every step (optional) or only at episode end."""
        return getattr(self, "_final_metrics", {})
    
class StackTwoBlocksPosition(StackTwoBlocks):
    """Stack two blocks with position randomization"""
    _BLOCKS_STEP = 0.05

    def _on_reset(self):
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.blocks,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True      
        )
        points = get_random_points_on_plane(
            len(self.blocks),
            self._BLOCKS_POS,
            self._BLOCKS_POS_EXTENTS,
            self._BLOCKS_STEP,
        )
        for block, point in zip(self.blocks, points):
            block.body.set_position(point)
            
        self._init_z_1_pos = self.blocks[0].body.get_position()[2] - 0.0301
        self._init_z_2_pos = self.blocks[1].body.get_position()[2] - 0.0301
        for idx in range(1, 4): self._metric_stage(idx, False)


class StackTwoBlocksOrientation(StackTwoBlocks):
    """Stack two blocks with orientation randomization"""
    def _on_reset(self):
        super()._on_reset()
        
        for block in self.blocks:
            angle = np.random.uniform(-self._BLOCKS_ROT_BOUNDS_SINGLE, self._BLOCKS_ROT_BOUNDS_SINGLE)
        
            block.body.set_quaternion(
                Quaternion(axis=[0, 0, 1], angle=angle).elements
            )
            
        self._init_z_1_pos = self.blocks[0].body.get_position()[2] - 0.0301
        self._init_z_2_pos = self.blocks[1].body.get_position()[2] - 0.0301

class StackTwoBlocksPositionAndOrientation(StackTwoBlocks):
    """Stack two blocks with position and orientation randomization"""
    _BLOCKS_STEP = 0.15

    def _on_reset(self):

        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.blocks,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True      
        )
        points = get_random_points_on_plane(
            len(self.blocks),
            self._BLOCKS_POS,
            self._BLOCKS_POS_EXTENTS,
            self._BLOCKS_STEP,
        )
        for block, point in zip(self.blocks, points):
            # angle = np.random.uniform(-self._BLOCKS_ROT_BOUNDS_SINGLE, self._BLOCKS_ROT_BOUNDS_SINGLE)
        
            block.set_pose(
                point,
                position_bounds=self._BLOCKS_POS_BOUNDS,
                rotation_bounds=self._BLOCKS_ROT_BOUNDS
            )
            
        self._init_z_1_pos = self.blocks[0].body.get_position()[2] - 0.0301
        self._init_z_2_pos = self.blocks[1].body.get_position()[2] - 0.0301
        for idx in range(1, 4): self._metric_stage(idx, False)


"""------------- Cube Handover Tasks -------------"""
class CubeHandover(RoboEvalEnv, MetricRolloutEval):
    """Hand over a Static Cube from the left gripper to the right gripper."""
    _PRESET_PATH = PRESETS_PATH / "cube_handover.yaml"
    _BLOCKS_POS = np.array([0.5, 0, 1])
    _LIFT_DIST = 0.1
    
    success_check = False
    _final_metrics = {}
    left_target_distance = 0
    right_target_distance = 0

    def _initialize_env(self):
        """
        Called once when the environment is constructed.
        Create the cube(s) here and store references.
        """
        # Create a single cube or 'Prop' (assuming you have a Cube or Prop class).
        # If you need to add geometry, you can do so here. For example:
        self.cube = Rod(self._mojo)
        self.initial_gripper = None
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cube,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True     
        )


    def _on_reset(self):
        """
        We want to place the cube in the left gripper at the start of each episode.
        """       
        self.initial_gripper = None
        self.cube.set_pose(self._BLOCKS_POS)
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cube,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True     
        )
        
        self._init_z_pos = self.cube.body.get_position()[2] - 0.0301
        for idx in range(1, 3): self._metric_stage(idx, False)

    def _on_step(self):
        self._metric_step()
        
    def _success(self) -> bool:
        """
        We succeed if the right gripper is holding the cube, and the left is no longer holding it.
        """
        
        """Succeed only if the cube is handed from one gripper to the other."""
        self.success_check = False
        right_holding = self.robot.is_gripper_holding_object(self.cube, HandSide.RIGHT)
        left_holding = self.robot.is_gripper_holding_object(self.cube, HandSide.LEFT)

        # Track the initial gripper if not yet recorded
        if self.initial_gripper is None:
            if left_holding:
                self.initial_gripper = HandSide.LEFT
            elif right_holding:
                self.initial_gripper = HandSide.RIGHT

        # Validate success if cube is transferred to the opposite gripper
        if self.initial_gripper == HandSide.LEFT and right_holding and not left_holding:
            self.success_check = True
        elif self.initial_gripper == HandSide.RIGHT and left_holding and not right_holding:
            self.success_check = True
            
        # –––––––––––––– distance checking –––––––––––––––––––
        self.left_target_distance = distance(self.cube.body, self.robot.grippers[HandSide.LEFT].body)
        self.right_target_distance = distance(self.cube.body, self.robot.grippers[HandSide.RIGHT].body)
        self.lift_distance = self.cube.body.get_position()[2] - self._init_z_pos
        
        # ––––––––––––––– stage monitoring –––––––––––––––––––––
        grasping = right_holding or left_holding
        cube_transfer = self.initial_gripper == HandSide.LEFT and right_holding or self.initial_gripper == HandSide.RIGHT and left_holding
        
        if grasping: self._metric_stage(1)
        if cube_transfer: self._metric_stage(2)

        self._final_metrics = self._metric_finalize(
            success_flag=self.success_check,
            target_distance={
                "cube to left gripper": self.left_target_distance,
                "cube to right gripper": self.right_target_distance,
                "lift distance": self.lift_distance
            }
        )
        return self.success_check

    def _fail(self) -> bool:
        """
        Failure conditions. For instance, if the cube is dropped to the floor or 
        if the episode times out. 
        """
        # check if the parent class has any default fail conditions
        if super()._fail():
            return True
        
        # check if the cube is colliding with it (i.e., dropped).
        if self.cube.is_colliding(self._floor):
            return True
        
        return False

    def _get_task_info(self):
        """Expose metrics every step (optional) or only at episode end."""
        return getattr(self, "_final_metrics", {})
        
class CubeHandoverPosition(CubeHandover):
    """Hand over a Cube from the left gripper to the right gripper."""
    _PRESET_PATH = PRESETS_PATH / "cube_handover.yaml"
    _BLOCKS_POS_BOUNDS = np.array([0.0, 0.05, 0])
    _BLOCKS_ROT_BOUNDS = np.deg2rad([0, 0, 0])
    _BLOCKS_POS_EXTENTS = np.array([0.1, 0.25])
    _BLOCKS_STEP = 0.15
    _BLOCKS_POS = np.array([0.5, 0, 1])
    
    success_check = False
    _final_metrics = {}
    left_target_distance = 0
    right_target_distance = 0


    def _on_reset(self):
        """
        We want to place the cube in the left gripper at the start of each episode.
        """
        self.initial_gripper = None

        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cube,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True     
        )

        points = get_random_points_on_plane(
            1,
            self._BLOCKS_POS,
            self._BLOCKS_POS_EXTENTS,
            self._BLOCKS_STEP,
        )
        
        for point in points:
            self.cube.set_pose(
                point,
                position_bounds=self._BLOCKS_POS_BOUNDS,
                rotation_bounds=self._BLOCKS_ROT_BOUNDS,
            )

        self._init_z_pos = self.cube.body.get_position()[2] - 0.0301
        for idx in range(1, 3): self._metric_stage(idx, False)
        

class CubeHandoverOrientation(CubeHandover):
    """Hand over a Cube from the left gripper to the right gripper."""
    _PRESET_PATH = PRESETS_PATH / "cube_handover.yaml"
    _BLOCKS_POS_BOUNDS = np.array([0, 0, 0])
    _BLOCKS_ROT_BOUNDS = np.deg2rad([0, 0, 180])
    _BLOCKS_POS_EXTENTS = np.array([0.1, 0.25])
    _BLOCKS_STEP = 0.15


    _BLOCKS_POS = np.array([0.5, 0, 1])
    
    success_check = False
    _final_metrics = {}
    left_target_distance = 0
    right_target_distance = 0


    def _on_reset(self):
        """
        We want to place the cube in the left gripper at the start of each episode.
        """
        self.initial_gripper = None

        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cube,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True     
        )       
        self.cube.set_pose(
            position=self._BLOCKS_POS,
            rotation_bounds=self._BLOCKS_ROT_BOUNDS,
        )
        self._init_z_pos = self.cube.body.get_position()[2] - 0.0301
        for idx in range(1, 3): self._metric_stage(idx, False)
        


class CubeHandoverPositionAndOrientation(CubeHandover):
    """Hand over a Cube from the left gripper to the right gripper."""
    _PRESET_PATH = PRESETS_PATH / "cube_handover.yaml"
    _BLOCKS_POS_BOUNDS = np.array([0.0, 0.05, 0])
    _BLOCKS_ROT_BOUNDS = np.deg2rad([0, 0, 180])
    _BLOCKS_POS_EXTENTS = np.array([0.1, 0.25])
    _BLOCKS_STEP = 0.15


    _BLOCKS_POS = np.array([0.5, 0, 1])
    
    success_check = False
    _final_metrics = {}
    left_target_distance = 0
    right_target_distance = 0
    
    def _on_reset(self):
        """
        We want to place the cube in the left gripper at the start of each episode.
        """       
        self.initial_gripper = None
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cube,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True     
        )
        points = get_random_points_on_plane(
            1,
            self._BLOCKS_POS,
            self._BLOCKS_POS_EXTENTS,
            self._BLOCKS_STEP,
        )
        
        for point in points:
            self.cube.set_pose(
                point,
                position_bounds=self._BLOCKS_POS_BOUNDS,
                rotation_bounds=self._BLOCKS_ROT_BOUNDS,
            )
        self._init_z_pos = self.cube.body.get_position()[2] - 0.0301
        for idx in range(1, 3): self._metric_stage(idx, False)
    """Hand over a Cube from the left gripper to the right gripper."""
    _PRESET_PATH = PRESETS_PATH / "cube_handover.yaml"
    _BLOCKS_POS_BOUNDS = np.array([0.05, 0, 0])
    _BLOCKS_ROT_BOUNDS = np.deg2rad([0, 180, 0])
    _BLOCKS_POS_EXTENTS = np.array([0.1, 0.25])
    _BLOCKS_STEP = 0.15

    _BLOCKS_POS = np.array([0.5, 0, 1.1])
    _BLOCKS_ROT = np.array([0.7071, 0.7071, 0, 0])
    _LIFT_DIST = 0.1
    
    success_check = False
    _final_metrics = {}
    left_target_distance = 0
    right_target_distance = 0

    def _initialize_env(self):
        """
        Called once when the environment is constructed.
        Create the cube(s) here and store references.
        """
        # Create a single cube or 'Prop' (assuming you have a Cube or Prop class).
        # If you need to add geometry, you can do so here. For example:
        self.cube = Rod(self._mojo)
        self.initial_gripper = None
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cube,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True     
        )
        self._init_z_pos = self.cube.body.get_position()[2]

    def _on_reset(self):
        """
        We want to place the cube in the left gripper at the start of each episode.
        """
        self.initial_gripper = None
        points = get_random_points_on_plane(
            1,
            self._BLOCKS_POS,
            self._BLOCKS_POS_EXTENTS,
            self._BLOCKS_STEP,
        )

        self.cube.set_pose(
            points[0],
            self._BLOCKS_ROT,
            position_bounds=self._BLOCKS_POS_BOUNDS,
            rotation_bounds=self._BLOCKS_ROT_BOUNDS,
        )
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cube,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True     
        )
        
        self._init_z_pos = self.cube.body.get_position()[2] - 0.05
        for idx in range(1, 3): self._metric_stage(idx, False)

    def _on_step(self):
        self._metric_step()

    def _success(self) -> bool:
        """
        We succeed if the right gripper is holding the cube, and the left is no longer holding it.
        """
        
        """Succeed only if the cube is handed from one gripper to the other."""

        self.success_check = False

        right_holding = self.robot.is_gripper_holding_object(self.cube, HandSide.RIGHT)
        left_holding = self.robot.is_gripper_holding_object(self.cube, HandSide.LEFT)

        # Track the initial gripper if not yet recorded
        if self.initial_gripper is None:
            if left_holding:
                self.initial_gripper = HandSide.LEFT
            elif right_holding:
                self.initial_gripper = HandSide.RIGHT

        # Validate success if cube is transferred to the opposite gripper
        if self.initial_gripper == HandSide.LEFT and right_holding and not left_holding:
            self.success_check = True
        elif self.initial_gripper == HandSide.RIGHT and left_holding and not right_holding:
            self.success_check = True

        # –––––––––––––– distance checking –––––––––––––––––––
        self.left_target_distance = distance(self.cube.body, self.robot.grippers[HandSide.LEFT].body)
        self.right_target_distance = distance(self.cube.body, self.robot.grippers[HandSide.RIGHT].body)
        self.lift_distance = self.cube.body.get_position()[2] - self._init_z_pos
        
        # ––––––––––––––– stage monitoring –––––––––––––––––––––
        grasping = right_holding or left_holding
        cube_transfer = self.initial_gripper == HandSide.LEFT and right_holding or self.initial_gripper == HandSide.RIGHT and left_holding
        
        if grasping: self._metric_stage(1)
        if cube_transfer: self._metric_stage(2)

        self._final_metrics = self._metric_finalize(
            success_flag=self.success_check,
            target_distance={
                "cube to left gripper": self.left_target_distance,
                "cube to right gripper": self.right_target_distance,
                "lift distance": self.lift_distance
            }
        )

        return self.success_check

    def _fail(self) -> bool:
        """
        Failure conditions. For instance, if the cube is dropped to the floor or 
        if the episode times out. 
        """
        # check if the parent class has any default fail conditions
        if super()._fail():
            return True
        
        # check if the cube is colliding with it (i.e., dropped).
        if self.cube.is_colliding(self._floor):
            return True
        
        return False
    def _get_task_info(self):
        """Expose metrics every step (optional) or only at episode end."""
        return getattr(self, "_final_metrics", {})


class VerticalCubeHandover(RoboEvalEnv, MetricRolloutEval):
    """Hand over a Cube from the left gripper to the right gripper."""
    _PRESET_PATH = PRESETS_PATH / "cube_handover.yaml"
    _BLOCKS_POS_BOUNDS = np.array([0.05, 0, 0])
    _BLOCKS_ROT_BOUNDS = np.deg2rad([0, 180, 0])
    _BLOCKS_POS_EXTENTS = np.array([0.1, 0.25])
    _BLOCKS_STEP = 0.15

    _BLOCKS_POS = np.array([0.5, 0, 1.1])
    _BLOCKS_ROT = np.array([0.7071, 0.7071, 0, 0])
    _LIFT_DIST = 0.1
    
    success_check = False
    _final_metrics = {}
    left_target_distance = 0
    right_target_distance = 0

    def _initialize_env(self):
        """
        Called once when the environment is constructed.
        Create the cube(s) here and store references.
        """
        self.cube = Rod(self._mojo)
        self.initial_gripper = None
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cube,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True     
        )
        self._init_z_pos = self.cube.body.get_position()[2]

    def _on_reset(self):
        """
        We want to place the cube in the left gripper at the start of each episode.
        """
        self.initial_gripper = None
        points = get_random_points_on_plane(
            1,
            self._BLOCKS_POS,
            self._BLOCKS_POS_EXTENTS,
            self._BLOCKS_STEP,
        )

        self.cube.set_pose(
            points[0],
            self._BLOCKS_ROT,
            position_bounds=self._BLOCKS_POS_BOUNDS,
            rotation_bounds=self._BLOCKS_ROT_BOUNDS,
        )
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cube,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True     
        )
        
        self._init_z_pos = self.cube.body.get_position()[2] - 0.05
        for idx in range(1, 3): self._metric_stage(idx, False)

    def _on_step(self):
        self._metric_step()

    def _success(self) -> bool:
        """
        We succeed if the right gripper is holding the cube, and the left is no longer holding it.
        """
        
        """Succeed only if the cube is handed from one gripper to the other."""

        self.success_check = False

        right_holding = self.robot.is_gripper_holding_object(self.cube, HandSide.RIGHT)
        left_holding = self.robot.is_gripper_holding_object(self.cube, HandSide.LEFT)

        # Track the initial gripper if not yet recorded
        if self.initial_gripper is None:
            if left_holding:
                self.initial_gripper = HandSide.LEFT
            elif right_holding:
                self.initial_gripper = HandSide.RIGHT

        # Validate success if cube is transferred to the opposite gripper
        if self.initial_gripper == HandSide.LEFT and right_holding and not left_holding:
            self.success_check = True
        elif self.initial_gripper == HandSide.RIGHT and left_holding and not right_holding:
            self.success_check = True

        # –––––––––––––– distance checking –––––––––––––––––––
        self.left_target_distance = distance(self.cube.body, self.robot.grippers[HandSide.LEFT].body)
        self.right_target_distance = distance(self.cube.body, self.robot.grippers[HandSide.RIGHT].body)
        self.lift_distance = self.cube.body.get_position()[2] - self._init_z_pos
        
        # ––––––––––––––– stage monitoring –––––––––––––––––––––
        grasping = right_holding or left_holding
        cube_transfer = self.initial_gripper == HandSide.LEFT and right_holding or self.initial_gripper == HandSide.RIGHT and left_holding
        
        if grasping: self._metric_stage(1)
        if cube_transfer: self._metric_stage(2)

        self._final_metrics = self._metric_finalize(
            success_flag=self.success_check,
            target_distance={
                "cube to left gripper": self.left_target_distance,
                "cube to right gripper": self.right_target_distance,
                "lift distance": self.lift_distance
            }
        )

        return self.success_check

    def _fail(self) -> bool:
        """
        Failure conditions. For instance, if the cube is dropped to the floor or 
        if the episode times out. 
        """
        # check if the parent class has any default fail conditions
        if super()._fail():
            return True
        
        # check if the cube is colliding with it (i.e., dropped).
        if self.cube.is_colliding(self._floor):
            return True
        
        return False
        
    def _get_task_info(self):
        """Expose metrics every step (optional) or only at episode end."""
        return getattr(self, "_final_metrics", {})
