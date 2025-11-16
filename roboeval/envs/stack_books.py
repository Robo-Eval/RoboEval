from abc import ABC

import numpy as np
from pyquaternion import Quaternion

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH
from roboeval.envs.props.book_shelf import BookShelf
from roboeval.envs.props.items import Book
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.utils.physics_utils import distance
import quaternion
from scipy.spatial.transform import Rotation
from mojo.elements import Body, Geom
from mojo.elements.consts import GeomType
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.const import HandSide  # LEFT / RIGHT enums

class _StackBooksEnv(RoboEvalEnv, ABC, MetricRolloutEval):
    """Base Environment for Stacking Books"""

    _PRESET_PATH = PRESETS_PATH / "place_books.yaml"


    _BOOK_COUNT = 2

    _BOOK_POS = np.array([0.4, 0, 0.72])
    _BOOK_ROT = np.deg2rad(180)
    _BOOK_STEP = 0.15
    _BOOK_POS_EXTENTS = np.array([0.1, 0.25])
    _BOOK_POS_BOUNDS = np.array([0.02, 0.02, 0])
    _BOOK_ROT_BOUNDS = np.deg2rad(30)
    _SUCCESSFUL_DIST = 0.005
    
    _success_check = True
    _final_metrics = {}

    def _initialize_env(self):
        self.book_shelf=  self._preset.get_props(BookShelf)[0]
        self.books = [Book(self._mojo) for _ in range(self._BOOK_COUNT)]
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.books,
            slip_sample_window=20,
            track_collisions=True,
            robot=self.robot
        )
        self.book_init_height = self.books[0].body.get_position()[2]
        
    def _get_task_info(self):
        """Expose metrics every step (optional) or only at episode end."""
        return getattr(self, "_final_metrics", {})
        
class StackSingleBookShelf(_StackBooksEnv):
    """
    Simple pick and place task: Robot transports books from counter to a shelf and checks if the books are colliding with either the top or lower shelf
    """

    _BOOK_COUNT = 1
    SUCCESS_HEIGHT = 0.1
    
    def _on_reset(self):
        self.book_shelf.set_state(state=0)
        self.book_shelf.set_state(target_state={
            "bookshelf_v2/joint_0": 0,
            "bookshelf_v2/joint_1": 0,
            "bookshelf_v2/joint_2": 0
        })

        self.books[0].body.set_position(self._BOOK_POS)
        self.books[0].body.set_quaternion(Quaternion(axis=[0, 0, 1], angle=np.deg2rad(90)).elements)
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.books,
            slip_sample_window=20,
            track_collisions=True,
            robot=self.robot
        )
        
        self.book_init_height = self.books[0].body.get_position()[2] - .005
        for idx in range(1, 5): self._metric_stage(idx, False)


    def _on_step(self):
        self._metric_step()

    def _success(self) -> bool:
        self._success_check = True

        for book in self.books:
            #  We check if the book is colliding with the mesh of either the top or bottom shelf
            if not book.is_colliding(self.book_shelf.lower_shelf_body) and not book.is_colliding(self.book_shelf.upper_shelf_body):
                self._success_check = False
  
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(book, side):
                    self._success_check = False   
         
        # ––––––– define stage boundaries ––––––––––––––––––––           
        grasped  = any(
            self.robot.is_gripper_holding_object(self.books[0], side)
            for side in self.robot.grippers
        )
        on_shelf = (
            self.books[0].is_colliding(self.book_shelf.lower_shelf_body)
            or self.books[0].is_colliding(self.book_shelf.upper_shelf_body)
        )
        lifted   = (self.books[0].body.get_position()[2] - self.book_init_height) >= self.SUCCESS_HEIGHT
        released = on_shelf and not grasped
        
        # ─────── monitor task progression ───────────────────── 
        if grasped:   self._metric_stage(1)   # holding
        if lifted:    self._metric_stage(2)   # lifted: ——–––– KEEP
        if on_shelf:  self._metric_stage(3)   # touching shelf
        if released:  self._metric_stage(4)   # placed & released
                    
        book_lift_dist = self.books[0].body.get_position()[2] - self.book_init_height
        upper_shelf_dist = distance(self.books[0].body, self.book_shelf.upper_shelf_body)
        lower_shelf_dist = distance(self.books[0].body, self.book_shelf.lower_shelf_body)
        

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "book lift distance": book_lift_dist,
                "book-upper shelf": upper_shelf_dist,
                "book-lower shelf": lower_shelf_dist
            }
            )
        return self._success_check
    
class StackSingleBookShelfPosition(StackSingleBookShelf):
    """
    Stack single book on shelf with position randomization
    """
    
    def _on_reset(self):
        self.book_shelf.set_state(state=0)
        self.book_shelf.set_state(target_state={
            "bookshelf_v2/joint_0": 0,
            "bookshelf_v2/joint_1": 0,
            "bookshelf_v2/joint_2": 0
        })


        points = get_random_points_on_plane(
            len(self.books),
            self._BOOK_POS,
            self._BOOK_POS_EXTENTS,
            self._BOOK_STEP,
            self._BOOK_POS_BOUNDS,
        )
        for book, point in zip(self.books, points):
            book.body.set_position(point)
            self.books[0].body.set_quaternion(Quaternion(axis=[0, 0, 1], angle=np.deg2rad(90)).elements)
            
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.books,
            slip_sample_window=20,
            track_collisions=True,
            robot=self.robot
        )
        
        self.book_init_height = self.books[0].body.get_position()[2] - .005
        for idx in range(1, 5): self._metric_stage(idx, False)

class StackSingleBookShelfPositionAndOrientation(StackSingleBookShelf):
    """
    Stack single book on shelf with position and orientation randomization
    """
    
    def _on_reset(self):
        self.book_shelf.set_state(state=0)
        self.book_shelf.set_state(target_state={
            "bookshelf_v2/joint_0": 0,
            "bookshelf_v2/joint_1": 0,
            "bookshelf_v2/joint_2": 0
        })

        points = get_random_points_on_plane(
            len(self.books),
            self._BOOK_POS,
            self._BOOK_POS_EXTENTS,
            self._BOOK_STEP,
            self._BOOK_POS_BOUNDS,
        )
        for book, point in zip(self.books, points):
            book.body.set_position(point)

            angle = np.random.uniform(-self._BOOK_ROT_BOUNDS, self._BOOK_ROT_BOUNDS)
            self.books[0].body.set_quaternion(Quaternion(axis=[0, 0, 1], angle=angle).elements)
            
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.books,
            slip_sample_window=20,
            track_collisions=True,
            robot=self.robot
        )
        self.book_init_height = self.books[0].body.get_position()[2] - .005
        for idx in range(1, 5): self._metric_stage(idx, False)


class _PickBooksEnv(RoboEvalEnv, ABC, MetricRolloutEval):
    """Base initializations for picking books off the shelf"""

    _PRESET_PATH = PRESETS_PATH / "place_books.yaml"


    _BOOK_COUNT = 3

    _BOOK_SHELF_POS = np.array([0.6, -0.425, 1.1]) # Upper shelf placement 
    _BOOK_TABLE_POS = np.array([0.4, 0, 0.72])
    _BOOK_ROT = np.deg2rad(90)
    _BOOK_STEP = 0.15
    _BOOK_POS_EXTENTS = np.array([0.1, 0.25])
    _BOOK_POS_BOUNDS = np.array([0.01, 0.01, 0])
    _BOOK_ROT_BOUNDS = np.deg2rad(30)
    _SUCCESSFUL_DIST = 0.005
    
    _success_check = True
    _final_metrics = {}



    def _initialize_env(self):
        self.book_shelf=  self._preset.get_props(BookShelf)[0]
        self.books = [Book(self._mojo) for _ in range(self._BOOK_COUNT)]

        
    def _get_task_info(self):
        """Expose metrics every step (optional) or only at episode end."""
        return getattr(self, "_final_metrics", {})

class PickSingleBookFromTable(_PickBooksEnv):
    """Pick a single book up from the table (static)"""
    _BOOK_COUNT = 1
    _LIFT_SUCCESS = 0.1

    def _on_reset(self):
        self.book_shelf.set_state(state=0)
        self.book_shelf.set_state(target_state={
            "bookshelf_v2/joint_0": 0,
            "bookshelf_v2/joint_1": 0,
            "bookshelf_v2/joint_2": 0
        })

        self.books[0].body.set_position(self._BOOK_TABLE_POS)
        self.books[0].body.set_quaternion(
            Quaternion(axis=[0, 0, 1], angle=self._BOOK_ROT).elements
        )
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.books[0],
            slip_sample_window=20,
            track_collisions=True,
            robot=self.robot,    
        )
        
        self.book_init_height = self.books[0].body.get_position()[2] - .005
        for idx in range(1, 3): self._metric_stage(idx, False)

    def _on_step(self):
        self._metric_step()

    def _success(self):
        self._success_check = True

        book = self.books[0]
        if book.body.get_position()[2] < 0.77 or book.is_colliding(self.book_shelf.counter) or book.is_colliding(self._floor):
            self._success_check = False

        if not any(self.robot.is_gripper_holding_object(self.books[0], side) for side in self.robot.grippers): # At least one effector must be gripping the book
            self._success_check = False
            
        distance_from_left_gripper = distance(book.body, self.robot.grippers[HandSide.LEFT].body)
        distance_from_right_gripper = distance(book.body, self.robot.grippers[HandSide.RIGHT].body)
        book_lift_dist = self.books[0].body.get_position()[2] - self.book_init_height
        
        # –––––––– stage monitoring –––––––––––––––––––––––
        grasped = any(
            self.robot.is_gripper_holding_object(book, side)
            for side in self.robot.grippers
        )
        
        if grasped: self._metric_stage(1) # hold book or in touch with book
        if grasped and not book.is_colliding(self.book_shelf.counter) and not book.is_colliding(self._floor): self._metric_stage(2)

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "book lift distance": book_lift_dist,
                "left gripper-book distance": distance_from_left_gripper,
                "right gripper-book distance": distance_from_right_gripper
            }
            )
        return self._success_check

# Pick Single Book with position randomization
class PickSingleBookFromTablePosition(PickSingleBookFromTable):
    """Pick single book from table with position randomization"""
    _BOOK_COUNT = 1
    _LIFT_SUCCESS = 0.1

    def _on_reset(self):
        # super()._on_reset()

        self.book_shelf.set_state(state=0)
        self.book_shelf.set_state(target_state={
            "bookshelf_v2/joint_0": 0,
            "bookshelf_v2/joint_1": 0,
            "bookshelf_v2/joint_2": 0
        })

        points = get_random_points_on_plane(
            len(self.books),
            self._BOOK_TABLE_POS,
            self._BOOK_POS_EXTENTS,
            self._BOOK_STEP,
            self._BOOK_POS_BOUNDS,
        )
        for book, point in zip(self.books, points):
            book.body.set_position(point)
        
        self.books[0].body.set_quaternion(
            Quaternion(axis=[0, 0, 1], angle=self._BOOK_ROT).elements
        )
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.books[0],
            slip_sample_window=20,
            track_collisions=True,
            robot=self.robot,    
        )
        
        self.book_init_height = self.books[0].body.get_position()[2] - .005
        for idx in range(1, 3): self._metric_stage(idx, False)

    
class PickSingleBookFromTableOrientation(PickSingleBookFromTable):
    """Pick single book from table with orientation randomization"""
    _BOOK_ROT_BOUNDS = np.deg2rad(180)

    def _on_reset(self):
        # super()._on_reset()
        self.book_shelf.set_state(state=0)
        self.book_shelf.set_state(target_state={
            "bookshelf_v2/joint_0": 0,
            "bookshelf_v2/joint_1": 0,
            "bookshelf_v2/joint_2": 0
        })
        
        self.books[0].body.set_position(self._BOOK_TABLE_POS)

        angle = np.random.uniform(-self._BOOK_ROT_BOUNDS, self._BOOK_ROT_BOUNDS)
        self.books[0].body.set_quaternion(
            Quaternion(axis=[0, 0, 1], angle=angle).elements
        )

        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.books[0],
            slip_sample_window=20,
            track_collisions=True,
            robot=self.robot,    
        )

        self.book_init_height = self.books[0].body.get_position()[2] - .005
        for idx in range(1, 3): self._metric_stage(idx, False)
        


class PickSingleBookFromTablePositionAndOrientation(PickSingleBookFromTable):
    """Pick single book from table with position and orientation randomization"""
    _BOOK_ROT_BOUNDS = np.deg2rad(180)

    def _on_reset(self):

        self.book_shelf.set_state(state=0)
        self.book_shelf.set_state(target_state={
            "bookshelf_v2/joint_0": 0,
            "bookshelf_v2/joint_1": 0,
            "bookshelf_v2/joint_2": 0
        })
        
        points = get_random_points_on_plane(
            len(self.books),
            self._BOOK_TABLE_POS,
            self._BOOK_POS_EXTENTS,
            self._BOOK_STEP,
            self._BOOK_POS_BOUNDS,
        )
        for book, point in zip(self.books, points):
            book.body.set_position(point)
            angle = np.random.uniform(-self._BOOK_ROT_BOUNDS, self._BOOK_ROT_BOUNDS)
            book.body.set_quaternion(
                Quaternion(axis=[0, 0, 1], angle=angle).elements
            )

        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.books[0],
            slip_sample_window=20,
            track_collisions=True,
            robot=self.robot,    
        )

        self.book_init_height = self.books[0].body.get_position()[2] - .005
        for idx in range(1, 3): self._metric_stage(idx, False)