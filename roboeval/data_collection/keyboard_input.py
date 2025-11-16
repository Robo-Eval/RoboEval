# Standard library imports
import threading
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type, Any, Union

# Third-party imports
import numpy as np
from pynput import keyboard
from pyquaternion import Quaternion
from gymnasium.core import ActType

# Local imports
from roboeval.action_modes import ActionMode
from roboeval.robots.robot import Robot
from roboeval.roboeval_env import RoboEvalEnv
from roboeval.demonstrations.demo import TERMINATION_STEPS
from roboeval.demonstrations.demo_recorder import DemoRecorder
from roboeval.const import HandSide
from roboeval.ik.base_ik import Pose


class KeyboardControlProfile(ABC):
    """Keyboard control profile for environment interaction."""

    def __init__(self, env: RoboEvalEnv):
        """Initialize control profile with environment."""
        self._env = env

    def get_next_action(self) -> ActType:
        """Get next action based on control profile."""
        return np.zeros_like(self._env.action_space.low)

    def reset(self):
        """Reset control profile to initial state."""
        pass


@dataclass
class KeyboardTeleopStats:
    """Statistics for keyboard teleoperation."""
    is_recoding: bool = False
    time: float = 0
    reward: float = 0
    demos_counter: int = 0


class Countdown:
    """Countdown timer for time-based events."""

    def __init__(self, delay: int):
        """Initialize with specified delay in steps."""
        self._delay = delay

    def step(self):
        """Progress the countdown by one step."""
        if self._delay > 0:
            self._delay -= 1

    @property
    def is_up(self) -> bool:
        """Check if countdown has completed."""
        return self._delay <= 0


class KeyboardTeleop:
    """Keyboard teleoperation interface for RoboEval environments."""

    # Control constants
    STEPS_COUNT_FACTOR = 3
    POSITION_SMOOTHING = 0.1
    ROTATION_SMOOTHING = 0.1
    ARM_MOVEMENT_STEP = 0.005
    ARM_ROTATION_STEP = 0.005
    
    # Control modes
    MODE_POSITION = 'position'
    MODE_ORIENTATION = 'orientation'
    
    # Keyboard mappings
    LEFT_ARM_KEYS = {
        'x_minus': 'a', 'x_plus': 'd',
        'y_plus': 'z', 'y_minus': 'c',
        'z_plus': 'w', 'z_minus': 's'
    }
    RIGHT_ARM_KEYS = {
        'x_minus': 'j', 'x_plus': 'l',
        'y_plus': 'u', 'y_minus': 'o',
        'z_plus': 'i', 'z_minus': 'k'
    }
    GRIPPER_KEYS = {'left': 'v', 'right': 'b'}
    CONTROL_TOGGLE_KEY = 't'
    GRIPPER_MODE_TOGGLE_KEY = 'g'
    RECORD_KEY = 'r'
    SAVE_KEY = 'x'

    def __init__(
        self,
        env_cls: Type[RoboEvalEnv],
        action_mode: ActionMode,
        resolution: tuple[int, int] = (900, 1000),
        demo_directory: Optional[Union[str, Path]] = None,
        robot_cls: Optional[Type[Robot]] = None,
        config: Optional[dict] = None,
    ):
        """Initialize the keyboard teleoperation interface."""
        self.config = config
        self._width, self._height = resolution
        self._demo_recorder = DemoRecorder(demo_directory)

        # Set up environment
        keyboard_env_cls = self._create_keyboard_env(env_cls)
        self._env = keyboard_env_cls(
            render_mode="human", action_mode=action_mode, robot_cls=robot_cls
        )
        self._env.mojo.model.vis.global_.offwidth = self._width
        self._env.mojo.model.vis.global_.offheight = self._height
        self._env.reset()

        # Initialize state variables
        self._stats = KeyboardTeleopStats()
        self._stop_countdown = None
        self._current_action = np.zeros(self._env.action_space.shape)
        self._ik = self._env.inverse_kinematics
        
        # Set up keyboard listener
        self.pressed_keys = set()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

        # Control settings
        self.control_mode = self.MODE_POSITION
        self.toggle_gripper_mode = True  # True for toggle, False for hold
        
        # Set up grippers and arm state
        self._setup_grippers()
        self._init_arm_state()
        
        # Gripper state tracking
        self._gripper_toggle_state = {"left": 0.0, "right": 0.0}
        self._prev_key_state = {self.GRIPPER_KEYS["left"]: False, self.GRIPPER_KEYS["right"]: False}

    def _create_keyboard_env(self, env_cls: Type[RoboEvalEnv]) -> Type[RoboEvalEnv]:
        """Create a wrapper environment class with demo recording capabilities."""
        
        def get_demo_recorder():
            return self._demo_recorder

        class KeyboardTeleopRoboEvalEnv(env_cls, ABC):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._demo_recorder = get_demo_recorder()

            def step(self, action: ActType, fast: bool = True) -> tuple[Any, float, bool, bool, dict]:
                super().step(action, fast)
                timestep = ({}, self.reward, False, False, {})
                self._demo_recorder.add_timestep(timestep, action)
                return timestep

            @property
            def task_name(self) -> str:
                return self.__class__.__base__.__name__

        return KeyboardTeleopRoboEvalEnv

    def _setup_grippers(self):
        """Set up gripper references based on robot configuration."""
        grippers = self._env.robot.grippers

        if not grippers:
            raise ValueError("No grippers found in the robot configuration.")
        if len(grippers) > 2:
            raise ValueError("More than two grippers found. Only two are supported.")
            
        self.gripper_l = grippers.get(HandSide.LEFT)
        self.gripper_r = grippers.get(HandSide.RIGHT)
        
        if not (self.gripper_l or self.gripper_r):
            raise ValueError("No valid grippers found in the robot configuration.")

    def _init_arm_state(self):
        """Initialize arm positions and orientations."""
        self.left_arm_target = np.array(self.gripper_l.wrist_position) if self.gripper_l else np.zeros(3)
        self.right_arm_target = np.array(self.gripper_r.wrist_position) if self.gripper_r else np.zeros(3)

        self.left_arm_orientation = Quaternion(self.gripper_l.wrist_orientation) if self.gripper_l else Quaternion()
        self.right_arm_orientation = Quaternion(self.gripper_r.wrist_orientation) if self.gripper_r else Quaternion()

    def run(self, exit_event=None, on_running_event=None):
        """Start keyboard teleoperation session."""
        if exit_event is None:
            exit_event = threading.Event()
        if on_running_event is None:
            on_running_event = threading.Event()
            
        on_running_event.set()
        
        while not exit_event.is_set():
            self._handle_input()
            action = self._get_action()
            
            for _ in range(self.STEPS_COUNT_FACTOR):
                self._env.step(action, fast=True)
                
                # Handle recording termination
                if self._stop_countdown:
                    self._stop_countdown.step()
                    if self._stop_countdown.is_up:
                        self._stop_recording()
                elif self._env.reward > 0:
                    self._stop_countdown = Countdown(TERMINATION_STEPS)

            self._render_frame()

    def _handle_input(self):
        """Process special keyboard inputs."""
        if self.RECORD_KEY in self.pressed_keys:
            self._start_recording()
            print("Recording started.")
            
        if self.SAVE_KEY in self.pressed_keys and self._demo_recorder._recording:
            self._save_recording()
            print("Recording saved.")

    def _get_action(self) -> np.ndarray:
        """Get clipped action for environment."""
        action = self.get_next_action()
        return np.clip(action, self._env.action_space.low, self._env.action_space.high)
    
    def get_next_action(self) -> ActType:
        """Generate the next action based on keyboard input."""
        # Handle gripper controls
        grip_left, grip_right = self._process_gripper_input()
        
        # Handle arm controls based on control mode
        if self.control_mode == self.MODE_POSITION:
            self._handle_position_control()
        else:
            self._handle_orientation_control()
        
        # Calculate IK solution
        control = self._calculate_control(grip_left, grip_right)
        return control

    def _process_gripper_input(self):
        """Process gripper input and return grip states."""
        left_key = self.GRIPPER_KEYS["left"]
        right_key = self.GRIPPER_KEYS["right"]
        
        # Toggle-on-press logic for grippers
        if self.toggle_gripper_mode:
            # LEFT GRIPPER
            if left_key in self.pressed_keys and not self._prev_key_state[left_key]:
                self._gripper_toggle_state["left"] = 1.0 - self._gripper_toggle_state["left"]
            self._prev_key_state[left_key] = left_key in self.pressed_keys

            # RIGHT GRIPPER
            if right_key in self.pressed_keys and not self._prev_key_state[right_key]:
                self._gripper_toggle_state["right"] = 1.0 - self._gripper_toggle_state["right"]
            self._prev_key_state[right_key] = right_key in self.pressed_keys

            grip_left = self._gripper_toggle_state["left"]
            grip_right = self._gripper_toggle_state["right"]
        else:
            grip_left = 1.0 if left_key in self.pressed_keys else 0.0
            grip_right = 1.0 if right_key in self.pressed_keys else 0.0
            
        return grip_left, grip_right

    def _handle_position_control(self):
        """Handle position control mode for arm movement."""
        step = self.ARM_MOVEMENT_STEP
        
        # Right arm position control
        if self.gripper_r:
            keys = self.RIGHT_ARM_KEYS
            if keys['x_minus'] in self.pressed_keys: self.right_arm_target[0] -= step
            if keys['x_plus'] in self.pressed_keys: self.right_arm_target[0] += step
            if keys['y_plus'] in self.pressed_keys: self.right_arm_target[1] += step
            if keys['y_minus'] in self.pressed_keys: self.right_arm_target[1] -= step
            if keys['z_plus'] in self.pressed_keys: self.right_arm_target[2] += step
            if keys['z_minus'] in self.pressed_keys: self.right_arm_target[2] -= step
        
        # Left arm position control
        if self.gripper_l:
            keys = self.LEFT_ARM_KEYS
            if keys['x_minus'] in self.pressed_keys: self.left_arm_target[0] -= step
            if keys['x_plus'] in self.pressed_keys: self.left_arm_target[0] += step
            if keys['y_plus'] in self.pressed_keys: self.left_arm_target[1] += step
            if keys['y_minus'] in self.pressed_keys: self.left_arm_target[1] -= step
            if keys['z_plus'] in self.pressed_keys: self.left_arm_target[2] += step
            if keys['z_minus'] in self.pressed_keys: self.left_arm_target[2] -= step

    def _handle_orientation_control(self):
        """Handle orientation control mode for arm rotation."""
        step = self.ARM_ROTATION_STEP
        
        # Right arm orientation control
        if self.gripper_r:
            keys = self.RIGHT_ARM_KEYS
            self.right_arm_orientation = self._apply_rotation(
                self.right_arm_orientation,
                keys['x_minus'] in self.pressed_keys,
                keys['x_plus'] in self.pressed_keys,
                keys['y_plus'] in self.pressed_keys,
                keys['y_minus'] in self.pressed_keys,
                keys['z_plus'] in self.pressed_keys,
                keys['z_minus'] in self.pressed_keys,
                step
            )

        # Left arm orientation control
        if self.gripper_l:
            keys = self.LEFT_ARM_KEYS
            self.left_arm_orientation = self._apply_rotation(
                self.left_arm_orientation,
                keys['x_minus'] in self.pressed_keys,
                keys['x_plus'] in self.pressed_keys,
                keys['y_plus'] in self.pressed_keys,
                keys['y_minus'] in self.pressed_keys,
                keys['z_plus'] in self.pressed_keys,
                keys['z_minus'] in self.pressed_keys,
                step
            )

    def _apply_rotation(self, orientation, x_minus, x_plus, y_plus, y_minus, z_plus, z_minus, step):
        """Apply rotation to an orientation quaternion based on key presses."""
        if x_minus:
            orientation *= Quaternion(axis=[0, 1, 0], angle=step)
        if x_plus:
            orientation *= Quaternion(axis=[0, 1, 0], angle=-step)
        if y_plus:
            orientation *= Quaternion(axis=[1, 0, 0], angle=step)
        if y_minus:
            orientation *= Quaternion(axis=[1, 0, 0], angle=-step)
        if z_plus:
            orientation *= Quaternion(axis=[0, 0, 1], angle=step)
        if z_minus:
            orientation *= Quaternion(axis=[0, 0, 1], angle=-step)
            
        return orientation.normalised

    def _calculate_control(self, grip_left, grip_right):
        """Calculate control vector for the robot."""
        # Get pelvis pose
        pelvis = self._env.robot.pelvis
        pelvis_pose = Pose(pelvis.get_position(), Quaternion(pelvis.get_quaternion()))
        
        # Initialize control vector
        control = np.zeros_like(self._env.action_space.low)
        
        # Get arm joint positions
        floating_base = self._env.robot.floating_base
        start_index = floating_base.dof_amount
        end_index = start_index + len(self._env.robot.limb_actuators)
        arms_qpos = np.array(self._env.robot.qpos_actuated[start_index:end_index])
        
        # Split arm positions for IK
        num_arms = len(self._env.robot._arms)
        qpos_arms = np.split(arms_qpos, num_arms)
        
        # Create target poses
        target_poses = []
        if self.gripper_l is not None:
            target_pose_left = Pose(pelvis_pose.position + self.left_arm_target, self.left_arm_orientation)
            target_poses.append(target_pose_left)
        if self.gripper_r is not None:
            target_pose_right = Pose(pelvis_pose.position + self.right_arm_target, self.right_arm_orientation)
            target_poses.append(target_pose_right)
        
        # Solve IK
        solution = self._ik.solve(
            root_pose=pelvis_pose,
            qpos_arms=qpos_arms,
            target_poses=target_poses,
        )
        control[start_index:end_index] = solution
        
        # Apply gripper control
        if self.gripper_l is not None and self.gripper_r is not None:
            control[-2] = 1 if grip_left else 0
            control[-1] = 1 if grip_right else 0
        else:
            control[-1] = 1 if grip_left else 0
            
        return control

    def on_press(self, key):
        """Handle key press events."""
        try:
            self.pressed_keys.add(key.char)
            if key.char == self.CONTROL_TOGGLE_KEY:
                self.control_mode = self.MODE_ORIENTATION if self.control_mode == self.MODE_POSITION else self.MODE_POSITION
                print(f"Control mode: {self.control_mode}")
            if key.char == self.GRIPPER_MODE_TOGGLE_KEY:
                self.toggle_gripper_mode = not self.toggle_gripper_mode
                mode = "Autoclose" if self.toggle_gripper_mode else "Hold-to-Close"
                print(f"Gripper Mode: {mode}")
        except AttributeError:
            pass

    def on_release(self, key):
        """Handle key release events."""
        try:
            self.pressed_keys.discard(key.char)
        except AttributeError:
            pass

    def _render_frame(self):
        """Update display with current state."""
        self._update_stats()
        self._env.render()

    def _start_recording(self):
        """Start recording a demonstration."""
        self._stop_recording()
        self._stop_countdown = None
        self._env.reset()
        self._init_arm_state()
        self._gripper_toggle_state = {"left": 0.0, "right": 0.0}
        self._prev_key_state = {self.GRIPPER_KEYS["left"]: False, self.GRIPPER_KEYS["right"]: False}
        self._demo_recorder.record(self._env, lightweight_demo=True)
        
    def _stop_recording(self):
        """Stop recording if currently recording."""
        if self._demo_recorder.is_recording:
            self._demo_recorder.stop()

    def _save_recording(self):
        """Save the current recording."""
        self._stop_recording()
        if self._demo_recorder.save_demo():
            self._stats.demos_counter += 1

    def _update_stats(self):
        """Update teleoperation statistics."""
        self._stats.is_recoding = self._demo_recorder.is_recording
        self._stats.time = self._env.mojo.data.time
        self._stats.reward = self._env.reward
