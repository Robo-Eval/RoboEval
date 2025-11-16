# from base import InputMode

import threading
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type, Any, Union

import numpy as np

from gymnasium.core import ActType

from roboeval.action_modes import ActionMode
from roboeval.robots.robot import Robot
from roboeval.demonstrations.demo import TERMINATION_STEPS
from roboeval.demonstrations.demo_recorder import DemoRecorder
from roboeval.roboeval_env import RoboEvalEnv

from roboeval.ik.base_ik import Pose

from pyquaternion import Quaternion

# Try to import xr, but provide fallback if not available (GLIBC compatibility)
try:
    from xr import FrameState, Posef
except (ImportError, OSError):
    # Fallback: Posef is only used as a placeholder (never actually used in code)
    # FrameState is not used in this file
    class Posef:
        """Fallback Posef class for when xr is not available."""
        def __init__(self):
            pass
    
    class FrameState:
        """Fallback FrameState class for when xr is not available."""
        pass

from oculus_reader import OculusReader

import time

from scipy.spatial.transform import Rotation as R

from roboeval.const import (
    HandSide,
)

from roboeval.ik.base_ik import GenericUpperBodyIK




@dataclass
class OculusTeleopStats:
    """Oculus statistics."""

    is_recoding: bool = False
    time: float = 0
    reward: float = 0
    demos_counter: int = 0

def vec_to_reorder_mat(vec):
    X = np.zeros((len(vec), len(vec)))
    for i in range(X.shape[0]):
        ind = int(abs(vec[i])) - 1
        X[i, ind] = np.sign(vec[i])
    return X

def run_threaded_command(command, args=(), daemon=True):
    thread = threading.Thread(target=command, args=args, daemon=daemon)
    thread.start()

    return thread

class OculusControlProfile:
    def __init__(self,
                right_controller_as_main: bool = True,
                max_lin_vel: float = 1,
                max_rot_vel: float = 1,
                max_gripper_vel: float = 1,
                spatial_coeff: float = 1,
                pos_action_gain: float = 0.005,
                rot_action_gain: float = 0.005,
                gripper_action_gain: float = 3,
                rmat_reorder: list = [-2, -1, -3, 4],
                movement_threshold: float = 0.1
                ):
        self.oculus_reader = OculusReader()
        self.vr_to_global_mat = np.eye(4)
        self.max_lin_vel = max_lin_vel
        self.max_rot_vel = max_rot_vel
        self.max_gripper_vel = max_gripper_vel
        self.spatial_coeff = spatial_coeff
        self.pos_action_gain = pos_action_gain
        self.rot_action_gain = rot_action_gain
        self.gripper_action_gain = gripper_action_gain
        self.movement_threshold = movement_threshold
        self.global_to_env_mat = vec_to_reorder_mat(rmat_reorder)
        self.reset_orientation = True
        self.controller_id = "r" if right_controller_as_main else "l"
        self.toggle_gripper_mode = True  # Enable autoclose mode
        self._gripper_toggle_state = {"left": 0.0, "right": 0.0}  # 0.0 = open, 1.0 = closed
        self._prev_trigger_state = {"left": 0.0, "right": 0.0}

        self._init_state()

        # Start State Listening Thread #
        run_threaded_command(self._update_internal_state)
    
    def _init_state(self):
        self._state = {
            "poses": {},
            "buttons": {"A": False, "B": False, "X": False, "Y": False},
            "movement_enabled": False,
            "controller_on": True,
        }
        self.update_sensor = True
        self.reset_origin = True
        self.robot_origin_base = None
        self.robot_origin_l = None
        self.robot_origin_r = None
        self.vr_origin_r = None
        self.vr_origin_l = None
        self.vr_state = None
        self.record = False
    
    def reset_state(self):
        self.update_sensor = True
        self.reset_origin = True
        self.robot_origin_base = None
        self.robot_origin_l = None
        self.robot_origin_r = None
        self.vr_origin_r = None
        self.vr_origin_l = None
        self.vr_state = None
        self.record = False

        self._gripper_toggle_state = {"left": 0.0, "right": 0.0}
        self._prev_trigger_state = {"left": 0.0, "right": 0.0}
    
    def _update_internal_state(self, num_wait_sec=5, hz=50):
        last_read_time = time.time()
        while True:
            # Regulate Read Frequency #
            time.sleep(1 / hz)

            # Read Controller
            time_since_read = time.time() - last_read_time
            poses, buttons = self.oculus_reader.get_transformations_and_buttons()
            self._state["controller_on"] = time_since_read < num_wait_sec
            if poses == {}:
                continue

            # Check if either controller is missing
            if "l" not in poses or "r" not in poses:
                print("Controllers not all active, active controllers: " + str(poses.keys()))
                continue

            # Determine Control Pipeline #
            toggled = self._state["movement_enabled"] != buttons[self.controller_id.upper() + "G"]
            self.update_sensor = self.update_sensor or buttons[self.controller_id.upper() + "G"]
            self.reset_orientation = self.reset_orientation or buttons[self.controller_id.upper() + "J"]
            self.reset_origin = self.reset_origin or toggled

            # Save Info #
            self._state["poses"] = poses
            self._state["buttons"] = buttons
            self._state["movement_enabled"] = buttons[self.controller_id.upper() + "G"]
            self._state["controller_on"] = True
            last_read_time = time.time()

            # Check whether to record
            if buttons["A"]:
                self.record = True
            if buttons["B"]:
                self.record = False

            if self._state["buttons"]["Y"]:  # Use "Y" button to toggle mode
                self.toggle_gripper_mode = not self.toggle_gripper_mode
                print("Toggle Gripper Mode:", "Autoclose" if self.toggle_gripper_mode else "Hold-to-Close")
                time.sleep(0.5)  # Prevent accidental double toggle

            # Update Definition Of "Forward" #
            stop_updating = self._state["buttons"][self.controller_id.upper() + "J"] or self._state["movement_enabled"]
            if self.reset_orientation:
                rot_mat = np.asarray(self._state["poses"][self.controller_id])
                if stop_updating:
                    self.reset_orientation = False
                # try to invert the rotation matrix, if not possible, then just use the identity matrix
                try:
                    rot_mat = np.linalg.inv(rot_mat)
                except:
                    print(f"exception for rot mat: {rot_mat}")
                    rot_mat = np.eye(4)
                    self.reset_orientation = True
                self.vr_to_global_mat = rot_mat

    def _process_reading(self, zero_out_left=[], zero_out_right=[]):

        # Process right controller
        rot_mat_r = np.asarray(self._state["poses"]['r'])
        rot_mat_r = self.global_to_env_mat @ self.vr_to_global_mat @ rot_mat_r
        vr_pos_r = self.spatial_coeff * rot_mat_r[:3, 3]

        # Convert rotation matrix to euler angles (roll, pitch, yaw)
        euler_angles = R.from_matrix(rot_mat_r[:3, :3]).as_euler('xyz')
        
        # Swap pitch and yaw (indices 1 and 2)
        swapped_euler = euler_angles.copy()
        swapped_euler[0], swapped_euler[1], swapped_euler[2] = euler_angles[2]+np.pi/2, -euler_angles[1], -euler_angles[0]

        # Apply zeroing out
        for i in zero_out_left:
            swapped_euler[i] = 0.0
        
        # Convert back to rotation matrix
        rot_mat_r[:3, :3] = R.from_euler('xyz', swapped_euler).as_matrix()

        vr_quat_r = R.from_matrix(rot_mat_r[:3, :3]).as_quat()

        # Process left controller
        rot_mat_l = np.asarray(self._state["poses"]['l'])
        rot_mat_l = self.global_to_env_mat @ self.vr_to_global_mat @ rot_mat_l
        vr_pos_l = self.spatial_coeff * rot_mat_l[:3, 3]

        euler_angles = R.from_matrix(rot_mat_l[:3, :3]).as_euler('xyz')
        
        # Swap pitch and yaw (indices 1 and 2)
        swapped_euler = euler_angles.copy()
        swapped_euler[0], swapped_euler[1], swapped_euler[2] = euler_angles[2]+np.pi/2, -euler_angles[1], -euler_angles[0]

        # Apply zeroing out
        for i in zero_out_right:
            swapped_euler[i] = 0.0

        # Convert back to rotation matrix
        rot_mat_l[:3, :3] = R.from_euler('xyz', swapped_euler).as_matrix()

        vr_quat_l = R.from_matrix(rot_mat_l[:3, :3]).as_quat()

        # Raw trigger values
        raw_trigger_r = self._state["buttons"]["rightTrig"][0]
        raw_trigger_l = self._state["buttons"]["leftTrig"][0]

        # Rising edge detection for right trigger
        if self.toggle_gripper_mode:
            if raw_trigger_r > 0.8 and self._prev_trigger_state["right"] <= 0.8:
                self._gripper_toggle_state["right"] = 1.0 - self._gripper_toggle_state["right"]
            if raw_trigger_l > 0.8 and self._prev_trigger_state["left"] <= 0.8:
                self._gripper_toggle_state["left"] = 1.0 - self._gripper_toggle_state["left"]

            # Save trigger state
            vr_gripper_r = self._gripper_toggle_state["right"]
            vr_gripper_l = self._gripper_toggle_state["left"]
        else:
            vr_gripper_r = raw_trigger_r
            vr_gripper_l = raw_trigger_l

        # Update previous trigger state
        self._prev_trigger_state["right"] = raw_trigger_r
        self._prev_trigger_state["left"] = raw_trigger_l


        # Make it a quaternion object
        vr_quat_l = Quaternion(vr_quat_l)
        vr_quat_r = Quaternion(vr_quat_r)

        # Check whether joystick is pressed
        rotate, up_down = self._state["buttons"]["rightJS"]
        left_right, forward_backward = self._state["buttons"]["leftJS"]

        # Get the delta position and rotation via multiplication with gains
        delta_pos = np.array([0, 0, 0], dtype=float)
        delta_rot = np.array([0, 0, 0], dtype=float)

        # Update the state
        if rotate > self.movement_threshold or -rotate > self.movement_threshold:
            delta_rot[2] = rotate * self.rot_action_gain

        if up_down > self.movement_threshold or -up_down > self.movement_threshold:
            delta_pos[2] = up_down * self.pos_action_gain

        if left_right > self.movement_threshold or -left_right > self.movement_threshold:
            delta_pos[0] = left_right * self.pos_action_gain

        if forward_backward > self.movement_threshold or -forward_backward > self.movement_threshold:
            delta_pos[1] = forward_backward * self.pos_action_gain

        if not self.reset_origin and any([
                abs(rotate) > self.movement_threshold,
                abs(up_down) > self.movement_threshold,
                abs(left_right) > self.movement_threshold,
                abs(forward_backward) > self.movement_threshold
            ]):
            self.reset_orientation = True
            self.reset_origin = True


        self.vr_state = {"pos_r": vr_pos_r, "quat_r": vr_quat_r, "gripper_r": vr_gripper_r, 
                         "pos_l": vr_pos_l, "quat_l": vr_quat_l, "gripper_l": vr_gripper_l,
                         'rotate': rotate, 'up_down': up_down, 'left_right': left_right, 
                        'forward_backward': forward_backward,
                        'delta_pos': delta_pos, 'delta_rot': delta_rot}
        

    def _limit_velocity(self, lin_vel, rot_vel, gripper_vel):
        """Scales down the linear and angular magnitudes of the action"""
        lin_vel_norm = np.linalg.norm(lin_vel)
        rot_vel_norm = np.linalg.norm(rot_vel)
        gripper_vel_norm = np.linalg.norm(gripper_vel)
        if lin_vel_norm > self.max_lin_vel:
            lin_vel = lin_vel * self.max_lin_vel / lin_vel_norm
        if rot_vel_norm > self.max_rot_vel:
            rot_vel = rot_vel * self.max_rot_vel / rot_vel_norm
        if gripper_vel_norm > self.max_gripper_vel:
            gripper_vel = gripper_vel * self.max_gripper_vel / gripper_vel_norm
        return lin_vel, rot_vel, gripper_vel

class OculusFixedBase:
    """Control profile for arbitrary robot in fixed base mode.

    Notes:
        - Use left/right controllers to control position of end effector
        - Use left/right trigger to control gripper
    """

    POSITION_SMOOTHING = 0.75
    ROTATION_SMOOTHING = 0.75


    def __init__(self, env: RoboEvalEnv, ik: GenericUpperBodyIK, config=None):
        """Init."""
        self._env = env

        self._sync_position = True
        self._sync_rotation = True
        self._ik = ik
        
        # Handle both dict and object configs
        if config:
            if isinstance(config, dict):
                self.zero_out_left = config.get('zero_out_axis_left', [])
                self.zero_out_right = config.get('zero_out_axis_right', [])
            else:
                self.zero_out_left = getattr(config, 'zero_out_axis_left', [])
                self.zero_out_right = getattr(config, 'zero_out_axis_right', [])
        else:
            self.zero_out_left = []
            self.zero_out_right = []

        self._setup_grippers()

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


    def get_next_action(self, context, steps, offset) -> ActType:
        if context.update_sensor:
            context._process_reading(zero_out_left=self.zero_out_left, zero_out_right=self.zero_out_right)
            context.update_sensor = False

        data = context.vr_state

        # Current state of the robot
        pelvis = self._env.robot.pelvis
        grippers = self._env.robot.grippers
        gripper_l = grippers[HandSide.LEFT]
        gripper_r = grippers[HandSide.RIGHT]
        pelvis_pose = Pose(pelvis.get_position(), Quaternion(pelvis.get_quaternion()))


        # Reset origin on release
        if context.reset_origin:
            context.robot_origin_base = {"pos": pelvis_pose.position, "quat": pelvis_pose.orientation}
            context.robot_origin_l = {"pos": gripper_l.wrist_position, "quat": gripper_l.wrist_orientation}
            context.robot_origin_r = {"pos": gripper_r.wrist_position, "quat": gripper_r.wrist_orientation}
            context.vr_origin_r = {"pos": context.vr_state["pos_r"], "quat": context.vr_state["quat_r"]}
            context.vr_origin_l = {"pos": context.vr_state["pos_l"], "quat": context.vr_state["quat_l"]}

            context.reset_origin = False
        
        # Current state of the VR controllers
        l_pos = data['pos_l']
        l_quat = data['quat_l']
        r_pos = data['pos_r']
        r_quat = data['quat_r']
        l_gripper = data['gripper_l']
        r_gripper = data['gripper_r']

        # Calculate the positional action

        target_pos_l_offset = l_pos - context.vr_origin_l["pos"]
        target_pos_r_offset = r_pos - context.vr_origin_r["pos"]

        # Calculate the rotational action represented as a quaternion

        target_rot_l_offset = l_quat * context.vr_origin_l["quat"].inverse
        target_rot_r_offset = r_quat * context.vr_origin_r["quat"].inverse

        delta_pos_l = target_pos_l_offset  # Relative to VR controller only
        delta_pos_r = target_pos_r_offset

        delta_rot_l = target_rot_l_offset
        delta_rot_r = target_rot_r_offset

        # Calculate the desired pose
        target_pos_l = context.robot_origin_l["pos"]+ delta_pos_l*self.POSITION_SMOOTHING 
        target_pos_r = context.robot_origin_r["pos"] + delta_pos_r*self.POSITION_SMOOTHING

        target_rot_l = Quaternion.slerp(context.robot_origin_l["quat"], context.robot_origin_l["quat"] * delta_rot_l, self.ROTATION_SMOOTHING)
        target_rot_r = Quaternion.slerp(context.robot_origin_r["quat"], context.robot_origin_r["quat"] * delta_rot_r, self.ROTATION_SMOOTHING)

        control = np.zeros_like(self._env.action_space.low)

        delta_pos = data['delta_pos']
        delta_ypr = data['delta_rot']

        floating_base = self._env.robot.floating_base

        base_control = []
        if floating_base:
            for delta, actuator in zip(delta_pos, floating_base.position_actuators):
                if actuator:
                    base_control.append(delta)
            for delta, actuator in zip(delta_ypr, floating_base.rotation_actuators):
                if actuator:
                    base_control.append(delta)
            control[: len(base_control)] = base_control

        # Control arms
        start_index = floating_base.dof_amount
        end_index = start_index + len(self._env.robot.limb_actuators)

        arms_qpos = np.array(self._env.robot.qpos_actuated[start_index:end_index])
        # Split arm positions for IK
        num_arms = len(self._env.robot._arms)
        qpos_arms = np.split(arms_qpos, num_arms)

        # Create target poses
        target_poses = []
        if self.gripper_l is not None:
            target_pose_left = Pose(target_pos_l, target_rot_l)
            target_poses.append(target_pose_left)
        if self.gripper_r is not None:
            target_pose_right = Pose(target_pos_r, target_rot_r)
            target_poses.append(target_pose_right)

        pelvis_pose_original = Pose(context.robot_origin_base["pos"], context.robot_origin_base["quat"])

        solution = self._ik.solve(
            root_pose=pelvis_pose_original,
            qpos_arms=qpos_arms,
            target_poses=target_poses,
        )
        control[start_index:end_index] = solution

        # Control grippers
        control[-2] = np.clip(np.round(l_gripper), 0, 1)
        control[-1] = np.clip(np.round(r_gripper), 0, 1)
        return control
    
    def reset(self):
        pass
    
class Countdown:
    """Countdown timer."""

    def __init__(self, delay: int):
        self._delay = delay

    def step(self):
        if self._delay > 0:
            self._delay -= 1

    @property
    def is_up(self) -> bool:
        return self._delay <= 0

class OculusTeleop:

    STEPS_COUNT_FACTOR = 3

    def __init__(
        self,
        env_cls: Type[RoboEvalEnv],
        action_mode: ActionMode,
        resolution: tuple[int, int] = (900, 1000),
        demo_directory: Optional[Union[str, Path]] = None,
        robot_cls: Optional[Type[Robot]] = None,

        # Optional dictionary with config parameters
        config: Optional[dict] = None,
    ):
        self.config = config
        self._oculus_reader = OculusControlProfile()

        self._demo_recorder: DemoRecorder = DemoRecorder(demo_directory)

        vr_env_cls = self._oculus_env(env_cls)

        self._env = vr_env_cls(action_mode=action_mode, render_mode="human", robot_cls=robot_cls)

        self._env.reset()

        self._stats = OculusTeleopStats()

        self._stop_countdown: Optional[Countdown] = None

        self._ik = self._env.inverse_kinematics

        self._control_profile = self.get_control_profile(self._env)

        self._space_offset = Posef()

        

    def _oculus_env(self, env_cls: Type[RoboEvalEnv]) -> Type[RoboEvalEnv]:
        """Add VR controllers to standard mujoco environment."""

        def get_demo_recorder():
            return self._demo_recorder

        class VrRoboEvalEnv(env_cls, ABC):
            """RoboEvalEnv with VR controllers."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._demo_recorder = get_demo_recorder()
                
            def step(
                self, action: ActType, fast: bool = True
            ) -> tuple[Any, float, bool, bool, dict]:
                super().step(action, fast)
                timestep = ({}, self.reward, False, False, {})
                self._demo_recorder.add_timestep(timestep, action)
                return timestep
            
            @property
            def task_name(self) -> str:
                return self.__class__.__base__.__name__
            
        return VrRoboEvalEnv

    def get_control_profile(self, env: RoboEvalEnv) -> OculusFixedBase:
        return OculusFixedBase(env, self._ik, self.config)

    def _predict_steps_count(self, fps: float) -> int:
        # Convert FPS to frame duration (seconds) and divide by 2
        refresh_duration = (1 / fps) / 2
        return (
            int(round(refresh_duration / self._env.mojo.physics.timestep()))
            * self.STEPS_COUNT_FACTOR
        )
    
    def _get_action(self, steps_count: int) -> np.ndarray:
        action = self._control_profile.get_next_action(self._oculus_reader, steps_count, self._space_offset)
        action = np.clip(
            action, self._env.action_space.low, self._env.action_space.high
        )
        return action

    def run(
            self,
    ):
        # Render initial frame so window appears immediately
        print("\nWaiting for Oculus Quest controller data...")
        print("If you don't see a visualization window:")
        print("1. Put on your Oculus Quest headset")
        print("2. Allow USB debugging if prompted")
        print("3. Make sure controllers are turned on and tracked\n")
        
        self._render_frame()
        
        poses_received = False
        while True:
            data = self._oculus_reader._state
            if not data['poses']:
                # Render even when waiting for poses so window stays visible
                self._render_frame()
                if not poses_received:
                    import time
                    time.sleep(0.1)  # Brief sleep to avoid busy-waiting
                continue
            
            if not poses_received:
                print("✓ Oculus Quest controller data received! VR teleoperation active.\n")
                poses_received = True

            self._handle_input(self._oculus_reader)

            # steps_count = self._predict_steps_count(self._oculus_reader.oculus_reader.fps_counter.getAndPrintFPS())
            steps_count = 3
            action = self._get_action(steps_count)
            for _ in range(steps_count):
                self._env.step(action, fast=True)
                if self._stop_countdown:
                    self._stop_countdown.step()
                    if self._stop_countdown.is_up:
                        self._stop_recording()
                elif self._env.reward > 0:
                    self._stop_countdown = Countdown(TERMINATION_STEPS)
            self._render_frame()
    
    def _handle_input(self, context):
        # Control demo recoding
        if context._state["buttons"]["A"]:
            self._start_recording()
            print("Recording started.")

        elif context._state["buttons"]["B"]:
            self._save_recording()
            print("Recording saved.")

    def _render_frame(self):
        self._update_stats()
        self._env.render()
        pass

    def _start_recording(self):
        self._stop_recording()
        self._stop_countdown = None
        self._env.reset()
        self._control_profile.reset()
        self._oculus_reader.reset_state()
        self._demo_recorder.record(self._env, lightweight_demo=True)

    def _stop_recording(self):
        if self._demo_recorder.is_recording:
            self._demo_recorder.stop()
            self._stop_countdown = None

    def _save_recording(self):
        self._stop_recording()
        if self._demo_recorder.save_demo():
            self._stats.demos_counter += 1

    def _update_stats(self):
        self._stats.is_recoding = self._demo_recorder.is_recording
        self._stats.time = self._env.mojo.data.time
        self._stats.reward = self._env.reward