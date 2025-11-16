"""An example of demo collection and replay with action mode conversion."""
import copy
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from roboeval.action_modes import JointPositionActionMode
from roboeval.demonstrations.demo import Demo
from roboeval.demonstrations.demo_converter import DemoConverter
from roboeval.demonstrations.demo_recorder import DemoRecorder
from roboeval.envs.lift_pot import LiftPot
from roboeval.roboeval_env import RoboEvalEnv
from roboeval.robots.configs.panda import BimanualPanda

# Configuration
render = True
cam = "head"
cam_key = f"rgb_{cam}"
absolute = False
joint = False
robot = 'bimanual_panda'

# Robot class mapping
robot_cls_from_string = {
    "default": RoboEvalEnv.DEFAULT_ROBOT,
    "bimanual_panda": BimanualPanda,
}
robot_cls = robot_cls_from_string[robot]

# Create environment for recording
env = LiftPot(
    action_mode=JointPositionActionMode(floating_base=True, absolute=True, floating_dofs=[]),
    render_mode="human" if render else None,
    robot_cls=robot_cls,
)


def update_plots(axs, requested, expected, actual, xlim=None, ylim=None):
    """Update plots with requested, expected, and actual trajectories."""
    for i, (r, e, a) in enumerate(zip(requested, expected, actual)):
        axs[i].clear()  # Clear previous plot
        axs[i].plot(r, label="Request")
        axs[i].plot(e, label="Expected")
        axs[i].plot(a, label="Actual")
        axs[i].set_title(f"Variable {i}")
        axs[i].legend()
        if xlim is not None:
            axs[i].set_xlim(xlim)
        if ylim is not None:
            axs[i].set_ylim(ylim)
    plt.tight_layout()


def init_subplots(n):
    """Initialize subplots for trajectory visualization."""
    fig = plt.figure(figsize=(8, 2 * n))
    axs = [fig.add_subplot(n, 1, i + 1) for i in range(n)]
    return fig, axs


# Main demo collection and replay
with tempfile.TemporaryDirectory() as temp_dir:
    demo_recorder = DemoRecorder(temp_dir)

    # Demo recording parameters
    amplitude = 0.2
    frequency = 0.02
    episode_length = 500

    # Record the demo
    env.reset()
    demo_recorder.record(env)

    expected = []
    
    # Set initial action based on robot type
    if robot == 'bimanual_panda':
        action = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0, 0])
    else:
        action = np.zeros_like(env.action_space.sample())

    # Recording loop
    for i in range(episode_length):
        # Create sinusoidal motion for demo
        action[0] = amplitude * np.sin(frequency * i)
        action[7] = amplitude * -np.sin(frequency * i)
        
        timestep = env.step(action)
        demo_recorder.add_timestep(copy.deepcopy(timestep), copy.deepcopy(action))
        
        # Record both joint and EE positions for later comparison
        # We'll decide which to use based on the replay mode
        if joint:
            expected.append(env.robot.qpos_actuated)
        else:
            # Record EE positions for EE mode comparison
            ee_positions = []
            for wrist_side, wrist_site in env.robot._wrist_sites.items():
                position = wrist_site.get_position()
                quaternion = wrist_site.get_quaternion()
                # Convert quaternion to euler angles for consistent representation
                ee_rot_euler = R.from_quat(quaternion[[1, 2, 3, 0]]).as_euler('xyz')  # Convert wxyz to xyzw
                ee_positions.extend([*position, *ee_rot_euler])
            expected.append(np.array(ee_positions))
            
        if render:
            env.render()
    
    demo_recorder.stop()
    env.close()

    # Save and load demo
    filepath = demo_recorder.save_demo()
    demo = Demo.from_safetensors(filepath)

    # Create environment for replay with different action mode
    env = LiftPot(
        action_mode=JointPositionActionMode(floating_base=True, absolute=absolute, ee=not joint, floating_dofs=[]),
        render_mode="human" if render else None,
        robot_cls=robot_cls,
    )
    env.reset(seed=demo.seed)

    # Convert demo to different action mode if needed
    request = []
    actual = []
    if not joint and not absolute:
        demo = DemoConverter.joint_absolute_to_ee_delta(demo)
    elif not joint:
        demo = DemoConverter.joint_to_ee(demo)
    elif not absolute:
        demo = DemoConverter.absolute_to_delta(demo)

    env.reset(seed=demo.seed)

    # Store previous values for delta modes to track comparable quantities
    # This ensures we compare like with like: joint positions with joint positions,
    # or EE positions with EE positions, regardless of the action mode
    prev_joint_pos = env.robot.qpos_actuated.copy() if not absolute and joint else None
    prev_ee_pos = None
    if not joint:
        # Get initial EE positions for EE modes
        ee_positions = []
        for wrist_side, wrist_site in env.robot._wrist_sites.items():
            position = wrist_site.get_position()
            quaternion = wrist_site.get_quaternion()
            # Convert quaternion to euler angles for consistent representation
            ee_rot_euler = R.from_quat(quaternion[[1, 2, 3, 0]]).as_euler('xyz')  # Convert wxyz to xyzw
            ee_positions.extend([*position, *ee_rot_euler])
        prev_ee_pos = np.array(ee_positions) if not absolute else None

    # Replay the demo
    for timestep in demo.timesteps:
        action = timestep.executed_action
        obs, reward, termination, truncation, info = env.step(action)
        
        # Store comparable quantities based on action mode
        if joint:
            # Joint mode: compare joint positions
            if absolute:
                request.append(action)  # Action is absolute joint positions
            else:
                # Delta joint mode: integrate deltas to get target positions
                target_joint_pos = prev_joint_pos + action
                request.append(target_joint_pos)
                prev_joint_pos = target_joint_pos
            actual.append(env.robot.qpos_actuated)
        else:
            # EE mode: compare end-effector positions
            # Extract only the EE pose part (exclude gripper controls at the end)
            num_grippers = len(env.robot.grippers)
            ee_action = action[:-num_grippers] if num_grippers > 0 else action
            
            if absolute:
                request.append(ee_action)  # Action is absolute EE positions
            else:
                # Delta EE mode: integrate deltas to get target EE positions
                target_ee_pos = prev_ee_pos + ee_action
                request.append(target_ee_pos)
                prev_ee_pos = target_ee_pos
            
            # Get actual EE positions
            ee_positions = []
            for wrist_side, wrist_site in env.robot._wrist_sites.items():
                position = wrist_site.get_position()
                quaternion = wrist_site.get_quaternion()
                # Convert quaternion to euler angles for consistent representation
                ee_rot_euler = R.from_quat(quaternion[[1, 2, 3, 0]]).as_euler('xyz')  # Convert wxyz to xyzw
                ee_positions.extend([*position, *ee_rot_euler])
            actual.append(np.array(ee_positions))
            
        if render:
            env.render()

    # Visualize results
    request = np.array(request).T
    expected = np.array(expected).T
    actual = np.array(actual).T
    
    # Create descriptive labels based on action mode
    if joint:
        if absolute:
            request_label = "Requested Joint Positions"
            actual_label = "Actual Joint Positions"
        else:
            request_label = "Target Joint Positions (from deltas)"
            actual_label = "Actual Joint Positions"
        expected_label = "Expected Joint Positions (from recording)"
    else:
        if absolute:
            request_label = "Requested EE Positions"
            actual_label = "Actual EE Positions"
        else:
            request_label = "Target EE Positions (from deltas)"
            actual_label = "Actual EE Positions"
        expected_label = "Expected EE Positions (from recording)"
    
    fig, axs = init_subplots(len(env.action_space.sample()))
    amplitude = abs(actual).max()
    
    # Update plot function to use custom labels
    for i, (r, e, a) in enumerate(zip(request, expected, actual)):
        axs[i].clear()
        axs[i].plot(r, label=request_label)
        axs[i].plot(e, label=expected_label)
        axs[i].plot(a, label=actual_label)
        axs[i].set_title(f"Variable {i}")
        axs[i].legend()
        axs[i].set_ylim((-amplitude * 1.1, amplitude * 1.1))
    
    plt.tight_layout()
    plt.show()