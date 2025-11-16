"""Load demonstrations from dataset, convert to different action modes, and replay.

This script combines data loading from 1_data_replay.py with conversion methods
from 2_convert_and_replay.py to demonstrate different action mode conversions.
"""

import matplotlib.pyplot as plt
import numpy as np

from roboeval.action_modes import JointPositionActionMode
from roboeval.envs.lift_pot import LiftPot, LiftPotPosition, LiftPotOrientation, LiftPotPositionAndOrientation
from roboeval.envs.manipulation import (CubeHandover, CubeHandoverPosition, CubeHandoverOrientation, CubeHandoverPositionAndOrientation, VerticalCubeHandover, StackTwoBlocks, StackTwoBlocksPosition, StackTwoBlocksOrientation, StackTwoBlocksPositionAndOrientation)

from roboeval.envs.stack_books import (
    StackSingleBookShelf,
    StackSingleBookShelfPosition,
    StackSingleBookShelfPositionAndOrientation,
    PickSingleBookFromTable,
    PickSingleBookFromTablePosition,
    PickSingleBookFromTableOrientation,
    PickSingleBookFromTablePositionAndOrientation,
)

from roboeval.envs.pack_objects import (
    PackBox,
    PackBoxOrientation,
    PackBoxPosition,
    PackBoxPositionAndOrientation
)

from roboeval.envs.lift_tray import LiftTray, LiftTrayPosition, LiftTrayOrientation, LiftTrayPositionAndOrientation, DragOverAndLiftTray
from roboeval.envs.rotate_utility_objects import (
    RotateValve,
    RotateValvePosition, 
    RotateValvePositionAndOrientation
)

from roboeval.demonstrations.demo_converter import DemoConverter
from roboeval.demonstrations.demo_player import DemoPlayer
from roboeval.demonstrations.demo_store import DemoStore
from roboeval.demonstrations.utils import Metadata, ObservationMode
from roboeval.robots.configs.panda import BimanualPanda
from roboeval.utils.observation_config import ObservationConfig, CameraConfig

def update_plots(axs, requested, expected, actual, xlim=None, ylim=None):
    """Update plots."""
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
    """Initialize subplots."""
    fig = plt.figure(figsize=(8, 2 * n))
    axs = [fig.add_subplot(n, 1, i + 1) for i in range(n)]
    return fig, axs

def replay_with_conversion(demo, env, conversion_type="none", plot_results=False):
    """Replay demo with specified conversion type.
    
    Args:
        demo: The demonstration to replay
        env: The environment to replay in
        conversion_type: Type of conversion to apply
            - "none": No conversion
            - "joint_to_ee": Convert joint positions to end-effector positions
            - "absolute_to_delta": Convert absolute positions to delta positions
            - "joint_absolute_to_ee_delta": Convert joint absolute to EE delta
        plot_results: Whether to plot the results
    """
    # Apply conversion based on type
    converted_demo = demo
    if conversion_type == "joint_to_ee":
        converted_demo = DemoConverter.joint_to_ee(demo)
    elif conversion_type == "absolute_to_delta":
        converted_demo = DemoConverter.absolute_to_delta(demo)
    elif conversion_type == "joint_absolute_to_ee_delta":
        converted_demo = DemoConverter.joint_absolute_to_ee_delta(demo)
    
    env.reset(seed=converted_demo.seed)
    
    request = []
    actual = []
    expected = []
    
    # Replay the converted demo
    for timestep in converted_demo.timesteps:
        action = timestep.executed_action
        obs, reward, termination, truncation, info = env.step(action)
        request.append(action)
        actual.append(env.robot.qpos_actuated)
        expected.append(timestep.observation.get('robot_joint_positions', env.robot.qpos_actuated))
        env.render()
    
    if plot_results and len(request) > 0:
        # Plot results
        request = np.array(request).T
        expected = np.array(expected).T
        actual = np.array(actual).T
        
        fig, axs = init_subplots(len(env.action_space.sample()))
        amplitude = abs(actual).max()
        update_plots(
            axs,
            request,
            expected,
            actual,
            ylim=(-amplitude * 1.1, amplitude * 1.1),
        )
        plt.title(f"Conversion Type: {conversion_type}")
        plt.show()

# Configuration
control_frequency = 20
render = True  # Set to False for testing without GUI
plot_results = False  # Set to False for testing without plots

# Environment configurations for different action modes
env_configs = {
    "joint_absolute": {
        "action_mode": JointPositionActionMode(floating_base=True, absolute=True, floating_dofs=[]),
        "conversion": "none"
    },
    "joint_delta": {
        "action_mode": JointPositionActionMode(floating_base=True, absolute=False, floating_dofs=[]),
        "conversion": "absolute_to_delta"
    },
    "ee_absolute": {
        "action_mode": JointPositionActionMode(floating_base=True, absolute=True, ee=True, floating_dofs=[]),
        "conversion": "joint_to_ee"
    },
    "ee_delta": {
        "action_mode": JointPositionActionMode(floating_base=True, absolute=False, ee=True, floating_dofs=[]),
        "conversion": "joint_absolute_to_ee_delta"
    }
}

# Choose which environment and conversion to test
test_config = "joint_delta"  # Change this to test different configurations

# Create environment
env = PickSingleBookFromTablePositionAndOrientation(
    action_mode=env_configs[test_config]["action_mode"],
    render_mode="human" if render else None,
    control_frequency=control_frequency,
    robot_cls=BimanualPanda,
    observation_config=ObservationConfig(
        cameras=[
            CameraConfig(
                name="external",
                rgb=True,
                depth=False,
                resolution=(128, 128),
                pos=np.array([0.0, 10.0, 10.0]).tolist(),
            )
        ],
    ),
)

metadata = Metadata.from_env(env)

# Get demonstrations from DemoStore
demo_store = DemoStore()

# First, try to get demos for the exact metadata (may not exist for non-absolute modes)
try:
    demos = demo_store.get_demos(metadata, amount=5, frequency=control_frequency)
    print(f"Found demos with exact metadata match")
except Exception as e:
    print(f"No exact metadata match found: {e}")
    
    # Fall back to loading lightweight demos (which are always in absolute joint mode)
    print("Loading lightweight demos and converting them...")
    
    # Create metadata for lightweight absolute joint demos
    lightweight_metadata = Metadata.from_env(env)
    lightweight_metadata.observation_mode = ObservationMode.Lightweight
    lightweight_metadata.environment_data.action_mode_absolute = True
    lightweight_metadata.environment_data.end_effector_mode = False
    
    # Load the lightweight demos in absolute joint mode
    demos = demo_store.get_demos(lightweight_metadata, amount=5, frequency=control_frequency)
    print(f"Loaded {len(demos)} lightweight demos in absolute joint mode")
    
    # Convert each demo to the target action mode
    converted_demos = []
    for demo in demos:
        converted_demo = demo
        
        # Apply the appropriate conversion based on target action mode
        if metadata.environment_data.end_effector_mode and metadata.environment_data.action_mode_absolute:
            # Target: Absolute EE
            print(f"Converting demo {demo.uuid} from joint absolute to EE absolute")
            converted_demo = DemoConverter.joint_to_ee(demo)
        elif metadata.environment_data.end_effector_mode and not metadata.environment_data.action_mode_absolute:
            # Target: Delta EE
            print(f"Converting demo {demo.uuid} from joint absolute to EE delta")
            converted_demo = DemoConverter.joint_absolute_to_ee_delta(demo)
        elif not metadata.environment_data.end_effector_mode and not metadata.environment_data.action_mode_absolute:
            # Target: Delta Joint
            print(f"Converting demo {demo.uuid} from joint absolute to joint delta")
            converted_demo = DemoConverter.absolute_to_delta(demo)
        else:
            # Target is absolute joint (same as source), no conversion needed
            print(f"No conversion needed for demo {demo.uuid} (already in target mode)")
        
        converted_demos.append(converted_demo)
    
    demos = converted_demos
        
    
        # print(f"Failed to load lightweight demos: {e2}")
        # demos = []

print(f"Testing configuration: {test_config}")
print(f"Conversion type: {env_configs[test_config]['conversion']}")
print(f"Loaded {len(demos)} demonstrations")

if not demos:
    print("No demos found! Exiting.")
    env.close()
    exit()

# Replay demonstrations (no additional conversion needed since we already converted them)
for i, demo in enumerate(demos):  # Test first 3 demos
    print(f"\nReplaying demo {i+1}: {demo.uuid}")
    replay_with_conversion(
        demo, 
        env, 
        conversion_type="none",  # No additional conversion needed
        plot_results=plot_results
    )

env.close()
print("Done!")