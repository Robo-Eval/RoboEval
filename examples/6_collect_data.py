"""Data Collection Example Script.

This script demonstrates how to:
1. Set up keyboard-based teleoperation for data collection
2. Record demonstrations with the Bimanual Panda robot
3. Save demonstrations to disk
4. Load and replay saved demonstrations

The script replicates the functionality of the JSON launch configuration:
{
    "name": "Python: test single panda",
    "type": "python", 
    "request": "launch",
    "program": "${workspaceFolder}/roboeval/data_collection/demo_recorder.py",
    "args": ["input_mode=Keyboard", "robot=Bimanual Panda", "env=Lift Pot"]
}

Usage:
    python examples/6_collect_data.py
    
Controls (during data collection):
    Left Arm Movement:
    - A/D: Left/Right (X-axis)
    - Z/C: Forward/Backward (Y-axis) 
    - W/S: Up/Down (Z-axis)
    - V: Gripper open/close
    
    Right Arm Movement:
    - J/L: Left/Right (X-axis)
    - U/O: Forward/Backward (Y-axis)
    - I/K: Up/Down (Z-axis) 
    - B: Gripper open/close
    
    Recording Controls:
    - R: Start/stop recording demonstration
    - X: Save current demonstration
    - T: Toggle between position/orientation control modes
    - G: Toggle gripper mode (autoclose vs hold-to-close)
    - ESC: Exit
"""

import tempfile
from pathlib import Path
import numpy as np
from typing import Optional

# RoboEval imports
from roboeval.action_modes import JointPositionActionMode
from roboeval.envs.lift_pot import LiftPot
from roboeval.robots.configs.panda import BimanualPanda
from roboeval.utils.observation_config import ObservationConfig, CameraConfig
from roboeval.demonstrations.demo import Demo
from roboeval.demonstrations.demo_player import DemoPlayer
from roboeval.demonstrations.demo_recorder import DemoRecorder
from roboeval.data_collection.keyboard_input import KeyboardTeleop


def setup_environment(render_mode: str = "human") -> LiftPot:
    """Set up the Lift Pot environment with Bimanual Panda robot.
    
    Args:
        render_mode: Rendering mode ("human" for GUI, None for headless)
        
    Returns:
        Configured LiftPot environment
    """
    print("Setting up Lift Pot environment with Bimanual Panda robot...")
    
    env = LiftPot(
        action_mode=JointPositionActionMode(
            floating_base=True,
            absolute=True,
            floating_dofs=[]
        ),
        render_mode=render_mode,
        robot_cls=BimanualPanda,
        control_frequency=20,  # 20 Hz control frequency
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
    
    print(f"Environment created with action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    return env


def collect_demonstration_keyboard(demo_dir: Path) -> Optional[Path]:
    """Collect a demonstration using keyboard teleoperation.
    
    Args:
        demo_dir: Directory to save demonstrations
        
    Returns:
        Path to saved demonstration file, or None if no demo was recorded
    """
    print("\n" + "="*60)
    print("STARTING KEYBOARD DATA COLLECTION")
    print("="*60)
    print("\nKeyboard Controls:")
    print("  Left Arm Movement:")
    print("    A/D: Left/Right (X-axis)")
    print("    Z/C: Forward/Backward (Y-axis)")
    print("    W/S: Up/Down (Z-axis)")
    print("    V: Gripper Open/Close")
    print("\n  Right Arm Movement:")
    print("    J/L: Left/Right (X-axis)")
    print("    U/O: Forward/Backward (Y-axis)")
    print("    I/K: Up/Down (Z-axis)")
    print("    B: Gripper Open/Close")
    print("\n  Recording Controls:")
    print("    R: Start/Stop Recording")
    print("    X: Save Demonstration")
    print("    T: Toggle Control Mode (Position/Orientation)")
    print("    G: Toggle Gripper Mode (Autoclose/Hold-to-Close)")
    print("    ESC: Exit")
    print("\n" + "="*60)
    
    # Set up keyboard teleoperation
    teleop = KeyboardTeleop(
        env_cls=LiftPot,
        action_mode=JointPositionActionMode(
            floating_base=True,
            absolute=True,
            floating_dofs=[]
        ),
        resolution=(900, 1000),
        demo_directory=demo_dir,
        robot_cls=BimanualPanda,
        config={"env": "Lift Pot", "robot": "Bimanual Panda"}
    )
    
    print("\nStarting teleoperation interface...")
    print("Use the controls above to manipulate the robot and collect demonstrations.")
    print("Press 'R' to start recording, then 'X' to save when done.")
    
    try:
        # This will block until the user exits
        teleop.run()
        
        # Check if any demonstrations were saved
        demo_files = list(demo_dir.glob("*.safetensors"))
        if demo_files:
            latest_demo = max(demo_files, key=lambda f: f.stat().st_mtime)
            print(f"\nDemo saved: {latest_demo}")
            return latest_demo
        else:
            print("\nNo demonstrations were saved.")
            return None
            
    except KeyboardInterrupt:
        print("\nKeyboard teleoperation interrupted by user.")
        return None
    except Exception as e:
        print(f"\nError during teleoperation: {e}")
        return None


def collect_demonstration_programmatic(demo_dir: Path) -> Path:
    """Collect a demonstration programmatically (scripted movements).
    
    This demonstrates how to record demonstrations without human input,
    useful for automated data generation or testing.
    
    Args:
        demo_dir: Directory to save demonstrations
        
    Returns:
        Path to saved demonstration file
    """
    print("\n" + "="*60)
    print("STARTING PROGRAMMATIC DATA COLLECTION")
    print("="*60)
    
    # Set up environment
    env = setup_environment(render_mode="human")
    
    # Set up demo recorder
    demo_recorder = DemoRecorder(demo_dir)
    
    print("\nGenerating programmatic demonstration...")
    print("The robot will perform a simple lifting motion.")
    
    try:
        # Reset environment and start recording
        env.reset()
        demo_recorder.record(env)
        
        # Define a simple lifting trajectory
        episode_length = 200
        amplitude = 0.3
        frequency = 0.02
        
        print(f"Recording {episode_length} steps...")
        
        for i in range(episode_length):
            # Create a simple sinusoidal motion for demonstration
            # This moves both arms up and down in a coordinated fashion
            action = np.zeros_like(env.action_space.sample())
            
            # Bimanual Panda has 16 DOFs (7 per arm + 2 grippers)
            # Move shoulder and elbow joints in a lifting pattern
            action[0] = amplitude * np.sin(frequency * i)  # Left arm shoulder
            action[3] = -np.pi/2 + amplitude * np.sin(frequency * i)  # Left arm elbow
            action[7] = amplitude * np.sin(frequency * i)  # Right arm shoulder  
            action[10] = -np.pi/2 + amplitude * np.sin(frequency * i)  # Right arm elbow
            
            # Execute action and record
            timestep = env.step(action)
            demo_recorder.add_timestep(timestep, action.copy())
            
            # Render if visualization is enabled
            if env.render_mode:
                env.render()
        
        # Stop recording and save
        demo_recorder.stop()
        demo_path = demo_recorder.save_demo()
        
        env.close()
        
        print(f"\nProgrammatic demo saved: {demo_path}")
        return demo_path
        
    except Exception as e:
        print(f"Error during programmatic collection: {e}")
        env.close()
        raise


def replay_demonstration(demo_path: Path, render: bool = True):
    """Load and replay a saved demonstration.
    
    Args:
        demo_path: Path to the demonstration file
        render: Whether to render the replay visually
    """
    print("\n" + "="*60)
    print("REPLAYING DEMONSTRATION")
    print("="*60)
    
    if not demo_path.exists():
        print(f"Demo file not found: {demo_path}")
        return
    
    try:
        # Load the demonstration
        print(f"Loading demonstration from: {demo_path}")
        demo = Demo.from_safetensors(demo_path)
        
        print(f"Demo metadata:")
        print(f"  Environment: {demo.metadata.environment_data.env_name}")
        print(f"  Robot: {demo.metadata.environment_data.robot_name}")
        print(f"  Duration: {demo.duration} steps")
        print(f"  Seed: {demo.seed}")
        
        # Set up environment for replay
        render_mode = "human" if render else None
        env = setup_environment(render_mode=render_mode)
        
        print(f"\nReplaying demonstration...")
        if render:
            print("Close the visualization window to continue.")
        
        # Use DemoPlayer to replay the demonstration
        player = DemoPlayer()
        player.replay_in_env(demo, env, demo_frequency=20)
        
        env.close()
        print("Demonstration replay completed.")
        
    except Exception as e:
        print(f"Error during replay: {e}")
        raise


def main():
    """Main function demonstrating the complete data collection and replay workflow."""
    print("RoboEval Data Collection Example")
    print("="*40)
    
    # Create temporary directory for demonstrations
    with tempfile.TemporaryDirectory() as temp_dir:
        demo_dir = Path(temp_dir)
        print(f"Using temporary demo directory: {demo_dir}")
        
        # Option 1: Interactive keyboard-based data collection
        print("\nOption 1: Interactive Keyboard Data Collection")
        print("This will open a teleoperation interface for manual control.")
        
        response = input("\nWould you like to try keyboard teleoperation? (y/n): ").lower().strip()
        demo_path = None
        
        if response == 'y':
            demo_path = collect_demonstration_keyboard(demo_dir)
        
        # Option 2: Programmatic demonstration generation
        if demo_path is None:
            print("\nOption 2: Programmatic Data Collection")
            print("Generating a scripted demonstration...")
            demo_path = collect_demonstration_programmatic(demo_dir)
        
        # Replay the demonstration
        if demo_path and demo_path.exists():
            print(f"\nDemonstration successfully created: {demo_path.name}")
            
            response = input("\nWould you like to replay the demonstration? (y/n): ").lower().strip()
            if response == 'y':
                replay_demonstration(demo_path, render=True)
        
        print("\n" + "="*60)
        print("Data collection example completed!")
        print("="*60)
        print("\nTo save demonstrations permanently:")
        print("1. Create a dedicated directory for your demos")
        print("2. Pass that directory to the DemoRecorder or KeyboardTeleop")
        print("3. Use the same replay code to load and visualize your data")
        print("\nFor production use:")
        print("- Use the demo_recorder.py script with hydra configs")
        print("- Integrate with the DemoStore for dataset management")
        print("- Consider using the GUI tools in tools/demo_player/")


if __name__ == "__main__":
    main()
