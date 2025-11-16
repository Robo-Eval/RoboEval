"""Oculus VR Data Collection Example Script.

This script demonstrates how to:
1. Set up Oculus Quest VR-based teleoperation for data collection
2. Record demonstrations with immersive VR control
3. Save demonstrations to disk
4. Load and replay saved demonstrations

The Oculus Quest provides intuitive bimanual control where your hand movements
are directly mapped to the robot's end-effectors, making it ideal for collecting
high-quality demonstrations for bimanual manipulation tasks.

Prerequisites:
    - Oculus Quest headset (Quest 2, Quest Pro, or Quest 3)
    - oculus_reader package installed (pip install -e ".[vr]")
    - USB-C cable for connecting headset to computer
    - Developer mode enabled on Oculus Quest
    - ADB installed and configured

Setup Instructions:
    1. Enable Developer Mode on your Oculus Quest:
       - Open Oculus app on phone
       - Go to Settings > Developer Mode
       - Toggle Developer Mode ON
    
    2. Connect Oculus Quest to computer via USB-C cable
    
    3. Put on the headset and allow USB debugging when prompted
    
    4. Verify connection:
       adb devices
       # Should show your Quest device
    
    5. Install VR dependencies:
       pip install -e ".[vr]"

Usage:
    python examples/7_collect_data_oculus.py
    
VR Controls (in headset):
    Left Controller:
    - Position: Physical hand position controls left gripper position
    - Rotation: Physical hand rotation controls left gripper orientation
    - Trigger: Squeeze to close gripper, release to open
    
    Right Controller:
    - Position: Physical hand position controls right gripper position
    - Rotation: Physical hand rotation controls right gripper orientation
    - Trigger: Squeeze to close gripper, release to open
    
    Recording Controls:
    - A Button (right controller): Start/stop recording demonstration
    - B Button (right controller): Save current demonstration
    - X Button (left controller): Reset environment
    - Y Button (left controller): Toggle gripper autoclose mode
    - Thumbstick: Adjust robot base position (if floating base enabled)

For detailed VR setup instructions, see:
    roboeval/data_collection/README.md
"""

import tempfile
from pathlib import Path
import sys
from typing import Optional

# RoboEval imports
from roboeval.action_modes import JointPositionActionMode
from roboeval.envs.lift_pot import LiftPot
from roboeval.envs.lift_tray import LiftTray
from roboeval.envs.manipulation import CubeHandover, StackTwoBlocks
from roboeval.envs.stack_books import PickSingleBookFromTable, StackSingleBookShelf
from roboeval.envs.pack_objects import PackBox
from roboeval.robots.configs.panda import BimanualPanda
from roboeval.demonstrations.demo import Demo
from roboeval.demonstrations.demo_player import DemoPlayer

# Try to import VR dependencies - fail gracefully if not available
VR_AVAILABLE = True
VR_ERROR_MSG = None

try:
    from roboeval.data_collection.oculus_input import OculusTeleop
except ImportError as e:
    VR_AVAILABLE = False
    VR_ERROR_MSG = f"VR dependencies not installed: {e}"
except OSError as e:
    VR_AVAILABLE = False
    if "GLIBC" in str(e):
        VR_ERROR_MSG = (
            "GLIBC version incompatibility detected.\n"
            "The PyOpenXR library requires GLIBC 2.32 or later, but your system has an older version.\n\n"
            "Possible solutions:\n"
            "1. Upgrade your system (Ubuntu 20.10+ or equivalent)\n"
            "2. Use the direct demo_recorder.py script which may have better compatibility\n"
            "3. Build PyOpenXR from source for your system\n"
            "4. Use Docker with a newer base image (Ubuntu 20.10+)\n\n"
            f"Full error: {e}"
        )
    else:
        VR_ERROR_MSG = f"VR system libraries error: {e}"


def check_vr_setup() -> bool:
    """Check if VR dependencies are installed and Oculus is connected.
    
    Returns:
        True if setup is complete, False otherwise
    """
    print("Checking VR setup...")
    
    # First check if VR imports are available
    if not VR_AVAILABLE:
        print("✗ VR dependencies not available")
        print(f"\nError: {VR_ERROR_MSG}")
        return False
    
    # Check if oculus_reader is installed
    try:
        import oculus_reader
        print("✓ oculus_reader package installed")
    except ImportError:
        print("✗ oculus_reader package not found")
        print("\nPlease install VR dependencies:")
        print("  pip install -e \".[vr]\"")
        return False
    
    # Check if ADB is available
    import subprocess
    try:
        result = subprocess.run(
            ["adb", "devices"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✓ ADB installed")
            
            # Check if Quest is connected
            devices = result.stdout.strip().split('\n')[1:]  # Skip header
            if any(devices):
                print("✓ Oculus Quest device detected")
                return True
            else:
                print("✗ No Oculus Quest device connected")
                print("\nPlease connect your Oculus Quest via USB-C cable")
                print("and enable USB debugging in the headset.")
                return False
        else:
            print("✗ ADB not working properly")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("✗ ADB not found in PATH")
        print("\nPlease install Android Debug Bridge (ADB):")
        print("  sudo apt install android-tools-adb  # Ubuntu/Debian")
        print("  brew install android-platform-tools  # macOS")
        return False


def select_task():
    """Interactive task selection menu.
    
    Returns:
        Selected task class
    """
    tasks = {
        "1": ("Lift Pot", LiftPot),
        "2": ("Lift Tray", LiftTray),
        "3": ("Cube Handover", CubeHandover),
        "4": ("Stack Two Blocks", StackTwoBlocks),
        "5": ("Pick Single Book From Table", PickSingleBookFromTable),
        "6": ("Stack Single Book Shelf", StackSingleBookShelf),
        "7": ("Pack Box", PackBox),
    }
    
    print("\n" + "="*60)
    print("SELECT TASK FOR VR DATA COLLECTION")
    print("="*60)
    
    for key, (name, _) in tasks.items():
        print(f"  {key}. {name}")
    
    print("="*60)
    
    while True:
        choice = input("\nEnter task number (1-7): ").strip()
        if choice in tasks:
            task_name, task_cls = tasks[choice]
            print(f"\nSelected: {task_name}")
            return task_cls
        else:
            print("Invalid choice. Please enter a number between 1 and 7.")


def collect_demonstration_vr(
    task_cls,
    demo_dir: Path,
    control_frequency: int = 20
) -> Optional[Path]:
    """Collect demonstrations using Oculus VR teleoperation.
    
    Args:
        task_cls: Environment class to use for data collection
        demo_dir: Directory to save demonstrations
        control_frequency: Control frequency in Hz (default: 20)
        
    Returns:
        Path to saved demonstration file, or None if no demo was recorded
    """
    if not VR_AVAILABLE:
        print("\n✗ VR teleoperation is not available on this system.")
        print(f"\nReason: {VR_ERROR_MSG}")
        return None
    
    print("\n" + "="*60)
    print("STARTING OCULUS VR DATA COLLECTION")
    print("="*60)
    print("\nVR Controls (in headset):")
    print("  Left Controller:")
    print("    - Hand Position: Controls left gripper position")
    print("    - Hand Rotation: Controls left gripper orientation")
    print("    - Trigger: Close/open left gripper")
    print("\n  Right Controller:")
    print("    - Hand Position: Controls right gripper position")
    print("    - Hand Rotation: Controls right gripper orientation")
    print("    - Trigger: Close/open right gripper")
    print("\n  Recording Controls:")
    print("    - A Button: Start/Stop Recording")
    print("    - B Button: Save Demonstration")
    print("    - X Button: Reset Environment")
    print("    - Y Button: Toggle Gripper Autoclose Mode")
    print("\n" + "="*60)
    
    print("\n" + "="*60)
    print("VR SETUP INSTRUCTIONS")
    print("="*60)
    print("\n📍 Headset Placement:")
    print("  1. Place the Oculus Quest headset in front of you on a desk/table")
    print("  2. The front of the headset should face YOU (the data collector)")
    print("  3. This orientation allows the headset cameras to track your controllers")
    print("\n🎮 How VR Teleoperation Works:")
    print("  - The Quest's cameras track your hand controllers in 3D space")
    print("  - Your physical hand movements are mapped to the robot's grippers")
    print("  - The headset does NOT need to be worn during data collection")
    print("  - The headset acts as a tracking station for the controllers")
    print("  - Keep controllers within the headset's field of view for best tracking")
    print("\n⚠️  Important:")
    print("  - Ensure good lighting for optimal controller tracking")
    print("  - Keep the front cameras of the headset unobstructed")
    print("  - Controllers should remain visible to the headset cameras")
    print("\nThe VR environment will launch shortly...")
    print("="*60 + "\n")
    
    try:
        # Set up VR teleoperation
        teleop = OculusTeleop(
            env_cls=task_cls,
            action_mode=JointPositionActionMode(
                floating_base=True,
                absolute=True,
                floating_dofs=[]
            ),
            resolution=(900, 1000),
            demo_directory=demo_dir,
            robot_cls=BimanualPanda,
            config={
                "env": task_cls.__name__,
                "robot": "Bimanual Panda",
                "control_frequency": control_frequency
            }
        )
        
        print("\nVR teleoperation interface started!")
        print("Use the VR controllers to manipulate the robot.")
        print("Press 'A' button to start recording, then 'B' to save.")
        print("\nPress Ctrl+C to exit.")
        
        # This will block until the user exits
        teleop.run()
        
        # Check if any demonstrations were saved
        demo_files = list(demo_dir.glob("*.safetensors"))
        if demo_files:
            latest_demo = max(demo_files, key=lambda f: f.stat().st_mtime)
            print(f"\n✓ Demo saved: {latest_demo}")
            return latest_demo
        else:
            print("\nNo demonstrations were saved.")
            return None
            
    except KeyboardInterrupt:
        print("\n\nVR teleoperation interrupted by user.")
        return None
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("\nMake sure you have installed VR dependencies:")
        print("  pip install -e \".[vr]\"")
        return None
    except Exception as e:
        print(f"\n✗ Error during VR teleoperation: {e}")
        import traceback
        traceback.print_exc()
        return None


def replay_demonstration(demo_path: Path, task_cls, render: bool = True):
    """Load and replay a saved demonstration.
    
    Args:
        demo_path: Path to the demonstration file
        task_cls: Environment class used for the demonstration
        render: Whether to render the replay visually
    """
    print("\n" + "="*60)
    print("REPLAYING DEMONSTRATION")
    print("="*60)
    
    if not demo_path.exists():
        print(f"✗ Demo file not found: {demo_path}")
        return
    
    try:
        # Load the demonstration
        print(f"Loading demonstration from: {demo_path}")
        demo = Demo.from_safetensors(demo_path)
        
        print(f"\nDemo metadata:")
        print(f"  Environment: {demo.metadata.environment_data.env_name}")
        print(f"  Robot: {demo.metadata.environment_data.robot_name}")
        print(f"  Duration: {demo.duration} steps")
        print(f"  Seed: {demo.seed}")
        
        # Set up environment for replay
        render_mode = "human" if render else None
        env = task_cls(
            action_mode=JointPositionActionMode(
                floating_base=True,
                absolute=True,
                floating_dofs=[]
            ),
            render_mode=render_mode,
            robot_cls=BimanualPanda,
            control_frequency=20
        )
        
        print(f"\nReplaying demonstration...")
        if render:
            print("Close the visualization window to continue.")
        
        # Use DemoPlayer to replay the demonstration
        player = DemoPlayer()
        player.replay_in_env(demo, env, demo_frequency=20)
        
        env.close()
        print("✓ Demonstration replay completed.")
        
    except Exception as e:
        print(f"✗ Error during replay: {e}")
        import traceback
        traceback.print_exc()
        raise


def print_vr_setup_guide():
    """Print detailed VR setup instructions."""
    print("\n" + "="*60)
    print("OCULUS QUEST VR SETUP GUIDE")
    print("="*60)
    
    print("\n0. SYSTEM REQUIREMENTS")
    print("   - GLIBC 2.32 or later (Ubuntu 20.10+, Debian 11+, or equivalent)")
    print("   - Check your version: ldd --version")
    print("   - If you have an older system, consider using Docker")
    
    print("\n1. ENABLE DEVELOPER MODE")
    print("   a. Install Oculus app on your smartphone")
    print("   b. Sign in with your Meta/Facebook account")
    print("   c. Go to: Menu > Devices > [Your Quest] > Developer Mode")
    print("   d. Toggle Developer Mode ON")
    
    print("\n2. CONNECT QUEST TO COMPUTER")
    print("   a. Use a USB-C cable to connect Quest to your computer")
    print("   b. Put on the headset")
    print("   c. Allow USB debugging when prompted in VR")
    
    print("\n3. INSTALL DEPENDENCIES")
    print("   a. Install ADB (Android Debug Bridge):")
    print("      Ubuntu/Debian: sudo apt install android-tools-adb")
    print("      macOS: brew install android-platform-tools")
    print("      Windows: Download from https://developer.android.com/studio/releases/platform-tools")
    print("   b. Install VR Python packages:")
    print("      pip install -e \".[vr]\"")
    
    print("\n4. VERIFY CONNECTION")
    print("   Run: adb devices")
    print("   You should see your Quest listed")
    
    print("\n5. TROUBLESHOOTING")
    print("   - If Quest not detected: Try different USB port or cable")
    print("   - If USB debugging denied: Revoke USB debugging authorizations")
    print("     in Quest Settings > Developer, then reconnect")
    print("   - If 'oculus_reader' import fails: pip install -e \".[vr]\"")
    
    print("\nFor more details, see:")
    print("  roboeval/data_collection/README.md")
    print("="*60 + "\n")


def main():
    """Main function demonstrating VR data collection workflow."""
    print("="*60)
    print("RoboEval - Oculus VR Data Collection")
    print("="*60)
    
    # Check VR setup
    if not check_vr_setup():
        print("\n" + "="*60)
        print("VR SETUP INCOMPLETE")
        print("="*60)
        
        response = input("\nWould you like to see the VR setup guide? (y/n): ").lower().strip()
        if response == 'y':
            print_vr_setup_guide()
        
        print("\nPlease complete VR setup and try again.")
        sys.exit(1)
    
    print("\n✓ VR setup complete!")
    
    # Select task
    task_cls = select_task()
    
    # Create directory for demonstrations
    save_permanently = input("\nSave demos permanently? (y/n, default=n): ").lower().strip() == 'y'
    
    if save_permanently:
        demo_dir_input = input("Enter directory path (default=./vr_demos): ").strip()
        if demo_dir_input:
            demo_dir = Path(demo_dir_input)
        else:
            demo_dir = Path("./vr_demos")
        demo_dir.mkdir(parents=True, exist_ok=True)
        print(f"Demos will be saved to: {demo_dir.absolute()}")
        use_temp_dir = False
    else:
        demo_dir = None
        use_temp_dir = True
        print("Using temporary directory (demos will be deleted on exit)")
    
    # Data collection
    demo_path = None
    
    if use_temp_dir:
        with tempfile.TemporaryDirectory() as temp_dir:
            demo_dir = Path(temp_dir)
            print(f"\nUsing temporary demo directory: {demo_dir}")
            demo_path = collect_demonstration_vr(task_cls, demo_dir)
            
            # Replay if demo was created
            if demo_path and demo_path.exists():
                print(f"\n✓ Demonstration successfully created!")
                response = input("\nWould you like to replay the demonstration? (y/n): ").lower().strip()
                if response == 'y':
                    replay_demonstration(demo_path, task_cls, render=True)
    else:
        demo_path = collect_demonstration_vr(task_cls, demo_dir)
        
        # Replay if demo was created
        if demo_path and demo_path.exists():
            print(f"\n✓ Demonstration successfully created!")
            print(f"✓ Saved to: {demo_path}")
            
            response = input("\nWould you like to replay the demonstration? (y/n): ").lower().strip()
            if response == 'y':
                replay_demonstration(demo_path, task_cls, render=True)
    
    print("\n" + "="*60)
    print("VR Data Collection Completed!")
    print("="*60)
    
    if save_permanently and demo_path:
        print(f"\nYour demonstrations are saved in: {demo_dir.absolute()}")
        print("\nTo replay later:")
        print(f"  python examples/1_data_replay.py --demo_path {demo_path}")
    
    print("\nTips for collecting high-quality demonstrations:")
    print("  - Move controllers smoothly and deliberately")
    print("  - Complete the full task before saving")
    print("  - Collect multiple demonstrations of the same task")
    print("  - Try different starting configurations")
    print("  - Use the 'Y' button to toggle gripper autoclose mode")
    print("\nFor batch data collection:")
    print("  - Use the demo_recorder.py script with hydra configs")
    print("  - Integrate with the DemoStore for dataset management")
    print("  - See roboeval/data_collection/README.md for details")


if __name__ == "__main__":
    main()
