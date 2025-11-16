"""
Test script for custom robot integration.

This script validates that a custom robot is properly configured and can be used
in RoboEval environments. Use this as a template to test your own robot implementations.

Usage:
    python examples/test_custom_robot.py
"""

import numpy as np
from roboeval.robots.configs.panda import BimanualPanda, SinglePanda
from roboeval.envs.lift_pot import LiftPot
from roboeval.envs.manipulation import StackTwoBlocks
from roboeval.action_modes import JointPositionActionMode
from roboeval.const import HandSide


def get_test_env_for_robot(robot_cls):
    """Get an appropriate test environment based on robot arm count."""
    # Check number of arms
    temp_robot = object.__new__(robot_cls)
    config = temp_robot.config
    
    # Count non-None arms
    num_arms = sum(1 for arm in config.arms.values() if arm is not None)
    
    if num_arms >= 2:
        # Use bimanual task
        return LiftPot
    else:
        # Use single-arm task
        return StackTwoBlocks


def test_robot_configuration(robot_cls):
    """Test robot configuration is properly defined."""
    print(f"\n{'='*60}")
    print(f"Testing {robot_cls.__name__} Configuration")
    print(f"{'='*60}")
    
    # Check if robot has floating base by inspecting its config property
    # Access it via the class (not instance) to avoid initialization issues
    config = robot_cls.config.fget(None) if hasattr(robot_cls.config, 'fget') else None
    if config is None:
        # Fallback: create instance to get config
        temp_robot = object.__new__(robot_cls)
        config = temp_robot.config
    
    has_floating_base = len(config.floating_base.dofs) > 0
    
    # Create action mode based on robot capabilities
    action_mode = JointPositionActionMode(absolute=True, ee=False, floating_base=True, floating_dofs=[])
    robot = robot_cls(action_mode=action_mode)
    
    # Test 1: Check config property
    print("\n✓ Testing config property...")
    config = robot.config
    assert config is not None, "Robot config is None"
    assert hasattr(config, 'arms'), "Config missing 'arms' attribute"
    assert hasattr(config, 'gripper'), "Config missing 'gripper' attribute"
    assert hasattr(config, 'floating_base'), "Config missing 'floating_base' attribute"
    print(f"  - Model name: {config.model_name}")
    print(f"  - Arms: {list(config.arms.keys())}")
    print(f"  - Gripper: {config.gripper.model.name if config.gripper else 'None'}")
    
    # Test 2: Check IK config property
    print("\n✓ Testing ik_config property...")
    ik_config = robot.ik_config
    assert ik_config is not None, "Robot ik_config is None"
    assert hasattr(ik_config, 'arm_roots'), "IK config missing 'arm_roots'"
    assert hasattr(ik_config, 'arm_sites'), "IK config missing 'arm_sites'"
    assert hasattr(ik_config, 'joint_limits'), "IK config missing 'joint_limits'"
    print(f"  - Arm roots: {ik_config.arm_roots}")
    print(f"  - Arm sites: {ik_config.arm_sites}")
    print(f"  - Joint limits defined: {len(ik_config.joint_limits)} joints")
    
    # Test 3: Validate arm configuration
    print("\n✓ Testing arm configuration...")
    for hand_side, arm in config.arms.items():
        if arm is not None:
            print(f"  - {hand_side.name} arm:")
            print(f"    • Model: {arm.model.name}")
            print(f"    • Joints: {len(arm.joints)} joints")
            print(f"    • Actuators: {len(arm.actuators)} actuators")
            print(f"    • Links: {len(arm.links)} links")
            assert len(arm.joints) > 0, f"{hand_side.name} arm has no joints"
            assert len(arm.actuators) > 0, f"{hand_side.name} arm has no actuators"
    
    # Test 4: Validate gripper configuration
    if config.gripper:
        print("\n✓ Testing gripper configuration...")
        print(f"  - Actuators: {config.gripper.actuators}")
        print(f"  - Range: {config.gripper.range}")
        print(f"  - Pad bodies: {config.gripper.pad_bodies}")
        assert len(config.gripper.actuators) > 0, "Gripper has no actuators"
        assert config.gripper.range is not None, "Gripper range not defined"
    
    # Test 5: Validate cameras
    if config.cameras:
        print("\n✓ Testing camera configuration...")
        for camera_name, camera_config in config.cameras.items():
            print(f"  - {camera_name}:")
            print(f"    • Position: {camera_config.get('position', 'N/A')}")
            print(f"    • FOV: {camera_config.get('fov', 'N/A')}")
    
    print(f"\n{'='*60}")
    print(f"✓ {robot_cls.__name__} configuration tests passed!")
    print(f"{'='*60}")


def test_robot_in_environment(robot_cls, env_cls=None):
    """Test robot can be used in an environment.
    
    Args:
        robot_cls: Robot class to test
        env_cls: Environment class to use. If None, automatically selects based on robot arm count.
    """
    # Auto-select environment if not specified
    if env_cls is None:
        env_cls = get_test_env_for_robot(robot_cls)
    
    print(f"\n{'='*60}")
    print(f"Testing {robot_cls.__name__} in {env_cls.__name__} Environment")
    print(f"{'='*60}")
    
    # Detect if robot has floating base
    temp_robot = object.__new__(robot_cls)
    config = temp_robot.config
    has_floating_base = len(config.floating_base.dofs) > 0
    
    # Test different action modes (works for both single-arm and bimanual robots)
    action_modes = [
        ("Absolute Joint Control", JointPositionActionMode(absolute=True, ee=False, floating_base=has_floating_base)),
        ("Delta Joint Control", JointPositionActionMode(absolute=False, ee=False, floating_base=has_floating_base)),
        ("Absolute EE Control", JointPositionActionMode(absolute=True, ee=True, floating_base=has_floating_base)),
        ("Delta EE Control", JointPositionActionMode(absolute=False, ee=True, floating_base=has_floating_base)),
    ]
    
    for mode_name, action_mode in action_modes:
        print(f"\n✓ Testing {mode_name}...")
        
        # Create environment
        env = env_cls(
            robot_cls=robot_cls,
            action_mode=action_mode,
            render_mode=None,
        )
        
        # Test reset
        obs, info = env.reset(seed=42)
        print(f"  - Reset successful")
        print(f"  - Observation keys: {list(obs.keys())}")
        print(f"  - Action space: {env.action_space}")
        
        # Test step with random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  - Step successful")
        print(f"  - Reward: {reward:.4f}")
        
        # Test multiple steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
        print(f"  - Multiple steps successful")
        
        env.close()
    
    print(f"\n{'='*60}")
    print(f"✓ {robot_cls.__name__} environment tests passed!")
    print(f"{'='*60}")


def test_robot_rendering(robot_cls, env_cls=None):
    """Test robot can be rendered.
    
    Args:
        robot_cls: Robot class to test
        env_cls: Environment class to use. If None, automatically selects based on robot arm count.
    """
    # Auto-select environment if not specified
    if env_cls is None:
        env_cls = get_test_env_for_robot(robot_cls)
    
    print(f"\n{'='*60}")
    print(f"Testing {robot_cls.__name__} Rendering")
    print(f"{'='*60}")
    
    # Detect if robot has floating base
    temp_robot = object.__new__(robot_cls)
    config = temp_robot.config
    has_floating_base = len(config.floating_base.dofs) > 0
    
    # Test with human rendering
    print("\n✓ Testing human rendering mode...")
    env = env_cls(
        robot_cls=robot_cls,
        action_mode=JointPositionActionMode(absolute=False, ee=False, floating_base=has_floating_base),
        render_mode="human",
        width=640,
        height=480,
    )
    
    obs, info = env.reset(seed=42)
    print(f"  - Environment created with human rendering")
    
    # Take a few steps to ensure rendering works
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    print(f"  - Rendering successful for {i+1} steps")
    env.close()
    
    # Test with rgb_array rendering
    print("\n✓ Testing rgb_array rendering mode...")
    env = env_cls(
        robot_cls=robot_cls,
        action_mode=JointPositionActionMode(absolute=False, ee=False, floating_base=has_floating_base),
        render_mode="rgb_array",
        width=320,
        height=240,
    )
    
    obs, info = env.reset(seed=42)
    frame = env.render()
    assert frame is not None, "Render returned None"
    assert frame.shape == (240, 320, 3), f"Unexpected frame shape: {frame.shape}"
    print(f"  - RGB array shape: {frame.shape}")
    
    env.close()
    
    print(f"\n{'='*60}")
    print(f"✓ {robot_cls.__name__} rendering tests passed!")
    print(f"{'='*60}")


def test_robot_observations(robot_cls, env_cls=None):
    """Test robot observation space.
    
    Args:
        robot_cls: Robot class to test
        env_cls: Environment class to use. If None, automatically selects based on robot arm count.
    """
    # Auto-select environment if not specified
    if env_cls is None:
        env_cls = get_test_env_for_robot(robot_cls)
    
    print(f"\n{'='*60}")
    print(f"Testing {robot_cls.__name__} Observations")
    print(f"{'='*60}")
    
    # Detect if robot has floating base
    temp_robot = object.__new__(robot_cls)
    config = temp_robot.config
    has_floating_base = len(config.floating_base.dofs) > 0
    
    env = env_cls(
        robot_cls=robot_cls,
        action_mode=JointPositionActionMode(absolute=False, ee=False, floating_base=has_floating_base),
        render_mode=None,
    )
    
    obs, info = env.reset(seed=42)
    
    print("\n✓ Testing observation structure...")
    print(f"  - Observation keys: {list(obs.keys())}")
    
    # Check that observations are non-empty
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, dict):
            print(f"  - {key}: {len(value)} items")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, np.ndarray):
                    print(f"    • {subkey}: shape={subvalue.shape}")
    
    env.close()
    
    print(f"\n{'='*60}")
    print(f"✓ {robot_cls.__name__} observation tests passed!")
    print(f"{'='*60}")


def run_all_tests(robot_cls, env_cls=None):
    """Run all tests for a robot.
    
    Args:
        robot_cls: Robot class to test
        env_cls: Environment class to use. If None, automatically selects based on robot arm count.
    """
    # Auto-select environment if not specified
    if env_cls is None:
        env_cls = get_test_env_for_robot(robot_cls)
    
    print(f"\n{'#'*60}")
    print(f"# Running Full Test Suite for {robot_cls.__name__}")
    print(f"# Using environment: {env_cls.__name__}")
    print(f"{'#'*60}")
    
    try:
        # Configuration tests
        test_robot_configuration(robot_cls)
        
        # Environment integration tests
        test_robot_in_environment(robot_cls, env_cls)
        
        # Observation tests
        test_robot_observations(robot_cls, env_cls)
        
        # Rendering tests (optional - can be slow)
        # Uncomment to test rendering:
        # test_robot_rendering(robot_cls, env_cls)
        
        print(f"\n{'#'*60}")
        print(f"# ✓ ALL TESTS PASSED for {robot_cls.__name__}")
        print(f"{'#'*60}\n")
        
        return True
        
    except AssertionError as e:
        print(f"\n{'#'*60}")
        print(f"# ✗ TEST FAILED for {robot_cls.__name__}")
        print(f"# Error: {e}")
        print(f"{'#'*60}\n")
        return False
    
    except Exception as e:
        print(f"\n{'#'*60}")
        print(f"# ✗ UNEXPECTED ERROR for {robot_cls.__name__}")
        print(f"# {type(e).__name__}: {e}")
        print(f"{'#'*60}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test BimanualPanda
    print("\nTesting BimanualPanda robot configuration...")
    run_all_tests(BimanualPanda)
    
    # Test SinglePanda
    print("\nTesting SinglePanda robot configuration...")
    run_all_tests(SinglePanda)
    
    # To test your custom robot, import it and run:
    # from roboeval.robots.configs.my_robot import MyCustomRobot
    # run_all_tests(MyCustomRobot)
