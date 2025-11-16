# RoboEval

[**RoboEval: Where Robotic Manipulation Meets Structured and Scalable Evaluation**](https://www.arxiv.org/abs/2507.00435)  
[Yi Ru Wang](https://helen9975.github.io/)<sup>1</sup>, [Carter Ung](https://carteruh.github.io/)<sup>1</sup>, Christopher Tan<sup>1</sup>, [Grant Tannert](https://www.linkedin.com/in/grant-tannert/)<sup>1</sup>, [Jiafei Duan](https://www.duanjiafei.com)<sup>1,2</sup>,  [Josephine Li](https://www.linkedin.com/in/josephine-li-90ba02324/)<sup>1</sup>, [Amy Le](https://www.linkedin.com/in/amyle139/)<sup>1</sup>, [Rishabh Oswal](https://www.linkedin.com/in/rishabh-oswal/)<sup>1</sup>, [Markus Grotz](https://www.markusgrotz.com/)<sup>1</sup>,  [Wilbert Pumacay](https://wpumacay.github.io/)<sup>1</sup>, [Yuquan Deng](https://www.linkedin.com/in/yuquand/)<sup>2</sup>, [Ranjay Krishna](https://ranjaykrishna.com/index.html)<sup>1,2</sup>, [Dieter Fox](https://homes.cs.washington.edu/~fox/)<sup>*1</sup>,  [Siddhartha Srinivasa](https://goodrobot.ai/)<sup>*1</sup>

<sup>1</sup>University of Washington, <sup>2</sup>Allen Institute for AI, <sup>*</sup>Equal advising

[In Submission]() 

## About

RoboEval is a benchmark for bimanual manipulation featuring:
- **8 task families** with **28 total variations**
- **Bimanual tasks**: LiftPot, StackSingleBookShelf, PickSingleBookFromTable, StackTwoBlocks, CubeHandover (including VerticalCubeHandover), RotateValve, PackBox, LiftTray, DragOverAndLiftTray
- **Bimanual Franka Panda** robot configuration
- **Data collection tools**: Oculus Quest VR and keyboard teleoperation
- **Comprehensive metrics**: Coordination, efficiency, safety, and task progression tracking

### Tasks

RoboEval includes **8 task families** with **28 total variations**:

1. **Lift Pot** (4 variants) - `lift_pot.py`
   - LiftPot, LiftPotPosition, LiftPotOrientation, LiftPotPositionAndOrientation

2. **Stack Single Book Shelf** (3 variants) - `stack_books.py`
   - StackSingleBookShelf, StackSingleBookShelfPosition, StackSingleBookShelfPositionAndOrientation

3. **Pick Single Book From Table** (4 variants) - `stack_books.py`
   - PickSingleBookFromTable, PickSingleBookFromTablePosition, PickSingleBookFromTableOrientation, PickSingleBookFromTablePositionAndOrientation

4. **Stack Two Blocks** (4 variants) - `manipulation.py`
   - StackTwoBlocks, StackTwoBlocksPosition, StackTwoBlocksOrientation, StackTwoBlocksPositionAndOrientation

5. **Cube Handover** (5 variants) - `manipulation.py`
   - CubeHandover, CubeHandoverPosition, CubeHandoverOrientation, CubeHandoverPositionAndOrientation, VerticalCubeHandover

6. **Rotate Valve** (3 variants) - `rotate_utility_objects.py`
   - RotateValve, RotateValvePosition, RotateValvePositionAndOrientation

7. **Pack Box** (4 variants) - `pack_objects.py`
   - PackBox, PackBoxPosition, PackBoxOrientation, PackBoxPositionAndOrientation

8. **Lift Tray** (5 variants) - `lift_tray.py`
   - LiftTray, LiftTrayPosition, LiftTrayOrientation, LiftTrayPositionAndOrientation, DragOverAndLiftTray

## Overview

RoboEval is a structured benchmark for bimanual manipulation, featuring diverse tasks with varying coordination and complexity. Unlike existing benchmarks that evaluate policies solely based on task success, RoboEval introduces an initial suite of tiered, semantically diverse manipulation tasks with fine-grained diagnostic metrics to probe the capabilities and failure modes of learning-based agents. Our benchmark provides 8 task families with 28 total variations that target specific skills such as coordination, precision, and interaction under variability, and are accompanied with 3,000+ total human-collected demonstrations. It additionally includes a standardized asset library—collision meshes, annotated sites, and manipulable objects—for building and augmenting tasks with spatial perturbations and distractors; a VR-based teleoperation interface enables realistic data collection; and rich evaluation tools that go beyond binary success, measuring task progression, coordination, trajectory efficiency, and spatial proximity.

For more information, please visit our [full documentation site](https://robo-eval.github.io/RoboEval-Documentation/)

## Installation

### Prerequisites
- Python 3.10+
- Git with submodule support
- CUDA-compatible GPU (recommended for model evaluation)

### Setup Environment

1. **Clone the repository with submodules:**
```bash
git clone --recurse-submodules git@github.com:helen9975/RoboEval.git
cd RoboEval
```

2. **Create and activate conda environment:**
```bash
conda create -n roboeval python=3.10
conda activate roboeval
```

3. **Install the package:**
```bash
# Basic installation
pip install -e .

# Install with example dependencies (recommended)
pip install -e ".[examples]"

# Install with VR support for teleoperation
pip install -e ".[vr]"

# Install development dependencies
pip install -e ".[dev]"
```

### Verify Installation
Test your installation by running a simple demo replay:
```bash
python examples/1_data_replay.py
```

## Getting Started

<details>
<summary><b>Click to expand: Examples Overview</b></summary>

The `examples/` directory contains several scripts demonstrating different aspects of RoboEval:

| Example | Description | Purpose |
|---------|-------------|---------|
| `1_data_replay.py` | Load and replay demonstrations from dataset | Basic demo loading and environment usage |
| `2_convert_and_replay.py` | Demo recording with action mode conversion | Understanding action modes and conversions |
| `3_load_convert_replay.py` | Load demos and convert between action modes | Advanced action mode handling |
| `4_eval_openvla.py` | Evaluate OpenVLA models on tasks | Model evaluation framework |
| `5_gather_metrics.py` | Collect and analyze task metrics | Metrics aggregation and analysis |
| `6_collect_data.py` | Data collection pipeline (keyboard) | Keyboard teleoperation demonstration collection |
| `7_collect_data_oculus.py` | Data collection pipeline (Oculus VR) | VR teleoperation demonstration collection |

</details>

### 1. Basic Demo Replay

Start with the simplest example to verify your setup:

```bash
python examples/replay_demo.py
```

This script:
- Automatically downloads demonstration datasets on first run
- Loads human-collected teleoperation demonstrations
- Replays them in the simulated environment with visual rendering
- Demonstrates basic environment and robot control

### 2. Understanding Action Modes

RoboEval supports different action modes. Learn about them with:

```bash
python examples/2_convert_and_replay.py
```

This example demonstrates:
- Joint position vs. end-effector control
- Absolute vs. delta (relative) actions
- Recording custom demonstrations
- Converting between action modes
- Trajectory visualization and comparison

### 3. Advanced Demo Conversion

For more complex action mode conversions:

```bash
python examples/3_load_convert_replay.py
```

Features:
- Loading demonstrations in one action mode
- Converting to different target action modes
- Handling lightweight vs. full observation modes
- Batch processing of multiple demonstrations

### 4. Model Evaluation

Evaluate pre-trained models (e.g., OpenVLA) on RoboEval tasks:

```bash
# Model inference mode
python examples/4_eval_openvla.py --ckpt_path /path/to/model/checkpoint

# Demo replay evaluation mode
python examples/4_eval_openvla.py --ckpt_path /path/to/model/checkpoint \
                                  --use_demos --dataset_path /path/to/demos

# Custom configuration
python examples/4_eval_openvla.py --ckpt_path /path/to/model/checkpoint \
                                  --instruction "pick up the book" \
                                  --num_episodes 10 --max_steps 300
```

### 5. Data Collection via Teleoperation

RoboEval supports two modes of teleoperation for collecting demonstrations:

#### Keyboard Teleoperation

Collect demonstrations using keyboard control (good for testing and simple data collection):

```bash
# Using keyboard teleoperation
cd roboeval
python data_collection/demo_recorder.py input_mode=Keyboard robot="Bimanual Panda" env="LiftPot"

# Or use the example script
python examples/6_collect_data.py
```

#### VR Teleoperation (Oculus Quest)

Collect high-quality demonstrations with immersive VR control for more natural bimanual manipulation:

```bash
# Using Oculus Quest VR teleoperation
python examples/7_collect_data_oculus.py

# Or use the demo recorder directly
cd roboeval
python data_collection/demo_recorder.py input_mode=VR robot="Bimanual Panda" env="LiftPot"
```

**VR Setup Requirements:**
- Oculus Quest headset (Quest 2, Quest Pro, or Quest 3)
- USB-C cable for connecting headset to computer
- Developer mode enabled on Oculus Quest
- VR dependencies installed: `pip install -e ".[vr]"`
- **System requirement**: GLIBC 2.32+ (Ubuntu 20.10+, Debian 11+, or equivalent)
  - Check your version: `ldd --version`
  - If you have an older system, use Docker or the direct `demo_recorder.py` script

**📖 For detailed VR setup instructions, including:**
- Step-by-step Oculus Quest configuration
- ADB installation and troubleshooting  
- Developer mode activation
- USB debugging authorization
- Complete VR controls reference

**See the comprehensive guide:** [`roboeval/data_collection/README.md`](roboeval/data_collection/README.md)

**Note on VR Compatibility**: The VR teleoperation requires PyOpenXR which has GLIBC 2.32+ dependency. If you encounter GLIBC compatibility issues, you can:
1. Use the keyboard teleoperation mode instead (`examples/6_collect_data.py`)
2. Run in a Docker container with Ubuntu 20.10+ base image
3. Use the direct `demo_recorder.py` script which may have better system compatibility

## Available Tasks and Environments

Each task comes with multiple variants focusing on different aspects:

- **Base Task**: Standard version of the task
- **Position**: Only position control (orientation fixed)
- **Orientation**: Only orientation control (position fixed)  
- **PositionAndOrientation**: Both position and orientation control

### Action Modes

RoboEval supports different action modes for flexible control:

- **Joint Position Mode**: Direct joint angle control
  - `absolute=True`: Specify target joint positions
  - `absolute=False`: Specify joint position deltas
- **End-Effector Mode**: Cartesian space control
  - `ee=True`: Control end-effector poses directly
  - Combined with absolute/delta for position specification

### Observation Modes

- **Full**: Complete observations including RGB images, depth, point clouds
- **Lightweight**: Minimal observations for faster training (joint positions, object poses)

### Robot Configurations

- **BimanualPanda**: Dual Franka Panda arms with parallel grippers
- Configurable degrees of freedom and control frequencies
- Support for floating base and custom joint configurations

## Configuration Examples

<details>
<summary><b>Click to expand: Environment Configuration</b></summary>

```python
from roboeval.envs.lift_pot import LiftPotPositionAndOrientation
from roboeval.action_modes import JointPositionActionMode
from roboeval.robots.configs.panda import BimanualPanda
from roboeval.utils.observation_config import ObservationConfig, CameraConfig

# Create environment with specific action mode
env = LiftPotPositionAndOrientation(
    action_mode=JointPositionActionMode(
        floating_base=True,
        absolute=True,  # Use absolute positions
        ee=False,       # Joint control (not end-effector)
        floating_dofs=[]
    ),
    render_mode="human",
    control_frequency=20,
    robot_cls=BimanualPanda,
    observation_config=ObservationConfig(
        cameras=[
            CameraConfig(
                name="external",
                rgb=True,
                depth=False,
                resolution=(128, 128),
                pos=[0.0, 10.0, 10.0]
            )
        ]
    )
)
```

</details>

<details>
<summary><b>Click to expand: Demo Loading and Conversion</b></summary>

```python
from roboeval.demonstrations.demo_store import DemoStore
from roboeval.demonstrations.demo_converter import DemoConverter
from roboeval.demonstrations.utils import Metadata

# Load demonstrations
metadata = Metadata.from_env(env)
demo_store = DemoStore()
demos = demo_store.get_demos(metadata, amount=10, frequency=20)

# Convert between action modes
for demo in demos:
    # Convert joint absolute to end-effector delta
    converted_demo = DemoConverter.joint_absolute_to_ee_delta(demo)
    
    # Convert absolute to delta positions
    delta_demo = DemoConverter.absolute_to_delta(demo)
    
    # Convert joint to end-effector control
    ee_demo = DemoConverter.joint_to_ee(demo)
```

</details>

## Troubleshooting

<details>
<summary><b>Click to expand: Common Issues and Solutions</b></summary>

### Common Issues

1. **MuJoCo Installation Problems**
   ```bash
   # Make sure you have the correct MuJoCo version
   pip install mujoco==3.1.5
   ```

2. **Display Issues (Headless Servers)**
   ```bash
   # Use virtual display
   export DISPLAY=:99
   Xvfb :99 -screen 0 1024x768x24 &
   ```

3. **CUDA/GPU Issues**
   ```bash
   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

4. **Demo Download Failures**
   - Check internet connection
   - Verify GitHub access for private repositories
   - Clear demo cache: `rm -rf ~/.roboeval/`

5. **Import Errors**
   ```bash
   # Reinstall in development mode
   pip install -e .
   
   # Check Python path
   python -c "import roboeval; print(roboeval.__file__)"
   ```

6. **VR/Oculus Quest Issues**
   - **GLIBC version error**: PyOpenXR requires GLIBC 2.32+
     ```bash
     # Check your GLIBC version
     ldd --version
     
     # If version is < 2.32, use alternatives:
     # - Keyboard teleoperation: python examples/6_collect_data.py
     # - Docker with Ubuntu 20.10+ image
     # - Direct demo_recorder.py script
     ```
   - **Quest not detected**: Ensure USB debugging is enabled and cable is connected
   - **Permission denied**: Run `adb devices` and accept prompt in headset

### Performance Tips

- Use `render_mode=None` for faster training/evaluation
- Reduce camera resolution for better performance
- Use lightweight observation mode when possible
- Adjust `control_frequency` based on your needs (higher = more precise, slower)

</details>

## Development Workflow

<details>
<summary><b>Click to expand: Development and Testing</b></summary>

### Running Tests
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run specific test
python test_metric_rollout.py
```

</details>

## Tasks

RoboEval provides a comprehensive suite of bimanual manipulation tasks designed to evaluate different aspects of robotic coordination and control. Each task has multiple variants that test specific capabilities.

### Task Variants

Each base task comes with up to 4 variants:
- **Base**: Full 6-DOF control (position + orientation)
- **Position**: Position-only control (orientation fixed)
- **Orientation**: Orientation-only control (position fixed) 
- **PositionAndOrientation**: Combined position and orientation control

### Available Tasks

| Task | Description |
|------|-------------|
| **LiftPot** | Grip kitchen pot by handles and lift above table |
| **LiftTray** | Grasp breakfast tray with both grippers and lift |
| **PackBox** | Close two-flap packing box using both arms |
| **PickSingleBookFromTable** | Grip and lift target book from table |
| **RotateValve** | Rotate valves counterclockwise |
| **StackSingleBookShelf** | Place book on shelf in contact |
| **StackTwoCubes** | Stack two cubes on table |
| **CubeHandover** | Pass cube between robot arms |

### Task Examples

```python
# Import available tasks
from roboeval.envs.lift_pot import LiftPot, LiftPotPosition, LiftPotOrientation, LiftPotPositionAndOrientation
from roboeval.envs.manipulation import StackTwoBlocks, StackTwoBlocksPosition, CubeHandover
from roboeval.envs.stack_books import PickSingleBookFromTable, StackSingleBookShelf
from roboeval.envs.pack_objects import PackBox, PackBoxPosition
from roboeval.envs.lift_tray import LiftTray, LiftTrayPosition
from roboeval.envs.rotate_utility_objects import RotateValve, RotateValvePosition

# Create a task instance
env = LiftPotPositionAndOrientation(
    action_mode=JointPositionActionMode(floating_base=True, absolute=True),
    render_mode="human",
    control_frequency=20,
    robot_cls=BimanualPanda
)
```

### Evaluation Metrics

RoboEval goes beyond binary success metrics with comprehensive evaluation:

#### Primary Metrics
- **Task Success Rate**: Binary completion of primary objective
- **Partial Success**: Credit for partial task completion
- **Semantic Progress**: Task-specific milestone achievement

#### Diagnostic Metrics  
- **Trajectory Efficiency**: Path optimality and smoothness
- **Coordination Quality**: Synchronization between arms
- **Spatial Precision**: Accuracy of positioning and orientation
- **Safety Violations**: Collision and constraint violations

#### Example Evaluation
```python
from roboeval.envs.lift_pot import LiftPotPositionAndOrientation
from roboeval.demonstrations.demo_player import DemoPlayer

# Load environment and demo
env = LiftPotPositionAndOrientation(...)
demo = demo_store.get_demos(metadata, amount=1)[0]

# Replay and evaluate
player = DemoPlayer()
metrics = player.replay_in_env(demo, env, return_metrics=True)

print(f"Success Rate: {metrics['success_rate']}")
print(f"Trajectory Efficiency: {metrics['trajectory_efficiency']}")
print(f"Coordination Score: {metrics['coordination_quality']}")
```

### Detailed Metrics System

<details>
<summary><b>Click to expand: Comprehensive Metrics Documentation</b></summary>

RoboEval includes a comprehensive metrics tracking system (`MetricRolloutEval`) that provides fine-grained evaluation beyond binary success metrics. Environments can inherit from this class to enable detailed performance analysis.

#### Enabling Metrics in Your Environment

To enable metrics tracking, initialize the metric system in your environment's `_initialize_env` method:

```python
from roboeval.utils.metric_rollout import MetricRolloutEval

class MyTask(RoboEvalEnv, MetricRolloutEval):
    def _initialize_env(self):
        # Initialize your environment objects
        self.object = SomeObject(self._mojo)
        
        # Initialize metrics tracking
        self._metric_init(
            track_vel_sync=True,              # Track velocity synchronization
            track_vertical_sync=True,          # Track vertical alignment
            track_slippage=True,               # Track object slippage
            slip_objects=[self.object],        # Objects to monitor for slippage
            slip_sample_window=20,             # Frames between slip checks
            track_collisions=True,             # Track collision events
            track_cartesian_jerk=True,         # Track end-effector smoothness
            track_joint_jerk=True,             # Track joint smoothness
            track_cartesian_path_length=True,  # Track cartesian distance
            track_joint_path_length=True,      # Track joint space distance
            track_orientation_path_length=True, # Track orientation changes
            robot=self._robot                  # Robot instance
        )
    
    def _on_step(self):
        # Update metrics each step
        self._metric_step()
    
    def _success(self) -> bool:
        # Your success condition
        return self.object.position[2] > 1.0
    
    def get_info(self):
        info = super().get_info()
        if self.success or self.terminate:
            # Finalize metrics at episode end
            metrics = self._metric_finalize(
                success_flag=self.success,
                target_distance=self.target_distance,  # Optional
                pose_error=self.pose_error             # Optional
            )
            info["metrics"] = metrics
        return info
```

#### Available Metrics

##### 1. **Coordination Metrics** (Bimanual Tasks)

- **`bimanual_arm_velocity_difference`**: Average difference in joint velocities between left and right arms
  - **Lower is better** - indicates better synchronized movement
  - Measured as L2-norm of velocity difference
  - Units: rad/s

- **`bimanual_gripper_vertical_difference`**: Average vertical (Z-axis) height difference between grippers
  - **Lower is better** - indicates better height coordination
  - Units: meters
  - Useful for tasks requiring parallel lifting (e.g., LiftPot, LiftTray)

##### 2. **Collision Metrics**

- **`env_collision_count`**: Number of new collision events with the environment
  - Counts unique collision events (not contact duration)
  - Excludes target objects being manipulated
  - **Lower is better** - indicates safer execution

- **`self_collision_count`**: Number of robot self-collision events
  - Detects when robot parts collide with each other
  - **Lower is better** - indicates better motion planning

##### 3. **Trajectory Smoothness Metrics**

**Cartesian Jerk** (End-Effector Space):
- **`avg_cartesian_jerk`**: Average jerk magnitude in cartesian space
  - Jerk = rate of change of acceleration (m/s³)
  - **Lower is better** - smoother end-effector motion
  - Per-arm dictionary for bimanual: `{"left": 0.5, "right": 0.6}`

- **`rms_cartesian_jerk`**: Root mean square cartesian jerk
  - More sensitive to large jerk spikes than average
  - Better indicator of motion smoothness

- **`overall_avg_cartesian_jerk`** / **`overall_rms_cartesian_jerk`**: Combined metrics for bimanual robots

**Joint Jerk** (Joint Space):
- **`avg_joint_jerk`**: Average jerk in joint space (rad/s³)
- **`rms_joint_jerk`**: RMS joint jerk
- **`overall_avg_joint_jerk`** / **`overall_rms_joint_jerk`**: Combined bimanual metrics

##### 4. **Path Length Metrics**

**Cartesian Path Length**:
- **`cartesian_path_length`**: Total distance traveled by end-effector(s)
  - Per-arm for bimanual: `{"left": 1.2, "right": 1.5}`
  - Units: meters
  - Useful for evaluating trajectory efficiency

- **`total_cartesian_path_length`**: Sum of both arms (bimanual only)
- **`avg_cartesian_path_length`**: Average across arms (bimanual only)

**Joint Path Length**:
- **`joint_path_length`**: Total distance in joint space
  - Per-arm for bimanual
  - Units: radians
  - Indicates joint space efficiency

- **`total_joint_path_length`** / **`avg_joint_path_length`**: Combined bimanual metrics

**Orientation Path Length**:
- **`orientation_path_length`**: Total orientation change (quaternion angular distance)
  - Per-arm for bimanual
  - Units: radians
  - Measures rotational efficiency

- **`total_orientation_path_length`** / **`avg_orientation_path_length`**: Combined bimanual metrics

##### 5. **Object Manipulation Metrics**

- **`slip_count`**: Total number of slip events detected
  - Slip = object was held but gripper opened while moving
  - **Lower is better** - indicates stable grasping
  - Detection frequency controlled by `slip_sample_window`

- **`slip_count_per_object`**: Slip events per tracked object
  - Dictionary: `{"object_1": 0, "object_2": 1, ...}`
  - Useful for multi-object tasks

##### 6. **Task Progress Metrics**

- **`success`**: Binary task completion (0.0 or 1.0)

- **`completion_time`**: Wall-clock time to complete episode
  - Units: seconds
  - Includes rendering time

- **`subtask_progress`**: Percentage of subtask stages completed
  - Range: [0.0, 1.0]
  - Calculated from `task_stage_reached` flags
  - Useful for partial credit in failed attempts

- **`task_stage_reached`**: Boolean flags for each subtask stage
  - Dictionary: `{1: True, 2: True, 3: False, ...}`
  - Set via `self._metric_stage(stage_idx, success=True)` in environment code

##### 7. **Spatial Accuracy Metrics**

- **`target_distance`**: Final distance to target position(s)
  - Can be single float or dictionary for multiple targets
  - Units: meters
  - **Lower is better**

- **`object_pose_error`**: Pose error of manipulated object(s)
  - Combined position and orientation error
  - Can be single float or dictionary
  - **Lower is better**

#### Tracking Subtask Progress

For complex tasks with multiple stages, track intermediate progress:

```python
class MultiStageTask(RoboEvalEnv, MetricRolloutEval):
    def _on_step(self):
        self._metric_step()
        
        # Check and record subtask completion
        if self.object_grasped and not self.get_metric_stage(1):
            self._metric_stage(1, success=True)  # Stage 1: grasp
        
        if self.object_lifted and not self.get_metric_stage(2):
            self._metric_stage(2, success=True)  # Stage 2: lift
        
        if self.object_placed and not self.get_metric_stage(3):
            self._metric_stage(3, success=True)  # Stage 3: place
```

#### Example Metrics Output

```python
{
    "success": 1.0,
    "completion_time": 12.4,
    "subtask_progress": 1.0,
    "task_stage_reached": {1: True, 2: True, 3: True},
    
    # Coordination
    "bimanual_arm_velocity_difference": 0.05,
    "bimanual_gripper_vertical_difference": 0.008,
    
    # Collisions
    "env_collision_count": 0,
    "self_collision_count": 0,
    
    # Smoothness
    "avg_cartesian_jerk": {"left": 0.42, "right": 0.38},
    "rms_cartesian_jerk": {"left": 0.65, "right": 0.58},
    "overall_avg_cartesian_jerk": 0.40,
    "overall_rms_cartesian_jerk": 0.62,
    
    # Path efficiency
    "cartesian_path_length": {"left": 1.23, "right": 1.18},
    "total_cartesian_path_length": 2.41,
    "joint_path_length": {"left": 3.45, "right": 3.52},
    "orientation_path_length": {"left": 0.87, "right": 0.92},
    
    # Manipulation
    "slip_count": 0,
    "slip_count_per_object": {"object_1": 0},
    
    # Accuracy
    "target_distance": 0.012,
    "object_pose_error": 0.034
}
```

#### Performance Considerations

- **Slip Detection Window**: Set `slip_sample_window` to balance detection accuracy vs. computation
  - Higher values (20-30): Less frequent checks, faster
  - Lower values (5-10): More sensitive, higher overhead

- **Selective Tracking**: Only enable metrics you need for your evaluation
  - Full tracking has minimal overhead (~5-10% performance impact)
  - Collision tracking is most expensive

- **Jerk Calculation**: Requires computing numerical derivatives
  - Automatically handles timestep from `control_frequency`
  - More accurate with higher control frequencies

</details>

| Task                                                                                                                  | Description                                                                                                                | Preview                                                                                                         |
| --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| [LiftPot](https://robo-eval.github.io/RoboEval-Documentation/tasks/lift-pot.html)                             | Grip the kitchen pot by its handles and raise it above the table.                                                          | <img src="https://robo-eval.github.io/RoboEval-Documentation/_images/lift_pot.png" width=320>                |
| [LiftTray](https://robo-eval.github.io/RoboEval-Documentation/tasks/lift-tray.html)                           | Grasp the breakfast tray with the two grippers and lift it clear of the source table.                                      | <img src="https://robo-eval.github.io/RoboEval-Documentation/_images/lift_tray.png" width=320>               |
| [PackBox](https://robo-eval.github.io/RoboEval-Documentation/tasks/pack-box.html)                             | Have each arm interact with the two-flap packing box and close both flaps until the opening is fully covered.              | <img src="https://robo-eval.github.io/RoboEval-Documentation/_images/pack_box.png" width=320>                |
| [PickSingleBookFromTable](https://robo-eval.github.io/RoboEval-Documentation/tasks/stack-books.html) | Grip the target book on the table and lift it up.                                                                          | <img src="https://robo-eval.github.io/RoboEval-Documentation/_images/pick_single_book.png" width=320>        |
| [RotateValve](https://robo-eval.github.io/RoboEval-Documentation/tasks/rotate-valve.html)                     | Rotate each valve counterclockwise.                                                                                        | <img src="https://robo-eval.github.io/RoboEval-Documentation/_images/rotate_valve.png" width=320>            |
| [StackSingleBookShelf](https://robo-eval.github.io/RoboEval-Documentation/tasks/stack-books.html) | Pick up the book from the table and place it in contact with one of the shelves.                                           | <img src="https://robo-eval.github.io/RoboEval-Documentation/_images/stack_single_book_shelf.png" width=320> |
| [StackTwoBlocks](https://robo-eval.github.io/RoboEval-Documentation/tasks/manipulation.html)               | Manipulate two cubes placed on the table so that they are stacked.                                                         | <img src="https://robo-eval.github.io/RoboEval-Documentation/_images/stack_two_blocks.png" width=320>        |
| [CubeHandover](https://robo-eval.github.io/RoboEval-Documentation/tasks/manipulation.html)                 | Pass a cube between the robot's two arms.                                                                                  | <img src="https://robo-eval.github.io/RoboEval-Documentation/_images/cube_handover.png" width=320>        |

## Contributing

<details>
<summary><b>Click to expand: Contribution Guidelines</b></summary>

We welcome contributions to RoboEval! Here's how you can help:

### Adding New Tasks
1. Create task environment in `roboeval/envs/`
2. Follow existing task structure and naming conventions
3. Implement variants (Position, Orientation, PositionAndOrientation)
4. Add comprehensive evaluation metrics
5. Include demonstration data collection

### Reporting Issues
- Use GitHub Issues for bug reports and feature requests
- Include reproduction steps and environment details
- Check existing issues before creating new ones

### Development Guidelines
- Follow PEP 8 style guidelines
- Write comprehensive tests for new features
- Document all public APIs
- Use pre-commit hooks for code quality

### Pull Request Process
1. Fork the repository
2. Create feature branch from `main`
3. Make changes with tests and documentation
4. Run pre-commit checks and tests
5. Submit pull request with clear description

</details>

## Citation

If you use RoboEval in your research, please cite our paper:

```bibtex
@misc{wang2025roboevalroboticmanipulationmeets,
      title={RoboEval: Where Robotic Manipulation Meets Structured and Scalable Evaluation}, 
      author={Yi Ru Wang and Carter Ung and Grant Tannert and Jiafei Duan and Josephine Li and Amy Le and Rishabh Oswal and Markus Grotz and Wilbert Pumacay and Yuquan Deng and Ranjay Krishna and Dieter Fox and Siddhartha Srinivasa},
      year={2025},
      eprint={2507.00435},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2507.00435}, 
}
```

## Licenses

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Components
- MuJoCo: Uses MuJoCo physics simulator (Apache 2.0 License)
- BiGym: Builds upon BiGym framework components
- Mujoco Menagerie: Includes models from Mujoco Menagerie (Apache 2.0 License)

### Acknowledgments
Special thanks to:
- The BiGym team for the foundational bimanual manipulation framework
- MuJoCo team for the physics simulation engine
- The open-source robotics community for tools and inspiration

## Support

- **Documentation**: [Full Documentation Site](https://robo-eval.github.io/RoboEval-Documentation/)
- **Issues**: [GitHub Issues](https://github.com/helen9975/RoboEval/issues)
- **Discussions**: [GitHub Discussions](https://github.com/helen9975/RoboEval/discussions)
- **Email**: Contact the authors via the paper

---

*For the latest updates and detailed documentation, visit our [documentation site](https://robo-eval.github.io/RoboEval-Documentation/).*
