"""Retrieve and replay demonstration from the dataset.

Notes:
    - On the first run, the latest demos are downloaded and saved locally.
    - Demos are re-recorded at `control_frequency` and cached locally on the first run.
"""


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


from roboeval.action_modes import JointPositionActionMode
from roboeval.robots.configs.panda import BimanualPanda
from roboeval.utils.observation_config import ObservationConfig, CameraConfig
import numpy as np

from roboeval.demonstrations.demo_player import DemoPlayer
from roboeval.demonstrations.demo_store import DemoStore
from roboeval.demonstrations.utils import Metadata

control_frequency = 20

env = PickSingleBookFromTablePositionAndOrientation(
    action_mode=JointPositionActionMode(floating_base=True, absolute=True, floating_dofs=[]),
    render_mode="human",
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
demos = demo_store.get_demos(metadata, amount=100, frequency=control_frequency)

# Replay first demonstration
player = DemoPlayer()
for demo in demos:
    print(f"Replaying demo: {demo.uuid}")
    # Replay the demo in the environment
    player.replay_in_env(demo, env, demo_frequency=control_frequency)
