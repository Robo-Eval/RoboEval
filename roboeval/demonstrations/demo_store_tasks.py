"""Task registry and metadata helpers for DemoStore-style joint-absolute demos."""

from __future__ import annotations

from roboeval.action_modes import JointPositionActionMode
from roboeval.demonstrations.utils import Metadata, ObservationMode

from roboeval.envs.lift_pot import (
    LiftPot, LiftPotPosition, LiftPotOrientation, LiftPotPositionAndOrientation,
)
from roboeval.envs.manipulation import (
    CubeHandover, CubeHandoverPosition, CubeHandoverOrientation,
    CubeHandoverPositionAndOrientation, VerticalCubeHandover,
    StackTwoBlocks, StackTwoBlocksPosition, StackTwoBlocksOrientation,
    StackTwoBlocksPositionAndOrientation,
)
from roboeval.envs.stack_books import (
    StackSingleBookShelf, StackSingleBookShelfPosition,
    StackSingleBookShelfPositionAndOrientation,
    PickSingleBookFromTable, PickSingleBookFromTablePosition,
    PickSingleBookFromTableOrientation, PickSingleBookFromTablePositionAndOrientation,
)
from roboeval.envs.pack_objects import (
    PackBox, PackBoxOrientation, PackBoxPosition, PackBoxPositionAndOrientation,
)
from roboeval.envs.lift_tray import (
    LiftTray, LiftTrayPosition, LiftTrayOrientation,
    LiftTrayPositionAndOrientation, DragOverAndLiftTray,
)
from roboeval.envs.rotate_utility_objects import (
    RotateValve, RotateValvePosition, RotateValvePositionAndOrientation,
    RotateValveObstacle,
)

# Same layout as public DemoStore / replay_sample_check_success
DEMO_STORE_TASKS: dict[str, type] = {
    "LiftPot": LiftPot,
    "LiftPotPosition": LiftPotPosition,
    "LiftPotOrientation": LiftPotOrientation,
    "LiftPotPositionAndOrientation": LiftPotPositionAndOrientation,
    "CubeHandover": CubeHandover,
    "CubeHandoverPosition": CubeHandoverPosition,
    "CubeHandoverOrientation": CubeHandoverOrientation,
    "CubeHandoverPositionAndOrientation": CubeHandoverPositionAndOrientation,
    "VerticalCubeHandover": VerticalCubeHandover,
    "StackTwoBlocks": StackTwoBlocks,
    "StackTwoBlocksPosition": StackTwoBlocksPosition,
    "StackTwoBlocksOrientation": StackTwoBlocksOrientation,
    "StackTwoBlocksPositionAndOrientation": StackTwoBlocksPositionAndOrientation,
    "StackSingleBookShelf": StackSingleBookShelf,
    "StackSingleBookShelfPosition": StackSingleBookShelfPosition,
    "StackSingleBookShelfPositionAndOrientation": StackSingleBookShelfPositionAndOrientation,
    "PickSingleBookFromTable": PickSingleBookFromTable,
    "PickSingleBookFromTablePosition": PickSingleBookFromTablePosition,
    "PickSingleBookFromTableOrientation": PickSingleBookFromTableOrientation,
    "PickSingleBookFromTablePositionAndOrientation": PickSingleBookFromTablePositionAndOrientation,
    "PackBox": PackBox,
    "PackBoxOrientation": PackBoxOrientation,
    "PackBoxPosition": PackBoxPosition,
    "PackBoxPositionAndOrientation": PackBoxPositionAndOrientation,
    "LiftTray": LiftTray,
    "LiftTrayPosition": LiftTrayPosition,
    "LiftTrayOrientation": LiftTrayOrientation,
    "LiftTrayPositionAndOrientation": LiftTrayPositionAndOrientation,
    "DragOverAndLiftTray": DragOverAndLiftTray,
    "RotateValve": RotateValve,
    "RotateValvePosition": RotateValvePosition,
    "RotateValvePositionAndOrientation": RotateValvePositionAndOrientation,
    "RotateValveObstacle": RotateValveObstacle,
}


def joint_light_joint_abs_metadata(env_cls: type) -> Metadata:
    """Lightweight joint-absolute metadata matching released demo paths."""
    return Metadata.from_env_cls(
        env_cls=env_cls,
        action_mode=JointPositionActionMode,
        floating_dofs=[],
        obs_mode=ObservationMode.Lightweight,
        action_mode_absolute=True,
        end_effector_mode=False,
    )
