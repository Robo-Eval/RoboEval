"""Shared entities."""
from enum import Enum
from pathlib import Path
from typing import Type, Union, Callable, Optional, Dict

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.envs.lift_pot import LiftPot, LiftPotPosition, LiftPotOrientation, LiftPotPositionAndOrientation
from roboeval.envs.manipulation import (
    CubeHandover,
    CubeHandoverPosition,
    CubeHandoverOrientation,
    CubeHandoverPositionAndOrientation,
    VerticalCubeHandover,
    StackTwoBlocks,
    StackTwoBlocksPosition,
    StackTwoBlocksOrientation,
)

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

from roboeval.envs.lift_tray import (
    LiftTray,
    LiftTrayPosition,
    LiftTrayOrientation,
    LiftTrayPositionAndOrientation,
    DragOverAndLiftTray
)
from roboeval.envs.rotate_utility_objects import (
    RotateValve,
    RotateValvePosition, 
    RotateValvePositionAndOrientation,
    RotateValveObstacle
)
from roboeval.robots.robot import Robot
from roboeval.demonstrations.const import SAFETENSORS_SUFFIX

# Lazy import for dearpygui - only loaded when GUI functions are called
# This avoids GLIBC compatibility issues when only using ENVIRONMENTS dict
try:
    from dearpygui import dearpygui as dpg
    DEARPYGUI_AVAILABLE = True
except (ImportError, OSError) as e:
    DEARPYGUI_AVAILABLE = False
    dpg = None

from roboeval.robots.configs.panda import BimanualPanda, create_bimanual_panda_config, SinglePanda, create_single_panda_config
from roboeval.data_collection.teleop import Teleop


class ReplayMode(Enum):
    """Enum controlling joint position mode during demo replay."""

    Absolute = 0
    Delta = 1


DATA_COLLECTION_MODES: Dict[str, Callable[[], Type[Teleop]]] = {
    "VR": lambda: __import__("roboeval.data_collection.vr_input").data_collection.vr_input.VRTeleop,
    "Oculus": lambda: __import__("roboeval.data_collection.oculus_input").data_collection.oculus_input.OculusTeleop,
    "Keyboard": lambda: __import__("roboeval.data_collection.keyboard_input").data_collection.keyboard_input.KeyboardTeleop,
    "MotionPlanning": lambda: __import__("roboeval.data_collection.mp_input").data_collection.mp_input.MotionPlanningTeleop,
}

REPLAY_MODES: dict[str, ReplayMode] = {
    "Absolute": ReplayMode.Absolute,
    "Delta": ReplayMode.Delta,
}

ENVIRONMENTS: dict[str, Type[RoboEvalEnv]] = {
    "Pack Box": PackBox,
    "Pack Box Rotation": PackBoxOrientation,
    "Pack Box Position": PackBoxPosition,
    "Pack Box Position and Rotation": PackBoxPositionAndOrientation,
    "Stack Single Book Shelf": StackSingleBookShelf,
    "Stack Single Book Shelf Position": StackSingleBookShelfPosition,
    "Stack Single Book Shelf Position and Orientation": StackSingleBookShelfPositionAndOrientation,
    "Pick Single Book From Table": PickSingleBookFromTable,
    "Pick Single Book From Table Position": PickSingleBookFromTablePosition,
    "Pick Single Book From Table Orientation": PickSingleBookFromTableOrientation,
    "Pick Single Book From Table Position and Orientation": PickSingleBookFromTablePositionAndOrientation,
    "Lift Tray": LiftTray,
    "Lift Tray Position": LiftTrayPosition,
    "Lift Tray Orientation": LiftTrayOrientation,
    "Lift Tray Position and Orientation": LiftTrayPositionAndOrientation,
    "Drag Over and Lift Tray": DragOverAndLiftTray,
    "Cube Handover": CubeHandover,
    "Cube Handover Position": CubeHandoverPosition,
    "Cube Handover Orientation": CubeHandoverOrientation,
    "Cube Handover Position and Orientation": CubeHandoverPositionAndOrientation,
    "Vertical Cube Handover": VerticalCubeHandover,
    "Stack Two Blocks": StackTwoBlocks,
    "Stack Two Blocks Position": StackTwoBlocksPosition,
    "Stack Two Blocks Orientation": StackTwoBlocksOrientation,
    "Lift Pot": LiftPot,
    "Lift Pot Position":LiftPotPosition,
    "Lift Pot Orientation": LiftPotOrientation,
    "Lift Pot Position and Orientation": LiftPotPositionAndOrientation,
    "Rotate Valve": RotateValve,
    "Rotate Valve Random Position": RotateValvePosition,
    "Rotate Valve Random Position and Orientation": RotateValvePositionAndOrientation,
    "Rotate Valve Obstacle": RotateValveObstacle,
}

ROBOTS: dict[str, Optional[Type[Robot]]] = {
    "Default": None,
    "Bimanual Panda": BimanualPanda,
    "Single Panda": SinglePanda,
}

IK_CONFIGS: dict[str, dict] = {
    "Default": create_bimanual_panda_config(),
    "Bimanual Panda": create_bimanual_panda_config(),
    "Single Panda": create_single_panda_config(),
}


def get_demos_in_dir(directory: Path) -> list[Path]:
    """Get all demonstrations files in directory."""
    demos = list(directory.glob(f"*{SAFETENSORS_SUFFIX}"))
    return sorted(demos)


def select_directory(default_path: Union[Path, str], callback: Callable[[Path], None]):
    """Show directory selection dialog."""
    if not DEARPYGUI_AVAILABLE:
        raise ImportError("dearpygui is not available. This function requires dearpygui.")
    
    with dpg.file_dialog(
        modal=True,
        show=True,
        directory_selector=True,
        default_path=default_path,
        callback=lambda _, app_data: callback(Path(app_data["file_path_name"])),
        width=850,
        height=400,
    ):
        dpg.add_file_extension(".*")


def show_popup(
    header: str = "",
    message: str = "",
    actions: dict[str, Optional[Callable]] = None,
    loading_indicator: bool = False,
) -> int:
    """Show Popup."""
    if not DEARPYGUI_AVAILABLE:
        raise ImportError("dearpygui is not available. This function requires dearpygui.")

    def popup_callback(sender, app_data, user_data):
        popup_item = user_data["popup"]
        callback_action = user_data["action"]
        if callback_action:
            callback_action()
        dpg.delete_item(popup_item)

    def center():
        dpg.split_frame()
        window_width = dpg.get_viewport_width()
        window_height = dpg.get_viewport_height()
        popup_width, popup_height = dpg.get_item_rect_size(popup)
        x = (window_width - popup_width) / 2
        y = (window_height - popup_height) / 2
        dpg.set_item_pos(popup, [x, y])

    with dpg.window(
        label=header,
        modal=True,
        show=True,
        popup=True,
        no_resize=True,
        min_size=(400, 100),
        autosize=True,
        no_close=True,
    ) as popup:
        if message:
            dpg.add_text(message)
        if loading_indicator:
            dpg.add_loading_indicator(style=1)
        with dpg.group(horizontal=True):
            actions = actions or {}
            for label, action in actions.items():
                dpg.add_button(
                    label=label,
                    user_data={"popup": popup, "action": action},
                    callback=popup_callback,
                )
    center()
    return popup
