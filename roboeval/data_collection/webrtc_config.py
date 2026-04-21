"""WebRTC task configuration module, defining task configurations using Python classes and inheritance.

Adapted for telearms-roboplayground: only includes TaskConfigs whose names exist
in the playground's ENVIRONMENTS dict (see tools/shared/utils.py).
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from tools.shared.utils import ENVIRONMENTS


@dataclass
class TaskConfig:
    """Base task configuration class"""
    name: str
    description: str
    enabled: bool = True
    robot: str = "Bimanual Panda"

    def __post_init__(self):
        if self.name not in ENVIRONMENTS:
            raise ValueError(f"Task '{self.name}' is not in available environments list")


class BookTaskConfig(TaskConfig):
    def __init__(self, name: str, description: str = None, enabled: bool = False):
        base = "Book manipulation task: Move books from specified positions to target positions"
        super().__init__(name=name, description=description or base, enabled=enabled)


class PotTaskConfig(TaskConfig):
    def __init__(self, name: str, description: str = None, enabled: bool = True):
        base = "Pot manipulation task: Control pots to perform specified actions"
        super().__init__(name=name, description=description or base, enabled=enabled)


class TrayTaskConfig(TaskConfig):
    def __init__(self, name: str, description: str = None, enabled: bool = False):
        base = "Tray manipulation task: Balance and move trays"
        super().__init__(name=name, description=description or base, enabled=enabled)


class CubeTaskConfig(TaskConfig):
    def __init__(self, name: str, description: str = None, enabled: bool = False):
        base = "Cube manipulation task: Stack and move cubes"
        super().__init__(name=name, description=description or base, enabled=enabled)


class ValveTaskConfig(TaskConfig):
    def __init__(self, name: str, description: str = None, enabled: bool = True):
        base = "Valve manipulation task: Rotate valves to specified positions"
        super().__init__(name=name, description=description or base, enabled=enabled)


class PackingTaskConfig(TaskConfig):
    def __init__(self, name: str, description: str = None, enabled: bool = False):
        base = "Packing task: Place or remove items from containers"
        super().__init__(name=name, description=description or base, enabled=enabled)


# Task configuration mapping — all keys MUST exist in ENVIRONMENTS.
TASK_CONFIGS: Dict[str, TaskConfig] = {
    # Pot
    "Lift Pot": PotTaskConfig("Lift Pot", "Lift pot"),
    "Lift Pot Position": PotTaskConfig("Lift Pot Position", enabled=False),
    "Lift Pot Orientation": PotTaskConfig("Lift Pot Orientation", enabled=False),
    "Lift Pot Position and Orientation": PotTaskConfig("Lift Pot Position and Orientation", enabled=False),

    # Valve
    "Rotate Valve": ValveTaskConfig("Rotate Valve", "Rotate valve"),
    "Rotate Valve Random Position": ValveTaskConfig("Rotate Valve Random Position", enabled=False),
    "Rotate Valve Random Position and Orientation": ValveTaskConfig(
        "Rotate Valve Random Position and Orientation", enabled=False
    ),
    "Rotate Valve Obstacle": ValveTaskConfig("Rotate Valve Obstacle", "Rotate valve with obstacle", enabled=False),

    # Books
    "Stack Single Book Shelf": BookTaskConfig("Stack Single Book Shelf"),
    "Stack Single Book Shelf Position": BookTaskConfig("Stack Single Book Shelf Position"),
    "Stack Single Book Shelf Position and Orientation": BookTaskConfig(
        "Stack Single Book Shelf Position and Orientation"
    ),
    "Pick Single Book From Table": BookTaskConfig("Pick Single Book From Table", "Pick a single book from table"),
    "Pick Single Book From Table Position": BookTaskConfig("Pick Single Book From Table Position"),
    "Pick Single Book From Table Orientation": BookTaskConfig("Pick Single Book From Table Orientation"),
    "Pick Single Book From Table Position and Orientation": BookTaskConfig(
        "Pick Single Book From Table Position and Orientation"
    ),

    # Tray
    "Lift Tray": TrayTaskConfig("Lift Tray", "Lift tray"),
    "Lift Tray Position": TrayTaskConfig("Lift Tray Position"),
    "Lift Tray Orientation": TrayTaskConfig("Lift Tray Orientation"),
    "Lift Tray Position and Orientation": TrayTaskConfig("Lift Tray Position and Orientation"),
    "Drag Over and Lift Tray": TrayTaskConfig("Drag Over and Lift Tray", "Drag over and lift tray"),

    # Cubes
    "Cube Handover": CubeTaskConfig("Cube Handover", "Cube handover"),
    "Cube Handover Position": CubeTaskConfig("Cube Handover Position"),
    "Cube Handover Orientation": CubeTaskConfig("Cube Handover Orientation"),
    "Cube Handover Position and Orientation": CubeTaskConfig("Cube Handover Position and Orientation"),
    "Vertical Cube Handover": CubeTaskConfig("Vertical Cube Handover", "Vertical cube handover"),
    "Stack Two Blocks": CubeTaskConfig("Stack Two Blocks", "Stack two blocks"),
    "Stack Two Blocks Position": CubeTaskConfig("Stack Two Blocks Position"),
    "Stack Two Blocks Orientation": CubeTaskConfig("Stack Two Blocks Orientation"),

    # Packing
    "Pack Box": PackingTaskConfig("Pack Box", "Pack box"),
    "Pack Box Rotation": PackingTaskConfig("Pack Box Rotation"),
    "Pack Box Position": PackingTaskConfig("Pack Box Position"),
    "Pack Box Position and Rotation": PackingTaskConfig("Pack Box Position and Rotation"),
}


def get_available_tasks() -> List[TaskConfig]:
    return [c for c in TASK_CONFIGS.values() if c.enabled]


def get_random_task() -> TaskConfig:
    import random
    available = get_available_tasks()
    if not available:
        raise ValueError("No available task configurations")
    return random.choice(available)


def get_task_by_name(name: str) -> Optional[TaskConfig]:
    return TASK_CONFIGS.get(name)


def get_task_description(name: str) -> str:
    config = get_task_by_name(name)
    return config.description if config else "Unknown task"


def get_next_task(current_task: Optional[TaskConfig] = None) -> TaskConfig:
    available = get_available_tasks()
    if not available:
        raise ValueError("No available task configurations")
    if current_task is None:
        return available[0]
    try:
        idx = available.index(current_task)
        return available[(idx + 1) % len(available)]
    except ValueError:
        return available[0]
