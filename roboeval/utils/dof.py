"""Degree of freedom description."""
from dataclasses import dataclass
from typing import Tuple, Optional

from mojo.elements.consts import JointType


@dataclass(frozen=True)
class Dof:
    """Degree of freedom description."""

    joint_type: JointType
    axis: Tuple[int, int, int]
    joint_range: Optional[Tuple[float, float]] = None
    action_range: Optional[Tuple[float, float]] = None
    stiffness: Optional[float] = 0
