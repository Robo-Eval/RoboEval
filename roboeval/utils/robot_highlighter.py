"""Highlight actuated joints of the robot."""
import warnings

import numpy as np
from dm_control import mjcf
from mojo import Mojo
from mojo.elements import Geom

from roboeval.robots.robot import Robot


class RobotHighlighter:
    """Highlight actuated joints of the robot."""

    def __init__(self, robot: Robot, mojo: Mojo):
        """Init."""
        self._actuated_geoms: list[list[Geom]] = []
        self._original_colors: dict[mjcf.Element, np.ndarray] = {}
        for actuator in robot.limb_actuators:
            joints = []
            if actuator.joint:
                joints.append(actuator.joint)
            if actuator.tendon:
                for tendon_joint in actuator.tendon.joint:
                    joints.append(tendon_joint.joint)
            geoms = []
            for joint in joints:
                parent = joint.parent
                geoms = parent.find_all("geom", immediate_children_only=True) or []
                geoms.extend(geoms)
            self._actuated_geoms.append([Geom(mojo, g) for g in set(geoms)])
        for _, gripper in robot.grippers.items():
            self._actuated_geoms.append(gripper.body.geoms)

        for geoms in self._actuated_geoms:
            for geom in geoms:
                self._original_colors[geom.mjcf] = geom.get_color()

    def reset(self):
        """Clean highlight."""
        for geoms in self._actuated_geoms:
            for geom in geoms:
                geom.set_color(self._original_colors[geom.mjcf])

    def highlight(self, index: int, tint: np.ndarray = np.array([1, 1, 1, 1])):
        """Highlight joint by index."""
        allowed_range = range(len(self._actuated_geoms))
        if index not in allowed_range:
            warnings.warn(f"Index {index} out of {allowed_range} range.")
            return
        for geom in self._actuated_geoms[index]:
            geom.set_color(self._original_colors[geom.mjcf] + tint)
