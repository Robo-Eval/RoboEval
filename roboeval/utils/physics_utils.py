"""Physics utils."""
from __future__ import annotations

import math
import warnings
from typing import Iterable, Union, Any

import numpy as np
from dm_control import mjcf
from mojo.elements import MujocoElement, Geom, Joint


def get_critical_damping_from_stiffness(
    stiffness: float, joint_name: str, physics: mjcf.Physics
) -> float:
    """Compute the critical damping coefficient for a given stiffness.

    Args:
        stiffness: The stiffness coefficient.
        joint_name: The name of the joint to compute the critical damping for.
        physics: The mujoco Physics instance.

    Returns:
        The critical damping coefficient.
    """
    joint_id = physics.named.model.jnt_qposadr[joint_name]
    joint_mass = physics.model.dof_M0[joint_id]
    return 2 * math.sqrt(joint_mass * stiffness)


def is_target_reached(actuator: mjcf.Element, physics: mjcf.Physics, tolerance: float):
    """Check if the target state of the actuator is reached."""
    qpos = float(physics.bind(actuator.joint).qpos.item())
    ctrl = float(physics.bind(actuator).ctrl.item())
    if actuator.ctrlrange is not None:
        ctrl = np.clip(ctrl, *actuator.ctrlrange)
    return np.abs(qpos - ctrl) <= tolerance


def distance(element1: MujocoElement, element2: MujocoElement):
    """Distance between 2 Mujoco Elements."""
    return np.linalg.norm(element1.get_position() - element2.get_position())


# ToDo: Move to Mojo
def set_joint_position(joint: Joint, value: float, normalized: bool = False):
    """Set Mojo Joint position."""
    if normalized:
        value = np.clip(value, 0, 1)
        value = np.interp(value, [0, 1], joint.mjcf.range)
    bound_joint = joint._mojo.physics.bind(joint.mjcf)
    bound_joint.qpos = value
    bound_joint.qvel *= 0
    bound_joint.qacc *= 0


# ToDo: Move to Mojo
def get_joint_position(joint: Joint, normalized: bool = False) -> float:
    """Get Mojo Joint position."""
    value = joint.get_joint_position()
    if not normalized:
        return value
    joint_range = joint.mjcf.range
    return (value - joint_range[0]) / (joint_range[0] + joint_range[1])


def get_actuator_qpos(actuator: mjcf.Element, physics: mjcf.Physics) -> float:
    """Get qpos for actuator with joint or tendon."""
    if actuator.joint:
        return float(physics.bind(actuator.joint).qpos.item())
    elif actuator.tendon:
        return float(physics.bind(actuator).ctrl.item())
    else:
        warnings.warn(f"Actuator {actuator} is not supported.")


def get_actuator_qvel(actuator: mjcf.Element, physics: mjcf.Physics) -> float:
    """Get qvel for actuator with joint or tendon."""
    if actuator.joint:
        return float(physics.bind(actuator.joint).qvel.item())
    elif actuator.tendon:
        velocities = []
        for tendon_joint in actuator.tendon.joint:
            joint = physics.bind(tendon_joint.joint)
            velocities.append(joint.qvel.item())
        return np.average(velocities)
    else:
        warnings.warn(f"Actuator {actuator} is not supported.")


def get_colliders(obj: Union[Geom, Iterable[Geom], Any]) -> Iterable[Geom]:
    """Get list of collider geometries from arbitrary object."""
    if isinstance(obj, Geom):
        return [obj]
    elif isinstance(obj, Iterable):
        return obj
    elif hasattr(obj, "colliders"):
        return obj.colliders
    else:
        warnings.warn(f"Unknown type of object to get colliders!\n{obj}")
        return []


# Default minimum distance between two geoms for them to be considered in collision.
_DEFAULT_COLLISION_MARGIN: float = 1e-8


def has_collided_collections(
    physics: mjcf.Physics,
    colliders_1: Union[Iterable[Geom], Geom],
    colliders_2: Union[Iterable[Geom], Geom],
) -> bool:
    """Check collision between two collections of colliders."""
    if isinstance(colliders_1, Geom):
        colliders_1 = [colliders_1]
    if isinstance(colliders_2, Geom):
        colliders_2 = [colliders_2]
    ids_1 = set(physics.bind([c.mjcf for c in colliders_1]).element_id)
    ids_2 = set(physics.bind([c.mjcf for c in colliders_2]).element_id)

    for contact in physics.data.contact:
        if contact.dist > _DEFAULT_COLLISION_MARGIN:
            continue
        if (contact.geom1 in ids_1 and contact.geom2 in ids_2) or (
            contact.geom2 in ids_1 and contact.geom1 in ids_2
        ):
            return True
    return False
