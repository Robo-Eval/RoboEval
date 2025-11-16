import mujoco as mj
import numpy as np

from mojo import Mojo
from mojo.elements import Site


# Taken from Claude prompting
def check_obb_intersection(pos1, rot1, size1, pos2, rot2, size2):
    """
    Check if two oriented bounding boxes (OBBs) intersect in 3D space.

    Parameters:
    - pos1: np.ndarray of shape (3,), center position of first box
    - rot1: np.ndarray of shape (3, 3), rotation matrix of first box
    - size1: np.ndarray of shape (3,), half-extents of first box
    - pos2: np.ndarray of shape (3,), center position of second box
    - rot2: np.ndarray of shape (3, 3), rotation matrix of second box
    - size2: np.ndarray of shape (3,), half-extents of second box

    Returns:
    - bool: True if boxes intersect, False otherwise
    """
    # Get the vectors from box1 to box2
    translation = pos2 - pos1

    # Get the rotation axes of both boxes
    axes1 = [rot1[:, i] for i in range(3)]
    axes2 = [rot2[:, i] for i in range(3)]

    # We need to test 15 potential separating axes:
    # - 3 from box1's face normals
    # - 3 from box2's face normals
    # - 9 from cross products of edges (3x3)

    # First, test the 3 axes from box1
    for i in range(3):
        axis = axes1[i]

        # Project the translation vector onto the current axis
        t_proj = np.dot(translation, axis)

        # Project half-extents of box1 onto the current axis
        r1 = size1[i]

        # Project half-extents of box2 onto the current axis
        r2 = 0
        for j in range(3):
            r2 += size2[j] * abs(np.dot(axes2[j], axis))

        # If the projection of the translation is greater than the sum of half-extents,
        # we found a separating axis, so the boxes don't intersect
        if abs(t_proj) > r1 + r2:
            return False

    # Next, test the 3 axes from box2
    for i in range(3):
        axis = axes2[i]

        # Project the translation vector onto the current axis
        t_proj = np.dot(translation, axis)

        # Project half-extents of box2 onto the current axis
        r2 = size2[i]

        # Project half-extents of box1 onto the current axis
        r1 = 0
        for j in range(3):
            r1 += size1[j] * abs(np.dot(axes1[j], axis))

        # Check for separation
        if abs(t_proj) > r1 + r2:
            return False

    # Finally, test the 9 axes from cross products of edges
    for i in range(3):
        for j in range(3):
            # Cross product of axes from both boxes
            axis = np.cross(axes1[i], axes2[j])

            # Skip if the cross product is near zero (parallel edges)
            axis_length_sq = np.dot(axis, axis)
            if axis_length_sq < 1e-10:
                continue

            # Normalize the axis
            axis = axis / np.sqrt(axis_length_sq)

            # Project the translation vector onto the current axis
            t_proj = np.dot(translation, axis)

            # Project half-extents of both boxes onto the current axis
            r1 = 0
            for k in range(3):
                r1 += size1[k] * abs(np.dot(axes1[k], axis))

            r2 = 0
            for k in range(3):
                r2 += size2[k] * abs(np.dot(axes2[k], axis))

            # Check for separation
            if abs(t_proj) > r1 + r2:
                return False

    # If no separating axis was found, the boxes intersect
    return True


def check_sites_intersection(
    model: mj.MjModel,
    data: mj.MjData,
    first_site_id: int,
    second_site_id: int,
) -> bool:
    first_pos = data.site_xpos[first_site_id]
    first_rot = data.site_xmat[first_site_id].reshape(3, 3)
    first_size = model.site_size[first_site_id]

    second_pos = data.site_xpos[second_site_id]
    second_rot = data.site_xmat[second_site_id].reshape(3, 3)
    second_size = model.site_size[second_site_id]

    return check_obb_intersection(
        first_pos, first_rot, first_size, second_pos, second_rot, second_size
    )


def check_sites_intersection_mojo(
    mojo: Mojo,
    first_site: Site,
    second_site: Site,
) -> bool:
    first_pos = mojo.physics.bind(first_site.mjcf).xpos.copy()
    first_rot = mojo.physics.bind(first_site.mjcf).xmat.copy().reshape(3, 3)
    first_size = mojo.physics.bind(first_site.mjcf).size.copy()

    second_pos = mojo.physics.bind(second_site.mjcf).xpos.copy()
    second_rot = mojo.physics.bind(second_site.mjcf).xmat.copy().reshape(3, 3)
    second_size = mojo.physics.bind(second_site.mjcf).size.copy()

    return check_obb_intersection(
        first_pos, first_rot, first_size, second_pos, second_rot, second_size
    )
