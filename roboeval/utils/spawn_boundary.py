from enum import Enum
from typing import List

import mujoco as mj
import numpy as np

from mojo import Mojo
from mojo.elements import Site

from roboeval.envs.props.prop import Prop


class BoundaryCode(Enum):
    NO_ERROR = 0
    ERROR_CANT_FIT = 1
    ERROR_COLLISION = 2
    ERROR_MIN_DISTANCE = 3


class BoundaryObject:
    def __init__(self, mojo: Mojo, site: Site):
        self.mojo = mojo
        self.site = site
        self._contained_objects: List[Prop] = []

        self._min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
        self._max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
        self._compute_bounds()

    @property
    def min(self) -> np.ndarray:
        return self._min

    @property
    def max(self) -> np.ndarray:
        return self._max

    def add_no_sample(self, obj: Prop) -> None:
        obj.bbox.update()
        self._contained_objects.append(obj)

    def add(
        self,
        obj: Prop,
        ignore_collisions: bool = False,
        min_distance: float = 0.01,
    ) -> BoundaryCode:
        obj.bbox.update()

        # Check that the object can fit in this boundary
        boundary_width = np.abs(self._max[0] - self._min[0])
        boundary_depth = np.abs(self._max[1] - self._min[1])
        object_width = np.abs(obj.bbox.max[0] - obj.bbox.min[0])
        object_depth = np.abs(obj.bbox.max[1] - obj.bbox.min[1])
        if boundary_width < object_width or boundary_depth < object_depth:
            return BoundaryCode.ERROR_CANT_FIT

        _, _, z = obj.body.get_position()
        x, y = self._get_position_within_boundary(obj)
        new_pos = np.array([x, y, z])
        obj.set_pose(position=new_pos)
        obj.bbox.update()

        if not ignore_collisions:
            for contained_obj in self._contained_objects:
                contained_obj.bbox.update()
                if obj.bbox.intersects(contained_obj.bbox):
                    return BoundaryCode.ERROR_COLLISION
                dist = np.linalg.norm(new_pos - contained_obj.body.get_position())
                if dist < min_distance:
                    return BoundaryCode.ERROR_MIN_DISTANCE
            self._contained_objects.append(obj)
        return BoundaryCode.NO_ERROR

    def _compute_bounds(self) -> None:
        site_type = self.mojo.physics.bind(self.site.mjcf).type
        site_size = self.mojo.physics.bind(self.site.mjcf).size
        site_pos = self.mojo.physics.bind(self.site.mjcf).xpos
        site_rot = self.mojo.physics.bind(self.site.mjcf).xmat.reshape(3, 3)

        assert site_type == mj.mjtGeom.mjGEOM_BOX, "Boundary site must be a box"

        points = np.array(
            [
                [-site_size[0], -site_size[1], -site_size[2]],
                [site_size[0], -site_size[1], -site_size[2]],
                [site_size[0], site_size[1], -site_size[2]],
                [-site_size[0], site_size[1], -site_size[2]],
                [-site_size[0], -site_size[1], site_size[2]],
                [site_size[0], -site_size[1], site_size[2]],
                [site_size[0], site_size[1], site_size[2]],
                [-site_size[0], site_size[1], site_size[2]],
            ]
        )

        new_points = np.dot(points, site_rot.T) + site_pos

        self._min = np.min(new_points, axis=0)
        self._max = np.max(new_points, axis=0)

    def _get_position_within_boundary(self, obj: Prop) -> List[float]:
        width = np.abs(obj.bbox.max[0] - obj.bbox.min[0])
        depth = np.abs(obj.bbox.max[1] - obj.bbox.min[1])
        x = np.random.uniform(
            self._min[0] + width / 2,
            self._max[0] - width / 2,
        )
        y = np.random.uniform(
            self._min[1] + depth / 2,
            self._max[1] - depth / 2,
        )
        return [x, y]

    def clear(self) -> None:
        self._contained_objects.clear()


class SpawnBoundary:
    """Spawn boundary for objects."""

    MAX_SAMPLES = 100

    def __init__(self, mojo: Mojo, site: Site):
        self.mojo = mojo
        self.boundary = BoundaryObject(mojo, site)

    def add_no_sample(self, prop: Prop) -> None:
        self.boundary.add_no_sample(prop)

    def sample(
        self,
        prop: Prop,
        ignore_collisions: bool = False,
        min_distance: float = 0.01,
    ) -> None:
        n_collision_fails = n_boundary_fails = SpawnBoundary.MAX_SAMPLES
        while n_collision_fails > 0 and n_boundary_fails > 0:
            error_code = self.boundary.add(
                prop,
                ignore_collisions,
                min_distance,
            )
            if error_code == BoundaryCode.ERROR_CANT_FIT:
                n_boundary_fails -= 1
            elif error_code == BoundaryCode.ERROR_COLLISION:
                n_collision_fails -= 1
            elif error_code == BoundaryCode.ERROR_MIN_DISTANCE:
                n_boundary_fails -= 1
            else:
                break
        if n_boundary_fails <= 0:
            raise RuntimeError(
                "Cannot fit object in boundary, maybe object is too big for it?"
            )
        elif n_collision_fails <= 0:
            raise RuntimeError(
                "Cannot spawn object without collision with other objects"
            )

    def clear(self) -> None:
        self.boundary.clear()
