from __future__ import annotations

import abc

import mujoco as mj
import numpy as np

from mojo import Mojo
from mojo.elements import Body, Site


class IBoundingBox(abc.ABC):
    """Interface for Axis-Aligned Bounding Boxes"""

    def __init__(self):
        self.min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
        self.max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)

    @abc.abstractmethod
    def update(self) -> None:
        pass

    def intersects(self, other: IBoundingBox) -> bool:
        return np.all(self.min <= other.max) and np.all(self.max >= other.min)


class SiteBoundingBox(IBoundingBox):
    """Class associated with bounding box given by a site"""

    def __init__(self, mojo: Mojo, site: Site):
        super().__init__()
        self.mojo = mojo
        self.site = site

    def update(self) -> None:
        site_type = self.mojo.physics.bind(self.site.mjcf).type
        assert site_type == mj.mjtGeom.mjGEOM_BOX, "Bounding box site must be a box"

        site_pos = self.mojo.physics.bind(self.site.mjcf).xpos
        site_rot = self.mojo.physics.bind(self.site.mjcf).xmat.reshape(3, 3)
        site_size = self.mojo.physics.bind(self.site.mjcf).size

        points = np.array(
            [
                [-site_size[0], -site_size[1], -site_size[2]],
                [site_size[0], -site_size[1], -site_size[2]],
                [-site_size[0], site_size[1], -site_size[2]],
                [site_size[0], site_size[1], -site_size[2]],
                [-site_size[0], -site_size[1], site_size[2]],
                [site_size[0], -site_size[1], site_size[2]],
                [-site_size[0], site_size[1], site_size[2]],
                [site_size[0], site_size[1], site_size[2]],
            ]
        )
        new_points = np.dot(points, site_rot.T) + site_pos
        self.min = np.min(new_points, axis=0)
        self.max = np.max(new_points, axis=0)


class BodyBoundingBox(IBoundingBox):
    """Class associated with bounding box given by a body"""

    def __init__(self, mojo: Mojo, body: Body):
        super().__init__()
        self.mojo = mojo
        self.body = body

    def update(self) -> None:
        for geom in self.body.geoms:
            pos = self.mojo.physics.bind(geom.mjcf).xpos
            rot = self.mojo.physics.bind(geom.mjcf).xmat.reshape(3, 3)
            aabb = self.mojo.physics.bind(geom.mjcf).aabb
            center, size = aabb[:3], aabb[3:]
            points = np.array(
                [
                    [-size[0], -size[1], -size[2]],
                    [size[0], -size[1], -size[2]],
                    [-size[0], size[1], -size[2]],
                    [size[0], size[1], -size[2]],
                    [-size[0], -size[1], size[2]],
                    [size[0], -size[1], size[2]],
                    [-size[0], size[1], size[2]],
                    [size[0], size[1], size[2]],
                ]
            )
            breakpoint()
            new_points = np.dot(points + center, rot.T) + pos
            self.min = np.minimum(self.min, new_points)
            self.max = np.maximum(self.max, new_points)
