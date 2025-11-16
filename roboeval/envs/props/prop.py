"""Abstract prop."""
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Union, Iterable, Optional, Any

import numpy as np
from dm_control import mjcf
from mojo import Mojo
from mojo.elements import Body, Geom, Site, MujocoElement
from pyquaternion import Quaternion

from roboeval.utils.physics_utils import has_collided_collections, get_colliders
from roboeval.utils.bounding_box import IBoundingBox, SiteBoundingBox, BodyBoundingBox


class Prop(ABC):
    """Base prop."""

    _KINEMATIC = False
    _CACHE_COLLIDERS = False
    _CACHE_SITES = False

    __HIDDEN_POSITION = np.array([0, 0, -100])

    def __init__(
        self,
        mojo: Mojo,
        kinematic: Optional[bool] = None,
        cache_colliders: Optional[bool] = None,
        cache_sites: Optional[bool] = None,
        parent: Optional[MujocoElement] = None,
        **kwargs,
    ):
        """Init."""
        self._parse_kwargs(kwargs or {})
        kinematic = kinematic or self._KINEMATIC
        cache_colliders = cache_colliders or self._CACHE_COLLIDERS
        cache_sites = cache_sites or self._CACHE_SITES

        self._mojo = mojo
        self.body: Body = self._mojo.load_model(
            str(self._model_path),
            on_loaded=self._on_loaded,
            parent=parent,
        )
        self.aabb: Optional[Site] = self.get_boundary_aabb(self.body, self._mojo)
        self.bbox: IBoundingBox = self.get_bounding_box(self.body, self._mojo)

        # Cache colliders
        self.colliders: list[Geom] = []
        if cache_colliders:
            self.colliders = self.get_body_colliders(self.body)
        # Cache sites
        self.sites: list[Site] = []
        if cache_sites:
            self.sites = self.get_body_sites(self.body, self._mojo)
        # Set kinematic state
        self.kinematic = kinematic
        if self.kinematic:
            self.body.set_kinematic(True)
        # Cache original geom settings
        self._geoms = self.body.geoms
        self._geoms_settings_cache: dict[mjcf.Element, (int, int)] = {}
        for geom in self._geoms:
            self._geoms_settings_cache[geom.mjcf] = (
                self._mojo.physics.bind(geom.mjcf).contype,
                self._mojo.physics.bind(geom.mjcf).conaffinity,
            )
        self._post_init()

    @property
    @abstractmethod
    def _model_path(self) -> Path:
        raise NotImplementedError

    def _parse_kwargs(self, kwargs: dict[str, Any]):
        """Process initialization kwargs."""
        pass

    def _on_loaded(self, model: mjcf.RootElement):
        """Callback to customize prop model."""
        pass

    def _post_init(self):
        """Customize prop initialization."""
        pass

    def get_pose(self) -> np.ndarray:
        """Get pose in the world space."""
        return np.concatenate(
            (self.body.get_position(), self.body.get_quaternion()),
            axis=-1,
        )

    def set_pose(
        self,
        position: np.ndarray = np.zeros(3),
        quat: np.ndarray = Quaternion().elements,
        position_bounds: np.ndarray = np.zeros(3),
        rotation_bounds: np.ndarray = np.zeros(3),
    ):
        """Set pose in the world space."""
        offset_pos = np.random.uniform(-position_bounds, position_bounds)
        pos = position + offset_pos

        offset_rot = np.random.uniform(-rotation_bounds, rotation_bounds)
        quat = (
            Quaternion(quat)
            * Quaternion(axis=[1, 0, 0], angle=offset_rot[0])
            * Quaternion(axis=[0, 1, 0], angle=offset_rot[1])
            * Quaternion(axis=[0, 0, 1], angle=offset_rot[2])
        )

        self.body.set_position(pos, True)
        self.body.set_quaternion(quat.elements, True)

    def disable(self):
        """Disable prop."""
        for geom in self._geoms:
            geom = self._mojo.physics.bind(geom.mjcf)
            geom.contype = 0
            geom.conaffinity = 0
        if self.body.is_kinematic():
            freejoint = self._mojo.physics.bind(self.body.mjcf.freejoint)
            freejoint.damping = 10e6
        self.body.set_position(self.__HIDDEN_POSITION, True)

    def enable(self):
        """Enable prop."""
        for geom in self._geoms:
            contype, conaffinity = self._geoms_settings_cache[geom.mjcf]
            geom = self._mojo.physics.bind(geom.mjcf)
            geom.contype = contype
            geom.conaffinity = conaffinity

        if self.body.is_kinematic():
            freejoint = self._mojo.physics.bind(self.body.mjcf.freejoint)
            freejoint.damping = 0

    def get_velocities(self) -> np.ndarray:
        """Get velocities of the free body."""
        if self.kinematic:
            return np.array(self._mojo.physics.bind(self.body.mjcf.freejoint).qvel)
        else:
            return np.zeros(0)

    def is_static(self, atol_pos: float = 1.0e-3):
        """Check if object's position is static."""
        velocities = self.get_velocities()
        return np.allclose(velocities[:3], 0, atol=atol_pos)

    def is_colliding(self, other: Union[Geom, Iterable[Geom], "Prop"]) -> bool:
        """Check collision between two props."""
        other_colliders = get_colliders(other)
        return has_collided_collections(
            self._mojo.physics, self.colliders, other_colliders
        )

    @staticmethod
    def get_body_colliders(body: Body) -> list[Geom]:
        """Get all colliders of the body."""
        return [g for g in body.geoms if g.is_collidable()]

    @staticmethod
    def get_body_sites(body: Body, mojo: Mojo) -> list[Site]:
        """Get all sites of the body."""
        sites = body.mjcf.find_all("site")
        return [Site(mojo, site_mjcf) for site_mjcf in sites]

    @staticmethod
    def get_boundary_aabb(body: Body, mojo: Mojo) -> Optional[Site]:
        """Get an AABB site if any"""
        sites = Prop.get_body_sites(body, mojo)
        for site in sites:
            if site.mjcf.name == "boundary":
                return site
        return None

    @staticmethod
    def get_bounding_box(body: Body, mojo: Mojo) -> IBoundingBox:
        aabb = Prop.get_boundary_aabb(body, mojo)
        if aabb is not None:
            return SiteBoundingBox(mojo, aabb)
        return BodyBoundingBox(mojo, body)


class CollidableProp(Prop, ABC):
    """Collidable prop."""

    _CACHE_COLLIDERS = True


class KinematicProp(Prop, ABC):
    """Kinematic collidable prop."""

    _KINEMATIC = True
    _CACHE_COLLIDERS = True
