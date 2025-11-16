"""Modular cabinets."""
from abc import ABC
from pathlib import Path
from typing import Any

import numpy as np
from dm_control import mjcf
from mojo.elements import Body, MujocoElement, Site

from roboeval.const import ASSETS_PATH
from roboeval.envs.props.prop import CollidableProp
from roboeval.utils.physics_utils import set_joint_position, get_joint_position


class ModularCabinet(CollidableProp, ABC):
    """Base modular prop."""

    def _post_init(self):
        self._joints = self.body.joints

    def set_state(self, state: np.ndarray):
        """Set normalized state of joints."""
        for value, joint in zip(state, self._joints):
            set_joint_position(joint, value, True)

    def get_state(self) -> np.ndarray[float]:
        """Get normalized state of joints."""
        return np.array([get_joint_position(joint, True) for joint in self._joints])

    @staticmethod
    def _toggle_body(model: mjcf.RootElement, name: str, enable: bool):
        if not enable:
            model.find("body", name).remove()


class BaseCabinet(ModularCabinet):
    """Modular base cabinet."""

    _BIG_DRAWERS = ["drawer_big_1", "drawer_big_2"]
    _SMALL_DRAWERS = [
        "drawer_small_1",
        "drawer_small_2",
        "drawer_small_3",
        "drawer_small_4",
    ]
    _DOOR_LEFT = "door_left"
    _DOOR_RIGHT = "door_right"
    _PANEL = "panel"
    _SHELF = "shelf"
    _SHELF_BOTTOM = "shelf_bottom"
    _HOB = "hob"
    _WALLS = "walls"
    _COUNTER = "counter"
    _TARGET_SITE = "target_site"

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/kitchen/base_cabinet_600.xml"

    def _parse_kwargs(self, kwargs: dict[str, Any]):
        self._walls_enable = kwargs.get("walls_enable", True)
        self._big_drawers_enable = kwargs.get("big_drawers_enable", None)
        self._small_drawers_enable = kwargs.get("small_drawers_enable", None)
        self._hob_enable = kwargs.get("hob_enable", False)
        self._panel_enable = kwargs.get("panel_enable", False)
        self._door_left_enable = kwargs.get("door_left_enable", False)
        self._door_right_enable = kwargs.get("door_right_enable", False)
        self._shelf_enable = kwargs.get("shelf_enable", False)

        if self._big_drawers_enable is None:
            self._big_drawers_enable = [False] * len(self._BIG_DRAWERS)
        if self._small_drawers_enable is None:
            self._small_drawers_enable = [False] * len(self._SMALL_DRAWERS)

        assert len(self._big_drawers_enable) == len(self._BIG_DRAWERS)
        assert len(self._small_drawers_enable) == len(self._SMALL_DRAWERS)

        self._CACHE_SITES = (
            self._hob_enable
            or any(self._big_drawers_enable)
            or any(self._small_drawers_enable)
        )

    def _on_loaded(self, model: mjcf.RootElement):
        cabinet = MujocoElement(self._mojo, model)
        self.shelf = Body.get(self._mojo, self._SHELF, cabinet).geoms[-1]
        self.shelf_bottom = Body.get(self._mojo, self._SHELF_BOTTOM, cabinet).geoms[-1]
        self.counter = Body.get(self._mojo, self._COUNTER, cabinet).geoms[-1]
        self.hob = Body.get(self._mojo, self._HOB, cabinet).geoms[-1]

        for name, enable in zip(self._BIG_DRAWERS, self._big_drawers_enable):
            self._toggle_body(model, name, enable)
        for name, enable in zip(self._SMALL_DRAWERS, self._small_drawers_enable):
            self._toggle_body(model, name, enable)
        self._toggle_body(model, self._WALLS, self._walls_enable)
        self._toggle_body(model, self._HOB, self._hob_enable)
        self._toggle_body(model, self._PANEL, self._panel_enable)
        self._toggle_body(model, self._DOOR_LEFT, self._door_left_enable)
        self._toggle_body(model, self._DOOR_RIGHT, self._door_right_enable)
        self._toggle_body(model, self._SHELF, self._shelf_enable)


class WallCabinet(ModularCabinet):
    """Modular wall cabinet."""

    _DOORS = ["door_right", "door_left"]
    _GLASS_DOORS = ["door_right_glass", "door_left_glass"]
    _VENT = "vent"
    _SHELF = "shelf"
    _SHELF_BOTTOM = "shelf_bottom"

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/kitchen/wall_cabinet_600.xml"

    def _parse_kwargs(self, kwargs: dict[str, Any]):
        self._doors_enable = kwargs.get("doors_enable", False)
        self._glass_doors_enable = kwargs.get("glass_doors_enable", False)
        self._vent_enable = kwargs.get("vent_enable", False)

    def _on_loaded(self, model: mjcf.RootElement):
        cabinet = MujocoElement(self._mojo, model)
        self.shelf = Body.get(self._mojo, self._SHELF, cabinet).geoms[-1]
        self.shelf_bottom = Body.get(self._mojo, self._SHELF_BOTTOM, cabinet).geoms[-1]

        for door_name in self._DOORS:
            self._toggle_body(model, door_name, self._doors_enable)
        for door_name in self._GLASS_DOORS:
            self._toggle_body(model, door_name, self._glass_doors_enable)
        self._toggle_body(model, self._VENT, self._vent_enable)


class OpenShelf(ModularCabinet):
    """Modular open shelf."""

    _SHELF = "shelf"

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/kitchen/open_shelf_600.xml"

    def _on_loaded(self, model: mjcf.RootElement):
        shelf = MujocoElement(self._mojo, model)
        self.shelf = Body.get(self._mojo, self._SHELF, shelf).geoms[-1]
