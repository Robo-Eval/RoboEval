"""Bookshelf class"""
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

import numpy as np
from dm_control import mjcf
from mojo.elements import Body, MujocoElement, Joint, Site

from roboeval.const import ASSETS_PATH
from roboeval.envs.props.prop import CollidableProp
from roboeval.utils.physics_utils import set_joint_position, get_joint_position


class BookShelf(CollidableProp, ABC):
    '''
    Given an imported XML file, we want to import any articulated shelf into this modular class.
    The modular class should enable the following
    1. Define the joints after initialization
    2. Have setters and getters to the set the state of the joints
    3. Have options to ennable and disable specified parts based on args
    '''
    _LOWER_SHELF_SITE  = "lower_shelf_site"
    _UPPER_SHELF_SITE = "upper_shelf_site"
    _LOWER_SHELF = "lower_shelf"
    _UPPER_SHELF = "upper_shelf"
    _COUNTER = "counter"

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / self.asset_path

    def _post_init(self):
        self._joints = self.body.joints

    def set_state(self, state: Optional[float] = None, target_state: Optional[Dict[str, float]] = None):
        """
        Set the state of the joints.

        Args:
            state (Optional[float]): A single normalized value to set all joints to. If None, `target_state` must be provided.
            target_state (Optional[Dict[str, float]]): A dictionary mapping joint names to their desired normalized values.
        """
        if not self._joints:
            if state is not None or (target_state and len(target_state) > 0):
                print("[WARNING] set_state called but no joints available.")
            return

        if target_state:
            for joint_name, value in target_state.items():
                # Fetch the joint reference
                joint = (Joint.get(self._mojo, joint_name, self.body))
                # print(f'{joint_name} is being loaded')
                if joint and hasattr(joint.mjcf, 'range'):
                    set_joint_position(joint, value, True)
                else:
                    print(f"[WARNING] Joint '{joint_name}' not found in {self.asset_path}")
        elif state is not None:
            for joint in self.all_joints:
                if joint.range is None:
                    continue

                parent = joint.root.model
                # print(joint.root.model)
                joint_name = f'{parent}/{joint.name}'

                joint_obj = (Joint.get(self._mojo, joint_name, self.body))
                if joint_obj and hasattr(joint_obj.mjcf, 'range') and len(joint.range) > 0:
                    # print(joint.range)
                    # print(f'{joint_name} is being loaded')
                    set_joint_position(joint_obj, state, True)
        else:
            print("[WARNING] set_state called with neither 'state' nor 'target_state'. Nothing to do.")

    def get_state(self) -> np.ndarray[float]:
        """Get normalized state of joints."""
        if not self._joints:
            return np.array([])
        return np.array([get_joint_position(joint, True) for joint in self._joints])

    def _parse_kwargs(self, kwargs: dict[str, Any]):
        self.asset_path =  kwargs.get("asset_path", "props/box/box.xml") # specify the customizable asset path as input
        self.bodies_to_disable: list[str] = kwargs.get("bodies_to_disable", [])
        self.target_joints: list[str] = kwargs.get("target_joints", [])

    def _on_loaded(self, model: mjcf.RootElement): 
        shelf = MujocoElement(self._mojo, model)   
        self.lower_shelf_site = Site.get(self._mojo, self._LOWER_SHELF_SITE, shelf)
        self.upper_shelf_site = Site.get(self._mojo, self._UPPER_SHELF_SITE, shelf)
        self.lower_shelf_body = Body.get(self._mojo, self._LOWER_SHELF, shelf).geoms[-1]
        self.upper_shelf_body = Body.get(self._mojo, self._UPPER_SHELF, shelf).geoms[-1]
        self.counter = Body.get(self._mojo, self._COUNTER, shelf).geoms[-1]
          # Disable (remove) specific bodies if requested
        for body_name in self.bodies_to_disable:
            body_elem = model.find("body", body_name)
            if body_elem is not None:
                body_elem.remove()

        # Collect references to *all* bodies, joints, and sites
        self.all_bodies = model.find_all("body")
        self.all_joints = model.find_all("joint")
        self.all_sites  = model.find_all("site")