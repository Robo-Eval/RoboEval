"""Props Preset."""
from collections import defaultdict
from pathlib import Path
from typing import Optional, Callable, TypeVar, Type

import numpy as np
from mojo import Mojo
from yaml import safe_load

import roboeval.envs.props as props_module
from roboeval.envs.props.prop import Prop
from roboeval.utils.shared import find_class_in_module

PT = TypeVar("PT", bound=Prop)


class Preset:
    """Props Preset."""

    def __init__(self, mojo: Mojo, path: Optional[Path]):
        """Init."""
        self._props: list[Prop] = []
        self._props_lookup: dict[type[Prop], list[Prop]] = defaultdict(list)

        if path is None:
            return
        with open(path) as f:
            config = safe_load(f)
        for include in config.get("include", []):
            sub_layout = Preset(mojo, path.parent / include)
            self._props.extend(sub_layout._props)
        for prop_config in config.get("props", []):
            prop_and_children = self._load_prop(prop_config, mojo)
            self._props.extend(prop_and_children)
        # Build props lookup table by prop type
        for prop in self._props:
            self._props_lookup[type(prop)].append(prop)

    def get_props(self, prop_type: Optional[Type[PT]] = None) -> list[PT]:
        """Get preset props."""
        if prop_type is None:
            return self._props
        else:
            return self._props_lookup.get(prop_type, [])

    def _load_prop(
        self, config, mojo: Mojo, parent: Optional[Prop] = None
    ) -> list[Prop]:
        loaded_props = []
        # Parse config
        prop_type = config.pop("type")
        prop_cls: Optional[type[Prop]] = find_class_in_module(props_module, prop_type)
        if not prop_cls:
            return loaded_props
        position = self._get_float_array(config.pop("position", None), 3)
        euler = self._get_float_array(config.pop("euler", None), 3, np.deg2rad)
        children = config.pop("children", [])
        # Load prop
        if parent is None:
            prop = prop_cls(mojo, **config)
        else:
            parent_site_name = config.pop("parent_site")
            parent_site = next(
                (site for site in parent.sites if site.mjcf.name == parent_site_name),
                None,
            )
            if parent_site is None:
                raise ValueError(
                    f"Site with name '{parent_site_name}' not found in {parent}"
                )
            prop = prop_cls(mojo, parent=parent_site, **config)
        loaded_props.append(prop)
        # Set pose
        if position is not None:
            prop.body.set_position(position)
        if euler is not None:
            prop.body.set_euler(euler)
        # Load children
        for child_config in children:
            loaded_props.extend(self._load_prop(child_config, mojo, prop))
        return loaded_props

    @staticmethod
    def _get_float_array(
        value: Optional[list],
        target_length: int,
        post_process: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Optional[np.ndarray]:
        if value is None:
            return None
        if len(value) != target_length:
            raise ValueError(
                f"Incorrect array length: {value}, expected length: {target_length}"
            )
        result = np.array([float(item) for item in value])
        if post_process:
            result = post_process(result)
        return result
