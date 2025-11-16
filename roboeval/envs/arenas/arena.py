from pathlib import Path
from typing import Dict, Optional
from yaml import safe_load

import numpy as np
from mojo import Mojo
from mojo.elements import Body

ARENAS_FOLDER = Path(__file__).parent.parent / "xmls" / "arenas"


class Arena:
    def __init__(self, mojo: Mojo, path: Optional[Path] = None):
        self._mojo: Mojo = mojo
        self._name: str = ""
        self._objects: Dict[str, Body] = {}

        if path is None:
            return

        with open(path) as fhandle:
            config = safe_load(fhandle)
            if "arena" not in config:
                return
            self._name = config["arena"]["name"]
            for item_cfg in config["arena"]["items"]:
                item_name = item_cfg["name"]
                item_xml = item_cfg["xml"]
                item_path = str(
                    (ARENAS_FOLDER / self._name / f"{item_xml}.xml").resolve()
                )
                self._objects[item_name] = self._mojo.load_model(item_path)

                if "position" in item_cfg:
                    position = item_cfg["position"]
                    self._objects[item_name].set_position(
                        np.array(position, dtype=np.float64)
                    )

                if "euler" in item_cfg:
                    euler = item_cfg["euler"]
                    self._objects[item_name].set_euler(
                        np.array(euler, dtype=np.float64)
                    )

                if "kinematic" in item_cfg:
                    is_kinematic = item_cfg["kinematic"]
                    self._objects[item_name].set_kinematic(is_kinematic)

    def get_object(self, name: str) -> Body:
        if name not in self._objects:
            raise KeyError(f"Body with name '{name}' not found in this arena")
        return self._objects[name]

    @property
    def name(self) -> str:
        return self._name
