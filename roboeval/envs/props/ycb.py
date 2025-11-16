from pathlib import Path
from typing import Optional

from mojo import Mojo
from mojo.elements.element import MujocoElement
from roboeval.utils.physics_utils import has_collided_collections, get_colliders
from roboeval.envs.props.prop import Prop

from roboeval.const import ASSETS_PATH
from roboeval.envs.props.prop import KinematicProp

YCB_ASSETS_PATH = ASSETS_PATH / "ycb"


class YcbProp(KinematicProp):
    _CACHE_SITES = True

    def __init__(
        self,
        ycb_model_name: str,
        mojo: Mojo,
        kinematic: Optional[bool] = None,
        cache_colliders: Optional[bool] = None,
        cache_sites: Optional[bool] = None,
        parent: Optional[MujocoElement] = None,
        **kwargs,
    ):
        self._ycb_model_name = ycb_model_name
        super().__init__(
            mojo, kinematic, cache_colliders, cache_sites, parent, **kwargs
        )

    @property
    def _model_path(self) -> Path:
        return YCB_ASSETS_PATH / self._ycb_model_name / "model.xml"
