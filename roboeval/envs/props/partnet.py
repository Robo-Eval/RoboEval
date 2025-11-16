from pathlib import Path

from mojo import Mojo
from mojo.elements import Joint, Body

from roboeval.const import ASSETS_PATH
from roboeval.envs.props.prop import KinematicProp
from roboeval.utils.physics_utils import get_joint_position, set_joint_position

PARTNET_ASSETS_PATH = ASSETS_PATH / "partnet"


class MicrowaveDoor:
    HINGE_JOINT = "microwave/microwave_hinge"
    HINGE_ANGLE_CLOSE = -0.524
    HINGE_ANGLE_OPEN = 1.4

    def __init__(self, mojo: Mojo, microwave_body: Body):
        self._mojo = mojo

        self.joint = Joint.get(self._mojo, MicrowaveDoor.HINGE_JOINT, microwave_body)

    def get_hinge_angle(self) -> float:
        return get_joint_position(self.joint, True)

    def set_hinge_angle(self, hinge_value: float):
        set_joint_position(self.joint, hinge_value, True)

    def close_door(self):
        self.set_hinge_angle(MicrowaveDoor.HINGE_ANGLE_CLOSE)

    def open_door(self):
        self.set_hinge_angle(MicrowaveDoor.HINGE_ANGLE_OPEN)


class Microwave(KinematicProp):
    @property
    def _model_path(self) -> Path:
        return PARTNET_ASSETS_PATH / "microwave" / "mobility.xml"

    def _post_init(self):
        self.door = MicrowaveDoor(self._mojo, self.body)


class KitchenPot(KinematicProp):
    _CACHE_SITES = True

    @property
    def _model_path(self) -> Path:
        return PARTNET_ASSETS_PATH / "kitchenpot" / "mobility.xml"


class Toaster(KinematicProp):
    _CACHE_SITES = True

    @property
    def _model_path(self) -> Path:
        return PARTNET_ASSETS_PATH / "toaster" / "mobility.xml"


class Stapler(KinematicProp):
    @property
    def _model_path(self) -> Path:
        return PARTNET_ASSETS_PATH / "stapler" / "mobility.xml"


class Trashcan(KinematicProp):
    @property
    def _model_path(self) -> Path:
        return PARTNET_ASSETS_PATH / "trashcan" / "mobility.xml"
