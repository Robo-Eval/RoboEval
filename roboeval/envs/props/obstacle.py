"""Tables."""
from pathlib import Path


from roboeval.const import ASSETS_PATH
from roboeval.envs.props.prop import CollidableProp


class Obstacle(CollidableProp):
    """Default Table."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/obstacle/obstacle.xml"