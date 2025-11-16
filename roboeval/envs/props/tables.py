"""Tables."""
from pathlib import Path
from mojo.elements import Body, MujocoElement, Joint, Site


from roboeval.const import ASSETS_PATH
from roboeval.envs.props.prop import CollidableProp


class Table(CollidableProp):
    """Default Table."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/table/table.xml"
    
    def _on_loaded(self, model):
        object = MujocoElement(self._mojo, model)
        self.table_top = Body.get(self._mojo, "counter", object).geoms[-1]


class SmallTable(CollidableProp):
    """Shorter version of the default table."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/table_dishwasher/table_dishwasher.xml"
