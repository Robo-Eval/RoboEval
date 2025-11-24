"""Custom rendering classes for demo player."""
from typing import Optional

import mujoco
import numpy as np
from mojo import Mojo

from roboeval.roboeval_renderer import RoboEvalRenderer, RoboEvalWindowViewer
from demonstrations.demo import DemoStep

MAX_LABEL_LENGTH = 30


class DemoPlayerRenderer(RoboEvalRenderer):
    """Custom renderer for demo player.

    This renderer uses custom DemoPlayerWindowViewer.
    It allows to display information about current step to simplify debugging.
    """

    def __init__(self, mojo: Mojo):
        """See base."""
        super().__init__(mojo)
        self.viewer = DemoPlayerWindowViewer(self.model, self.data)

    def _get_viewer(self, render_mode: Optional[str] = None):
        """See base."""
        self.viewer.make_context_current()
        return self.viewer

    def close(self):
        """See base."""
        super().close()
        self.viewer.close()

    def set_demo_data(self, demo_info: DemoStep, actual_info: DemoStep):
        """Send current step information to viewer."""
        self._get_viewer().set_step_data(demo_info, actual_info)


class DemoPlayerWindowViewer(RoboEvalWindowViewer):
    """Custom viewer for demo player."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ):
        """See base."""
        super().__init__(model, data)
        self.demo_info: Optional[DemoStep] = None
        self.actual_info: Optional[DemoStep] = None

    def set_step_data(self, demo_info: DemoStep, actual_info: DemoStep):
        """Set current step information."""
        self.demo_info = demo_info
        self.actual_info = actual_info

    def _create_overlay(self):
        super()._create_overlay()
        top_right = mujoco.mjtGridPos.mjGRID_TOPRIGHT
        if self.demo_info:
            self.add_overlay(top_right, "Demo Info", "")
            self._add_step_info(top_right, self.demo_info)
        if self.actual_info:
            self.add_overlay(top_right, "Actual Info", "")
            self._add_step_info(top_right, self.actual_info)

    def _add_step_info(self, position, step: DemoStep):
        self.add_overlay(
            position, "Observation", f"{len(step.observation.keys())} key(s)"
        )
        for key, value in step.observation.items():
            self.add_overlay(position, key, self._format_value(value))
        self.add_overlay(position, "Reward", self._format_value(step.reward))

    def _format_value(self, value) -> str:
        formatted_value = np.array2string(
            np.array(value), formatter={"float_kind": self._float_formatter}
        )
        if len(formatted_value) > MAX_LABEL_LENGTH:
            formatted_value = f"{formatted_value[:MAX_LABEL_LENGTH-3]}..."
        return formatted_value

    @staticmethod
    def _float_formatter(value):
        return f"{value:.4f}"
