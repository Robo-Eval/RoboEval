"""VR Demo Recorder."""

from tools.demo_recorder.demo_recorder_window import DemoRecorderWindow
from tools.shared.primary_window import PrimaryWindow


class DemoRecorder:
    """VR Demo Recorder."""

    def __init__(self):
        """Init."""
        PrimaryWindow(
            DemoRecorderWindow,
            title="VR Demo Recorder",
            height=450,
            resizable=True,
        )


if __name__ == "__main__":
    DemoRecorder()
