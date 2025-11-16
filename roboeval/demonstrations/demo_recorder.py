"""Demo recorder class."""
import tempfile
from pathlib import Path
from typing import Optional, Union

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.demonstrations.demo import Demo, DemoStep, LightweightDemo


class DemoRecorder:
    """Demo recorder class."""

    def __init__(self, demo_dir: Optional[Union[str, Path]] = None):
        """Init.

        Args:
            demo_dir: The directory where the demos will be saved.
        """
        self._temp_dir = None
        if demo_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory()
            demo_dir = Path(self._temp_dir.name)
        elif isinstance(demo_dir, str):
            demo_dir = Path(demo_dir)
        self._demo_dir: Path = demo_dir
        self._demo: Optional[Demo] = None
        self._env: Optional[RoboEvalEnv] = None
        self._recording: bool = False

    def __del__(self):
        """Remove the temporary demo directory if it was used."""
        if self._temp_dir:
            self._temp_dir.cleanup()

    def record(self, env: RoboEvalEnv, lightweight_demo: bool = False):
        """Enable recording.

        Args:
            env(RoboEvalEnv): The RoboEval environment.
            lightweight_demo(bool): Whether to record a lightweight demo.
        """
        if self._recording:
            return
        self._env = env
        self._recording = True
        if lightweight_demo:
            self._demo = LightweightDemo.from_env(env)
        else:
            self._demo = Demo.from_env(env)

    def stop(self):
        """Disable recording."""
        if self._recording:
            self._recording = False

    def add_timestep(self, timestep: tuple, action):
        """Add a timestep to the recording.

        Args:
            timestep: A dictionary containing time step information.
            action: The action taken to reach the time step.
        """
        if self._recording:
            self._demo.add_timestep(*timestep, action)

    @property
    def timesteps(self) -> list[DemoStep]:
        """Time steps."""
        return self._demo.timesteps.copy()

    @property
    def demo(self) -> Optional[Demo]:
        """Demo."""
        return self._demo

    def save_demo(self, filename: Optional[str] = None) -> Optional[Path]:
        """Save a demo to a file.

        Args:
            filename: The name of the file.

        Returns:
            Path: The path of the saved file.
        """
        if not self._demo:
            return None
        if not self._demo_dir.exists():
            self._demo_dir.mkdir(parents=True)
        filename = filename or self._demo.metadata.filename
        filepath = self._demo_dir / filename
        self._demo.save(filepath)
        self._demo = None
        self._env = None
        return filepath

    @property
    def is_recording(self):
        """Check if recording in progress."""
        return self._recording is True
