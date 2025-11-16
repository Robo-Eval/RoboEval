from abc import ABC
from typing import Optional, Type
import multiprocessing
import traceback

from roboeval.action_modes import ActionMode
from roboeval.robots.robot import Robot
from roboeval.roboeval_env import RoboEvalEnv
from tools.shared.utils import DATA_COLLECTION_MODES


class InputMode(ABC):
    """Base class for input modes."""

    def __init__(self, 
                 action_mode: Type[ActionMode], 
                 env_cls: Type[RoboEvalEnv], 
                 target_dir: str, 
                 robot_cls: Type[Robot], 
                 floating_dofs: Optional[list[str]],
                 teleop_cls_name: str,
                 config: Optional[dict] = None):
        self._action_mode = action_mode
        self._env_cls = env_cls
        self._target_dir = target_dir
        self._robot_cls = robot_cls
        self._floating_dofs = floating_dofs
        self.config = config
        try:
            self._teleop_cls = DATA_COLLECTION_MODES[teleop_cls_name]()
        except KeyError:
            raise ValueError(f"Invalid teleop_cls: {teleop_cls_name}")

    def setup(self):
        """Initialize the teleop interface."""
        action_mode_instance = self._action_mode(
            absolute=True, floating_base=True, floating_dofs=self._floating_dofs
        )
        self._teleop = self._teleop_cls(
            env_cls=self._env_cls,
            action_mode=action_mode_instance,
            resolution=(900, 1000),
            demo_directory=self._target_dir,
            robot_cls=self._robot_cls,
            config=self.config
        )

    def start(self):
        """Start the teleop interface."""
        self._teleop.run()

    def stop(self):
        """Stop the teleop interface."""
        pass

    def _run_input_mode(self, exit_event: multiprocessing.Event, launched_event: multiprocessing.Event, error_event: multiprocessing.Event):
        """Run the input mode in a controlled environment."""
        try:
            # Assuming start() should use these events
            self.start()
            launched_event.set()
            while not exit_event.is_set():
                pass
        except Exception as e:
            print(f"Error while running simulation: {e}")
            traceback.print_exc()
            error_event.set()
        finally:
            self.stop()
