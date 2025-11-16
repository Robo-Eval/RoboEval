"""Tracks the stability of the simulation."""
import contextlib
import warnings
from typing import Optional

from dm_control.rl.control import PhysicsError


class UnstableSimulationWarning(UserWarning):
    """Raised if the state of the physics simulation becomes divergent."""


class UnstableSimulationError(RuntimeError):
    """Raised if the UnstableSimulationWarning occurred multiple times consecutively."""


class EnvHealth:
    """Tracks the stability of the simulation.

    This class catches and counts PhysicsError from a Mujoco environment.
    If more than 10 consecutive PhysicsError instances occur, an exception is raised.
    """

    CONSECUTIVE_WARNINGS_THRESHOLD = 10

    def __init__(self):
        """Init."""
        self._consecutive_errors: list[PhysicsError] = []
        self._current_error: Optional[PhysicsError] = None

    def reset(self):
        """Used to track consecutive errors. Should be called on environment reset."""
        if self._current_error is None:
            self._consecutive_errors.clear()
        self._current_error = None

    @contextlib.contextmanager
    def track(self):
        """A context manager to track physics errors during simulation steps."""
        try:
            yield
        except PhysicsError as physics_error:
            self._current_error = physics_error
            self._consecutive_errors.append(self._current_error)
            error = (
                f"Physics error has occurred. "
                f"Truncate current episode and reset the environment.\n"
                f"{str(self._current_error)}"
            )
            warnings.warn(error, UnstableSimulationWarning)
            if len(self._consecutive_errors) >= self.CONSECUTIVE_WARNINGS_THRESHOLD:
                raise UnstableSimulationError

    @property
    def is_healthy(self) -> bool:
        """Checks if the simulation is currently healthy."""
        return self._current_error is None
