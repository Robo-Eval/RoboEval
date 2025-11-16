from abc import ABC, abstractmethod
class Teleop(ABC):
    """Base class for teleoperation."""

    @abstractmethod
    def run(self):
        pass
    
    def _render_frame(self):
        self._update_stats()
        self._env.render()
        # Simulate rendering process (e.g., saving frames, updating UI)
        pass

    def _start_recording(self):
        self._stop_recording()
        self._env.reset()
        self._demo_recorder.record(self._env, lightweight_demo=True)

    def _stop_recording(self):
        if self._demo_recorder.is_recording:
            self._demo_recorder.stop()

    def _save_recording(self):
        self._stop_recording()
        if self._demo_recorder.save_demo():
            self._stats.demos_counter += 1

    def _update_stats(self):
        self._stats.is_recoding = self._demo_recorder.is_recording
        self._stats.time = self._env.mojo.data.time
        self._stats.reward = self._env.reward
    