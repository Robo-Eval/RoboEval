"""VR Demo Recorder Window."""
import multiprocessing
import time
import traceback
import warnings
from pathlib import Path
from typing import Optional, Type, Callable


from roboeval.action_modes import PelvisDof, JointPositionActionMode
from roboeval.roboeval_env import RoboEvalEnv
from roboeval.robots.robot import Robot
from tools.shared.base_window import BaseWindow
import dearpygui.dearpygui as dpg

from tools.shared.utils import (
    ENVIRONMENTS,
    select_directory,
    show_popup,
    ROBOTS,
    CONTROL_PROFILES,
)
from vr.viewer.control_profiles.control_profile import ControlProfile
from vr.viewer.vr_viewer import VRViewer


class DemoRecorder:
    """Demo Recorder."""

    _VIEWER_PROCESS_TERMINATION_DELAY = 2

    def __init__(self, target_dir: Path):
        """Init."""
        self._target_dir = target_dir
        self._exit_event: Optional[multiprocessing.Event] = None
        self._viewer_process: Optional[multiprocessing.Process] = None

        self._env_cls: Optional[Type[RoboEvalEnv]] = None
        self._floating_dofs: dict[PelvisDof, bool] = {dof: True for dof in PelvisDof}

        self._robots = ROBOTS
        self._robot_cls: Optional[Type[Robot]] = None

        self._control_profiles = CONTROL_PROFILES
        self._control_profile_cls: Optional[Type[ControlProfile]] = None

    @property
    def is_running(self) -> bool:
        """Check if VR viewer is running."""
        return self._viewer_process is not None

    @property
    def target_dir(self) -> Path:
        """Get target directory."""
        return self._target_dir

    @target_dir.setter
    def target_dir(self, value: Path):
        """Set target directory."""
        self._target_dir = value

    def set_env_cls(self, env_cls_name: str):
        """Set environment class by name."""
        if env_cls_name in ENVIRONMENTS:
            self._env_cls = ENVIRONMENTS[env_cls_name]

    def toggle_floating_dof(self, dof_name: str, active: bool):
        """Toggle floating dof."""
        dof = PelvisDof(dof_name)
        self._floating_dofs[dof] = active

    def get_active_floating_dofs(self) -> list[PelvisDof]:
        """Get enabled floating dofs."""
        return [dof for dof, is_active in self._floating_dofs.items() if is_active]

    def get_floating_dofs(self) -> dict[PelvisDof, bool]:
        """Get all names of floating base dofs."""
        return self._floating_dofs

    def get_robots(self) -> list[str]:
        """Get names of all available robots."""
        return [k for k in self._robots.keys()]

    def set_robot(self, robot_name: str):
        """Set robot class by name."""
        self._robot_cls = self._robots[robot_name]

    def get_control_profiles(self) -> list[str]:
        """Get names of all available control profiles."""
        return [k for k in self._control_profiles.keys()]

    def set_control_profile(self, profile_name: str):
        """Set control profile class by name."""
        self._control_profile_cls = self._control_profiles[profile_name]

    @staticmethod
    def get_env_names(filter_string: str = "") -> list[str]:
        """Get all environment names."""
        names = ENVIRONMENTS.keys()
        if filter_string:
            names = [name for name in names if filter_string.lower() in name.lower()]
        return list(sorted(names))

    def start_viewer(self, on_started: Optional[Callable] = None) -> bool:
        """Launch VR viewer."""
        if self._viewer_process:
            warnings.warn("Another instance of VRViewer is already running.")
            return False
        if self._env_cls is None:
            warnings.warn("Environment class is not set.")
            return False
        if len(self._floating_dofs) == 0:
            warnings.warn("Floating base DOFs are not set.")
            return False
        if self._control_profile_cls is None:
            warnings.warn("Control profile is not ste.")
            return False

        self._exit_event = multiprocessing.Event()
        launched_event = multiprocessing.Event()
        error_event = multiprocessing.Event()
        self._viewer_process = multiprocessing.Process(
            target=self._run_vr_viewer,
            args=(self._exit_event, launched_event, error_event),
        )
        self._viewer_process.start()

        # Wait for the viewer to launch/crash
        while True:
            if launched_event.is_set():
                if on_started:
                    on_started()
                return True
            if error_event.is_set():
                self._viewer_process.kill()
                self._viewer_process = None
                return False
            multiprocessing.Event().wait(0.1)

    def _run_vr_viewer(
        self,
        exit_event: multiprocessing.Event,
        running_event: multiprocessing.Event,
        error_event: multiprocessing.Event,
    ):
        try:
            viewer = VRViewer(
                env_cls=self._env_cls,
                action_mode=JointPositionActionMode(
                    absolute=True,
                    floating_base=True,
                    floating_dofs=self.get_active_floating_dofs(),
                ),
                control_profile_cls=self._control_profile_cls,
                demo_directory=self._target_dir,
                robot_cls=self._robot_cls,
            )
            viewer.run(exit_event, running_event)
        except Exception as e:
            print(f"Exception while running VR: {e}")
            traceback.print_exc()
            error_event.set()

    def stop_viewer(self, on_stopped: Optional[Callable] = None):
        """Stop VR viewer."""
        if self._viewer_process is None:
            return
        self._exit_event.set()
        time.sleep(self._VIEWER_PROCESS_TERMINATION_DELAY)
        self._viewer_process.kill()
        self._viewer_process = None
        if on_stopped:
            on_stopped()


class DemoRecorderWindow(BaseWindow):
    """VR Demo Recorder Window."""

    def __init__(self):
        """Init."""
        self._recorder = DemoRecorder(Path(__file__).parent / "demo")
        super().__init__()

    def _setup_ui(self):
        # Environment selection
        with dpg.group(horizontal=True):
            dpg.add_text("Filter:")
            dpg.add_input_text(callback=self._filter_env_names, width=-1)
        self._env_listbox = dpg.add_listbox(
            num_items=10,
            items=self._recorder.get_env_names(),
            width=-1,
        )

        # Floating DOFs
        self._dof_toggles = {}
        dpg.add_text("Floating Base DOFs")
        with dpg.group(horizontal=True):
            for dof, is_active in self._recorder.get_floating_dofs().items():
                self._dof_toggles[dof.value] = dpg.add_checkbox(
                    label=dof.name,
                    default_value=is_active,
                )
                dpg.add_spacer()

        # Robot
        dpg.add_text("Robot Model")
        self._robot_combo = dpg.add_combo(
            self._recorder.get_robots(),
            default_value=self._recorder.get_robots()[0],
            width=-1,
        )

        # Control Profile
        dpg.add_text("Control Profile")
        self._profile_combo = dpg.add_combo(
            self._recorder.get_control_profiles(),
            default_value=self._recorder.get_control_profiles()[0],
            width=-1,
        )

        # Output directory
        dpg.add_text("Output Directory")
        self._target_dir_button = dpg.add_button(
            label=str(self._recorder.target_dir),
            callback=self._change_target_dir,
            indent=0,
            width=-1,
        )

        # Buttons
        dpg.add_spacer()
        self._start_button = dpg.add_button(
            label="Record", width=-1, callback=self._start
        )
        self._stop_button = dpg.add_button(
            label="Stop", width=-1, callback=self._stop, show=False
        )

    def _filter_env_names(self, sender, filter_string: str = ""):
        names = self._recorder.get_env_names(filter_string)
        dpg.configure_item(self._env_listbox, items=names)

    def _configure_recorder(self):
        self._recorder.set_env_cls(dpg.get_value(self._env_listbox))
        for dof, toggle in self._dof_toggles.items():
            self._recorder.toggle_floating_dof(dof, dpg.get_value(toggle))
        self._recorder.set_robot(dpg.get_value(self._robot_combo))
        self._recorder.set_control_profile(dpg.get_value(self._profile_combo))

    def _change_target_dir(self):
        select_directory(self._recorder.target_dir, self._set_target_dir)

    def _set_target_dir(self, target_dir: Path):
        self._recorder.target_dir = target_dir
        dpg.configure_item(self._target_dir_button, label=str(target_dir))

    def _start(self):
        if self._recorder.is_running:
            return
        popup = show_popup(message="Launching VR...", loading_indicator=True)
        self._configure_recorder()
        started = self._recorder.start_viewer(lambda: self._on_started(popup))
        if not started:
            dpg.delete_item(popup)

    def _on_started(self, popup: int):
        dpg.delete_item(popup)
        dpg.hide_item(self._start_button)
        dpg.show_item(self._stop_button)

    def _stop(self):
        if not self._recorder.is_running:
            return
        popup = show_popup(message="Terminating VR...", loading_indicator=True)
        self._recorder.stop_viewer(lambda: self._on_stopped(popup))

    def _on_stopped(self, popup: int):
        dpg.delete_item(popup)
        dpg.hide_item(self._stop_button)
        dpg.hide_item(self._start_button)
        dpg.split_frame()
        show_popup(
            message="Please relaunch to record new demonstrations.",
            actions={"Exit": self._exit},
        )

    @staticmethod
    def _exit():
        dpg.stop_dearpygui()
