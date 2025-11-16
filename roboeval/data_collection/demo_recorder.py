"""Demo recorder module for collecting robot demonstrations."""
import multiprocessing
import os
import traceback
import warnings
from pathlib import Path
from typing import Callable, Optional, Type

import dearpygui.dearpygui as dpg
import hydra
from omegaconf import DictConfig

from base import InputMode
from roboeval.action_modes import JointPositionActionMode, PelvisDof
from tools.shared.utils import ENVIRONMENTS, ROBOTS


class DemoRecorder:
    """Demo Recorder with support for multiple input modes."""

    def __init__(
        self, 
        target_dir: Path, 
        input_mode: str, 
        action_mode=JointPositionActionMode, 
        cfg: DictConfig = None
    ):
        """Initialize the demo recorder.

        Args:
            target_dir: Directory where to save demo recordings
            input_mode: Input mode name ("VR", "Keyboard", etc.)
            action_mode: Action mode class to use
            cfg: Configuration dictionary
        """
        self.cfg = cfg

        self._env_cls: Optional[Type] = None
        self._floating_dofs = self._get_floating_dofs()  # Define floating base DOFs

        self._robots = ROBOTS
        self._robot_cls: Optional[Type] = None

        self._simulation_process: Optional[multiprocessing.Process] = None
        self._exit_event: Optional[multiprocessing.Event] = None

        self.set_env_cls(cfg.env)
        self.set_robot(cfg.robot)
        self.set_ik(cfg.robot)
        self.action_mode = action_mode
        # Remove Spaces from the environment name
        env_name = cfg.env.replace(" ", "_")
        self._target_dir = self._construct_target_dir(target_dir, env_name)
        self._input_mode_name = input_mode
        self._input_mode = self._initialize_input_mode(input_mode)
    
    def _get_floating_dofs(self):
        """Get enabled floating degrees of freedom based on config.
        
        Returns:
            List of enabled PelvisDof values
        """
        if self.cfg.enable_all_floating_dof:
            return [PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]
        elif self.cfg.enable_floating_base:
            return [PelvisDof.X, PelvisDof.Y, PelvisDof.RZ]
        else:
            return []

    def _construct_target_dir(self, target_dir: str, env_name: str) -> str:
        """Create and return the target directory path.
        
        Args:
            target_dir: Base directory
            env_name: Environment name
            
        Returns:
            Full target directory path
        """
        # Put together the target directory and name of the environment
        target_dir = os.path.join(target_dir, env_name, self.cfg.robot)

        # Create the target directory if it does not exist
        os.makedirs(target_dir, exist_ok=True)
        return target_dir

    def _initialize_input_mode(self, input_mode: str):
        """Initialize the input mode.
        
        Args:
            input_mode: Input mode name
            
        Returns:
            Initialized InputMode object
        """
        return InputMode(
            action_mode=self.action_mode,
            env_cls=self._env_cls,
            target_dir=self._target_dir,
            robot_cls=self._robot_cls,
            floating_dofs=self._floating_dofs,
            teleop_cls_name=input_mode,
            config=self.cfg
        )

    def set_env_cls(self, env_cls: Type):
        """Set the environment class.
        
        Args:
            env_cls: Environment class to use
            
        Raises:
            ValueError: If environment class is not supported
        """
        if env_cls in ENVIRONMENTS:
            self._env_cls = ENVIRONMENTS[env_cls]
        else:
            raise ValueError(f"Unsupported environment class: {env_cls}")    
        
    def set_robot(self, robot_name: str):
        """Set robot class by name.
        
        Args:
            robot_name: Name of the robot
        """
        self._robot_cls = self._robots[robot_name]
    
    def set_ik(self, robot_name: str):
        """Set robot inverse kinematics config.
        
        Args:
            robot_name: Name of the robot
        """
        self._robot_ik_config = self._robot_cls.ik_config

    @staticmethod
    def get_env_names(filter_string: str = "") -> list[str]:
        """Get all environment names.
        
        Args:
            filter_string: Optional string to filter environment names
            
        Returns:
            List of environment names matching the filter
        """
        names = ENVIRONMENTS.keys()
        if filter_string:
            names = [name for name in names if filter_string.lower() in name.lower()]
        return list(sorted(names))

    def start_simulation(self, on_started: Optional[Callable] = None) -> bool:
        """Start the simulation with the selected input mode.
        
        Args:
            on_started: Optional callback to run after simulation starts
            
        Returns:
            True if simulation started successfully, False otherwise
        """
        if self._simulation_process:
            warnings.warn("A simulation is already running.")
            return False

        # Ensure everything is properly configured
        if not self._env_cls:
            warnings.warn("Environment class is not set.")
            return False

        # Setup the input mode
        self._input_mode.setup()
        self._input_mode.start()
        return True
    
    def get_robots(self) -> list[str]:
        """Get names of all available robots.
        
        Returns:
            List of robot names
        """
        return list(self._robots.keys())

    def _run_input_mode(
        self, 
        exit_event: multiprocessing.Event, 
        launched_event: multiprocessing.Event,
        error_event: multiprocessing.Event
    ):
        """Run the selected input mode.
        
        Args:
            exit_event: Event to signal exit
            launched_event: Event to signal successful launch
            error_event: Event to signal errors
        """
        try:
            self._input_mode.start(exit_event, launched_event)
        except Exception as e:
            print(f"Error while running simulation: {e}")
            traceback.print_exc()
            error_event.set()
        finally:
            self.stop_simulation()

    def stop_simulation(self):
        """Stop the running simulation."""
        print("Stopping simulation.")
        
        if self._simulation_process:
            self._exit_event.set()
            self._simulation_process.kill()
            self._simulation_process = None
        self._input_mode.stop()


class DemoRecorderWindow:
    """GUI for the Demo Recorder."""

    def __init__(self):
        """Initialize the demo recorder window."""
        self._input_modes = ["VR", "Keyboard"]
        self._recorder = DemoRecorder(
            target_dir=Path("demo"),
            input_mode=self._input_modes[0],  # Default to VR
        )
        self._setup_ui()

    def _setup_ui(self):
        """Setup the DearPyGUI interface."""
        dpg.create_context()
        dpg.create_viewport(title="Demo Recorder", width=600, height=400)

        with dpg.window(label="Demo Recorder", width=580, height=380):
            # Input Mode Dropdown
            dpg.add_text("Select Input Mode:")
            self._input_mode_combo = dpg.add_combo(
                self._input_modes,
                default_value=self._input_modes[0],
                callback=self._on_input_mode_change,
            )

            # Robot Selection
            dpg.add_text("Robot Model")
            self._robot_combo = dpg.add_combo(
                items=self._recorder.get_robots(),
                default_value=self._recorder.get_robots()[0],
                width=-1,
            )

            # Environment Selection
            dpg.add_text("Filter:")
            dpg.add_input_text(callback=self._filter_env_names, width=-1)
            self._env_listbox = dpg.add_listbox(
                num_items=10,
                items=self._recorder.get_env_names(),
                width=-1,
            )

            # Start/Stop Buttons
            dpg.add_button(label="Start Simulation", callback=self._start_simulation)
            dpg.add_button(label="Stop Simulation", callback=self._stop_simulation)

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()
    
    def _filter_env_names(self, sender, filter_string: str = ""):
        """Filter environment names based on input string.
        
        Args:
            sender: UI element that triggered the callback
            filter_string: String to filter by
        """
        names = self._recorder.get_env_names(filter_string)
        dpg.configure_item(self._env_listbox, items=names)

    def _on_input_mode_change(self, sender, value):
        """Handle input mode selection.
        
        Args:
            sender: UI element that triggered the callback
            value: Selected input mode
        """
        print(f"Input mode changed to: {value}")
        self._recorder = DemoRecorder(target_dir=Path("demo"), input_mode=value)
    
    def _configure_recorder(self):
        """Configure recorder with current UI settings."""
        self._recorder.set_env_cls(dpg.get_value(self._env_listbox))
        self._recorder.set_robot(dpg.get_value(self._robot_combo))

    def _start_simulation(self):
        """Start the simulation."""
        self._configure_recorder()
        if self._recorder.start_simulation():
            print("Viewer started.")

    def _stop_simulation(self):
        """Stop the simulation."""
        self._recorder.stop_simulation()
        print("Viewer stopped.")


@hydra.main(config_path="../configs", config_name="data_collection")
def main(cfg: DictConfig):
    """Main entry point.
    
    Args:
        cfg: Hydra configuration
    """
    # Initialize DemoRecorder with Hydra config values
    recorder = DemoRecorder(
        target_dir=Path(cfg.target_dir),
        input_mode=cfg.input_mode,
        cfg=cfg
    )

    # Start simulation
    recorder.start_simulation()


if __name__ == "__main__":
    main()
