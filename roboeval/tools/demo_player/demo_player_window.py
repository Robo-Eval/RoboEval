"""Demo Player GUI."""
import multiprocessing
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import glfw
import numpy as np
from dearpygui import dearpygui as dpg

from roboeval.roboeval_env import CONTROL_FREQUENCY_MAX, CONTROL_FREQUENCY_MIN
from roboeval.const import CACHE_PATH
from demonstrations.demo import Demo, DemoStep
from demonstrations.demo_store import DemoStore
from tools.demo_player.demo_player_rendering import DemoPlayerRenderer
from tools.shared.base_window import BaseWindow
from tools.shared.utils import (
    get_demos_in_dir,
)
from demonstrations.demo_converter import DemoConverter


@dataclass
class DemosTableRow:
    """Demos table row."""

    demo_file: Path
    table_row: Union[int, str]
    checkbox: Union[int, str]
    termination_col: Union[int, str]
    reward_col: Union[int, str]
    stable_col: Union[int, str]


class DemosTable:
    """Demos Table."""

    def __init__(self):
        """Init."""
        self._rows: list[DemosTableRow] = []

        with dpg.theme() as table_theme:
            with dpg.theme_component(dpg.mvTable):
                dpg.add_theme_color(
                    dpg.mvThemeCol_HeaderActive,
                    (0, 0, 0, 0),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_Header, (0, 0, 0, 0), category=dpg.mvThemeCat_Core
                )

        with dpg.child_window(height=400):
            with dpg.table(tag="SelectRows", header_row=True) as self._table:
                dpg.add_table_column(
                    width_fixed=True, width=20, init_width_or_weight=20
                )
                dpg.add_table_column(
                    label="No.", width_fixed=True, width=20, init_width_or_weight=20
                )
                dpg.add_table_column(
                    label="File", width_stretch=True, init_width_or_weight=0.0
                )
                dpg.add_table_column(
                    label="Termination",
                    width_fixed=True,
                    width=100,
                    init_width_or_weight=100,
                )
                dpg.add_table_column(
                    label="Reward",
                    width_fixed=True,
                    width=100,
                    init_width_or_weight=100,
                )
                dpg.add_table_column(
                    label="Stability",
                    width_fixed=True,
                    width=100,
                    init_width_or_weight=100,
                )
                dpg.bind_item_theme(self._table, table_theme)

    def populate(self, demo_files: list[Path]):
        """Fill table."""
        self.clean()
        for file in demo_files:
            self._add_row(file)

    def clean(self):
        """Clean table."""
        for row in self._rows:
            dpg.delete_item(row.table_row)
        self._rows.clear()

    def get_selected_demos(self) -> list[Path]:
        """Get selected demo files."""
        selected = []
        for row in self._rows:
            if dpg.get_value(row.checkbox):
                selected.append(row.demo_file)
        return selected

    def update_demo_data(
        self,
        demo_file: Path,
        termination: bool = False,
        reward: float = 0,
        stability: float = 0,
        select: bool = False,
    ):
        """Update demo information."""
        row = next((r for r in self._rows if r.demo_file == demo_file), None)
        if not row:
            return
        dpg.set_item_label(row.termination_col, str(termination))
        dpg.set_item_label(row.reward_col, f"{reward:.2f}")
        dpg.set_item_label(row.stable_col, f"{int(np.round(stability * 100))}%")
        dpg.set_value(row.checkbox, select)

    def _add_row(self, demo_file: Path):
        """Add table row."""
        with dpg.table_row(parent=self._table) as table_row:
            row_index = len(self._rows)
            checkbox = dpg.add_checkbox()
            dpg.add_selectable(
                label=str(row_index + 1),
                span_columns=True,
                callback=self._row_selected_callback,
                user_data=row_index,
            )
            dpg.add_selectable(
                label=demo_file.name,
                span_columns=True,
                callback=self._row_selected_callback,
                user_data=row_index,
            )
            termination_col = dpg.add_selectable(
                label="-",
                span_columns=True,
                callback=self._row_selected_callback,
                user_data=row_index,
            )
            reward_col = dpg.add_selectable(
                label="-",
                span_columns=True,
                callback=self._row_selected_callback,
                user_data=row_index,
            )
            stable_col = dpg.add_selectable(
                label="-",
                span_columns=True,
                callback=self._row_selected_callback,
                user_data=row_index,
            )
            self._rows.append(
                DemosTableRow(
                    demo_file,
                    table_row,
                    checkbox,
                    termination_col,
                    reward_col,
                    stable_col,
                )
            )

    def _row_selected_callback(self, sender, app_data, user_data: int):
        if user_data < 0 or user_data >= len(self._rows):
            return
        checkbox = self._rows[user_data].checkbox
        dpg.set_value(checkbox, not dpg.get_value(checkbox))


class DemoPlayerWindow(BaseWindow):
    """Demo Player GUI."""

    SETTINGS_FILE = "demo_player"
    CURRENT_DIR = "current_dir"
    ENV_ID = "env_id"

    VALIDATION_THREADS = 4

    def __init__(self):
        """Init."""
        self._validation_processes: list[multiprocessing.Process] = []
        self._is_playing = False
        self._current_dir_value = CACHE_PATH
        super().__init__()

    def _setup_ui(self):
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Change Directory...", callback=self._select_directory_callback
            )
            dpg.add_button(label="Pull Demos", callback=self._pull_demos_callback)
        dpg.add_spacer()
        with dpg.group(horizontal=True):
            self._table = DemosTable()
        dpg.add_spacer()
        with dpg.group(horizontal=True, width=200):
            with dpg.group(horizontal=False):
                dpg.add_text("Control Frequency")
                self.frequency_entry = dpg.add_input_int(
                    default_value=50,
                    min_value=CONTROL_FREQUENCY_MIN,
                    max_value=CONTROL_FREQUENCY_MAX,
                    min_clamped=True,
                    max_clamped=True,
                    step=10,
                )
        dpg.add_spacer()
        with dpg.group(horizontal=True, width=200):
            dpg.add_button(
                label="Replay",
                callback=self._replay_selected_callback,
                width=200,
            )
        self._update_files_list()

    def on_close(self):
        """See base."""
        self._stop_demo_replay()

    def _stop_demo_replay(self):
        """Stop active demo replay."""
        if not self._is_playing:
            return
        self._is_playing = False

    def _start_demo_replay(self, demo_path: Path):
        """Replay a demo from a file.

        :param demo_path: The path to the demo file.
        """
        if not demo_path.is_file() or self._is_playing:
            return
        self._is_playing = True
        self._run_demo(Demo.from_safetensors(demo_path))

    def _run_demo(self, demo: Demo):
        self._run_env(demo)
        # ToDo: Add visual observations replay

    def _run_env(self, demo: Demo):
        frequency = self._get_frequency()
        env = demo.metadata.get_env(frequency, "human")
        demo = DemoConverter.decimate(demo, frequency, robot=env.robot)
        demo_renderer = DemoPlayerRenderer(env.mojo)
        env.mujoco_renderer = demo_renderer

        def render() -> bool:
            if (
                not self._is_playing
                or demo_renderer.viewer.window is None
                or glfw.window_should_close(demo_renderer.viewer.window)
            ):
                env.close()
                self._stop_demo_replay()
                return False
            else:
                env.render()
                return True

        while True:
            env.reset(seed=int(demo.seed))
            for timestep in demo.timesteps:
                try:
                    actual_timestep = DemoStep(
                        *env.step(timestep.executed_action), timestep.executed_action
                    )
                except ValueError as e:
                    warnings.warn(str(e))
                    env.close()
                    self._stop_demo_replay()
                    return
                demo_renderer.set_demo_data(
                    demo_info=timestep, actual_info=actual_timestep
                )
                if not render():
                    return

    def _select_directory_callback(self):
        self._select_directory(self._current_dir or "", self._on_directory_selected)

    def _on_directory_selected(self, selected_path: Path):
        self._current_dir = selected_path
        self._update_files_list()

    def _update_files_list(self):
        """Select the current directory."""
        current_dir = self._current_dir
        if current_dir is None:
            return
        demos = get_demos_in_dir(current_dir)
        self._table.populate(demos)

    def _replay_selected_callback(self):
        demos = self._table.get_selected_demos()
        if not demos:
            return
        demo = demos[-1]
        if not demo.is_file():
            return
        self._start_demo_replay(demo)

    @staticmethod
    def _pull_demos_callback():
        DemoStore().pull_demos()

    def _get_frequency(self) -> int:
        return int(dpg.get_value(self.frequency_entry))

    @property
    def _current_dir(self) -> Optional[Path]:
        if self._current_dir_value:
            if not isinstance(self._current_dir_value, Path):
                self._current_dir_value = Path(self._current_dir_value)
            if self._current_dir_value.exists():
                return self._current_dir_value
        return None

    @_current_dir.setter
    def _current_dir(self, value):
        self._current_dir_value = value
