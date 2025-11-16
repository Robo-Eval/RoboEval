"""Demo Converter GUI."""
from pathlib import Path
from typing import Optional, Callable

from dearpygui import dearpygui as dpg

from demonstrations.demo import Demo, TERMINATION_STEPS
from demonstrations.utils import Metadata
from tools.shared.utils import (
    REPLAY_MODES,
    get_demos_in_dir,
    ReplayMode,
)
from tools.shared.base_window import BaseWindow
from demonstrations.demo_converter import DemoConverter


class DemoConverterWindow(BaseWindow):
    """Demo Converter GUI."""

    def __init__(self):
        """Init."""
        self._source_dir: Optional[Path] = None
        self._target_dir: Optional[Path] = None
        super().__init__()

    def _setup_ui(self):
        with dpg.window(
            label="Convert Demos",
            modal=True,
            show=False,
            popup=True,
            no_resize=False,
            autosize=True,
        ) as self._window:
            dpg.add_text("1. Action Mode")
            dpg.add_text("Original Mode")
            self._source_mode = dpg.add_listbox(
                num_items=3, items=list(REPLAY_MODES.keys()), width=-1
            )
            dpg.set_value(self._source_mode, list(REPLAY_MODES.keys())[0])
            dpg.add_text("Target Mode")
            self._target_mode = dpg.add_listbox(
                num_items=3, items=list(REPLAY_MODES.keys()), width=-1
            )
            dpg.set_value(self._target_mode, list(REPLAY_MODES.keys())[1])
            dpg.add_spacer()

            dpg.add_text("2. Override Metadata")
            dpg.add_text("Action Mode Name:")
            self._action_mode_input = dpg.add_input_text()
            dpg.add_text("Environment Name:")
            self._env_name_input = dpg.add_input_text()
            dpg.add_spacer()

            dpg.add_text("3. Output")
            dpg.add_text("Clip actions")
            self._clip_actions = dpg.add_checkbox()
            dpg.add_text("Add termination steps")
            self._add_steps_checkbox = dpg.add_checkbox()
            dpg.add_text("Keep names")
            self._keep_names_checkbox = dpg.add_checkbox()
            dpg.add_text("Output Directory")
            with dpg.group(horizontal=True):
                dpg.add_button(label="Change", callback=self._select_directory_callback)
                self.out_directory_label = dpg.add_text("...")
            dpg.add_spacer()

            self._convert_buttons = []
            self._convert_buttons.append(
                dpg.add_button(
                    label="Convert",
                    width=-1,
                    callback=self._convert,
                )
            )
            self._convert_buttons.append(
                dpg.add_button(
                    label="Re-Record",
                    width=-1,
                    callback=self._re_record,
                )
            )
            self._progress = dpg.add_progress_bar(
                width=-1, default_value=0.0, show=False
            )

    def _re_record(self):
        self._process_demos()

    def _convert(self):
        source_mode = REPLAY_MODES[dpg.get_value(self._source_mode)]
        target_mode = REPLAY_MODES[dpg.get_value(self._target_mode)]
        if source_mode == ReplayMode.Absolute and target_mode == ReplayMode.Delta:
            demo_converter = DemoConverter.absolute_to_delta
            self._process_demos(demo_converter)

    def _process_demos(self, demo_converter: Callable[[Demo], Demo] = None):
        if self._source_dir is None or self._target_dir is None:
            return
        demo_files = get_demos_in_dir(self._source_dir)
        self._start_processing()
        self._update_progress(0, len(demo_files))
        for i, demo_file in enumerate(demo_files):
            try:
                metadata = Metadata.from_safetensors(demo_file)
                self._update_metadata(metadata)
                demo = Demo.from_safetensors(demo_file, metadata)
                if demo_converter:
                    demo = demo_converter(demo)
                if dpg.get_value(self._clip_actions):
                    demo = DemoConverter.clip_actions(demo)
                if dpg.get_value(self._add_steps_checkbox):
                    demo.add_termination_steps(TERMINATION_STEPS)
                file_name = demo.metadata.filename
                if dpg.get_value(self._keep_names_checkbox):
                    file_name = demo_file.name
                demo.save(self._target_dir / file_name)
            except Exception as ex:
                self._finish_processing()
                raise ex
            self._update_progress(i + 1, len(demo_files))
        self._finish_processing()

    def _update_progress(self, current, total):
        dpg.set_value(self._progress, float(current) / total)
        dpg.configure_item(self._progress, overlay=f"{current}/{total}")

    def _update_metadata(self, metadata: Metadata):
        """Update demo metadata before loading demo file."""
        if new_action_mode_name := dpg.get_value(self._action_mode_input):
            metadata.environment_data.action_mode_name = new_action_mode_name
        if new_env_name := dpg.get_value(self._env_name_input):
            metadata.environment_data.env_name = new_env_name

    def _start_processing(self):
        for button in self._convert_buttons:
            dpg.hide_item(button)
        dpg.show_item(self._progress)

    def _finish_processing(self):
        for button in self._convert_buttons:
            dpg.show_item(button)
        dpg.hide_item(self._progress)
        dpg.hide_item(self._window)

    def _select_directory_callback(self):
        self._select_directory(self._target_dir, self._on_target_dir_selected)

    def _on_target_dir_selected(self, target_path: Path):
        self._set_target_dir(target_path)
        self._show_window()

    def _set_target_dir(self, target_path: Path):
        self._target_dir = target_path
        dpg.set_value(self.out_directory_label, str(self._target_dir))

    def show(self, source_dir: Path):
        """Open converter window."""
        if source_dir is None:
            return
        self._source_dir = source_dir
        self._set_target_dir(source_dir)
        self._show_window()

    def _show_window(self):
        dpg.split_frame()
        dpg.show_item(self._window)
