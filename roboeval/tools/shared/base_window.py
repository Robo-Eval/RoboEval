"""Abstract Window GUI."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Union, Optional

from dearpygui import dearpygui as dpg


class BaseWindow(ABC):
    """Abstract GUI window."""

    def __init__(self):
        """Init."""
        self._setup_ui()

    @abstractmethod
    def _setup_ui(self):
        pass

    @staticmethod
    def _select_directory(
        default_path: Union[Path, str], callback: Callable[[Path], None]
    ):
        with dpg.file_dialog(
            modal=True,
            show=True,
            directory_selector=True,
            default_path=default_path,
            callback=lambda _, app_data: callback(Path(app_data["file_path_name"])),
            width=850,
            height=400,
        ):
            dpg.add_file_extension(".*")

    @staticmethod
    def _show_popup(
        header: str,
        message: str = "",
        actions: dict[str, Optional[Callable]] = None,
        loading_indicator: bool = False,
    ) -> int:
        def popup_callback(sender, app_data, user_data):
            popup_item = user_data["popup"]
            callback_action = user_data["action"]
            if callback_action:
                callback_action()
            dpg.delete_item(popup_item)

        with dpg.window(
            label=header,
            modal=True,
            show=True,
            popup=True,
            no_resize=True,
            min_size=(400, 100),
            autosize=True,
            no_close=True,
        ) as popup:
            with dpg.group(horizontal=True):
                if loading_indicator:
                    dpg.add_loading_indicator(style=1)
                if message:
                    dpg.add_text(message)
            with dpg.group(horizontal=True):
                actions = actions or {}
                for label, action in actions.items():
                    dpg.add_button(
                        label=label,
                        user_data={"popup": popup, "action": action},
                        callback=popup_callback,
                    )
        return popup

    def on_close(self):
        """Use to customize window termination."""
        pass
