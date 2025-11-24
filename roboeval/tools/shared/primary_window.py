"""Primary Window GUI."""
import dearpygui.dearpygui as dpg

from tools.shared.base_window import BaseWindow

DEBUG_MODE = False


class PrimaryWindow:
    """Primary Window GUI."""

    def __init__(
        self,
        main_window: type[BaseWindow],
        title: str,
        width: int = 1000,
        height: int = 600,
        resizable: bool = False,
    ):
        """Init."""
        dpg.create_context()
        if DEBUG_MODE:
            dpg.configure_app(manual_callback_management=True)

        with dpg.window() as window:
            main_window = main_window()
            dpg.set_primary_window(window, True)

        dpg.create_viewport(
            title=title, width=width, height=height, resizable=resizable
        )
        dpg.setup_dearpygui()

        # Main loop
        dpg.show_viewport()
        if DEBUG_MODE:
            while dpg.is_dearpygui_running():
                jobs = dpg.get_callback_queue()
                dpg.run_callbacks(jobs)
                dpg.render_dearpygui_frame()
        else:
            dpg.start_dearpygui()

        # Terminate viewport
        main_window.on_close()
        dpg.destroy_context()
