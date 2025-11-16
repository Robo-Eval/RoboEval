"""Fix to be able to render to on- and off- screen at the same time."""
import time
import glfw
import mujoco

from gymnasium.envs.mujoco.mujoco_rendering import (
    MujocoRenderer,
    WindowViewer,
    OffScreenViewer,
    BaseRender,
)
from mojo import Mojo


import sys
import mujoco.renderer
import mujoco.glfw

def patch_mujoco_renderer_and_context():
    """Patch MuJoCo renderer and GLContext to close safely without shutdown errors."""
    _original_renderer_close = mujoco.renderer.Renderer.close
    _original_glcontext_del = mujoco.glfw.GLContext.__del__

    def _safe_renderer_close(self):
        try:
            if 'mujoco' in sys.modules and getattr(mujoco.glfw, 'free', None) is not None:
                _original_renderer_close(self)
        except Exception as e:
            print(f"[MuJoCo] Warning: Renderer close failed safely: {e}")

    mujoco.renderer.Renderer.close = _safe_renderer_close

    def _safe_glcontext_del(self):
        try:
            if 'mujoco' in sys.modules and getattr(mujoco.glfw, 'free', None) is not None:
                _original_glcontext_del(self)
        except Exception as e:
            print(f"[MuJoCo] Warning: GLContext del failed safely: {e}")

    mujoco.glfw.GLContext.__del__ = _safe_glcontext_del

patch_mujoco_renderer_and_context()

class RoboEvalRenderer(MujocoRenderer):
    """Custom mujoco_rendering.MujocoRenderer with fixes.

    Notes:
        - Allows to render in human mode along with visual observations.
    """

    def __init__(self, mojo: Mojo):
        """Init."""
        super().__init__(mojo.model, mojo.data)

    def _get_viewer(self, render_mode: str) -> BaseRender:
        """See base."""
        self.viewer = self._viewers.get(render_mode)
        if self.viewer is None:
            if render_mode == "human":
                self.viewer = RoboEvalWindowViewer(self.model, self.data)

            elif render_mode in {"rgb_array", "depth_array"}:
                self.viewer = OffScreenViewer(self.model, self.data)
            else:
                raise AttributeError(
                    f"Unexpected mode: {render_mode}, "
                    f"expected modes: human, rgb_array, or depth_array"
                )
            self._set_cam_config()
            self._viewers[render_mode] = self.viewer

        self.viewer.make_context_current()

        return self.viewer

    def get_viewer(self, render_mode: str) -> BaseRender:
        """Get viewer for specified render mode."""
        return self._get_viewer(render_mode)


class RoboEvalWindowViewer(WindowViewer):
    """Custom mujoco_rendering.WindowViewer with fixes.

    Notes:
        - Fixes GUI overlap when viewer is paused.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        super().__init__(model, data)

        self.request_reset = False
        self.task_success = False
        self.task_fail = False

    def _key_callback(self, window, key: int, scancode, action: int, mods):
        super()._key_callback(window, key, scancode, action, mods)

        if action != glfw.RELEASE:
            return

        if key == glfw.KEY_Q:
            self.request_reset = True
        elif key in (glfw.KEY_0, glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4):
            self.vopt.sitegroup[key - glfw.KEY_0] ^= 1

    def _create_overlay(self):
        super()._create_overlay()

        bottomleft = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT
        self.add_overlay(
            bottomleft, "Task Success: ", "YES!" if self.task_success else "nope"
        )
        self.add_overlay(
            bottomleft, "Task Fail: ", "YES!" if self.task_fail else "nope"
        )

    def render(self):
        """See base."""

        def update():
            # fill overlay items
            self._create_overlay()

            render_start = time.time()
            if self.window is None:
                return
            elif glfw.window_should_close(self.window):
                glfw.destroy_window(self.window)
                glfw.terminate()
            self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(
                self.window
            )
            # update scene
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                self.vopt,
                mujoco.MjvPerturb(),
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL.value,
                self.scn,
            )

            # marker items
            for marker in self._markers:
                self._add_marker_to_scene(marker)

            # render
            mujoco.mjr_render(self.viewport, self.scn, self.con)

            # overlay items
            if not self._hide_menu:
                for gridpos, [t1, t2] in self._overlays.items():
                    mujoco.mjr_overlay(
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        gridpos,
                        self.viewport,
                        t1,
                        t2,
                        self.con,
                    )

            glfw.swap_buffers(self.window)
            glfw.poll_events()
            self._time_per_render = 0.9 * self._time_per_render + 0.1 * (
                time.time() - render_start
            )

            # clear overlay
            self._overlays.clear()
            # clear markers
            self._markers.clear()

        if self._paused:
            while self._paused:
                update()
                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
        else:
            self._loop_count += self.model.opt.timestep / (
                self._time_per_render * self._run_speed
            )
            if self._render_every_frame:
                self._loop_count = 1
            while self._loop_count > 0:
                update()
                self._loop_count -= 1
