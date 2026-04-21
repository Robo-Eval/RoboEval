"""Telearms teleoperation driver.

Bridges the Agora media plane (video out + RTM input) and the JSON-RPC
WebSocket control plane (TeleArmsWSClient) to RoboEval's simulation.

MVP scope: start/stop/reset commands, H.264 video stream over Agora,
keyboard + gamepad input from the frontend over Agora RTM. Joy-Con IMU
input, temporal success checking, and telemetry S3 upload are intentionally
out of scope for this port — add them back once the MVP is validated.
"""

# Standard library
import asyncio
import atexit
import json
import os
import queue as pyqueue
import signal
import threading
import time
import traceback
import uuid
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Type, Union

# Third-party
import numpy as np
import requests
from dotenv import load_dotenv
from gymnasium.core import ActType
from pyquaternion import Quaternion

load_dotenv()

# GStreamer — imported after load_dotenv so DISPLAY/MUJOCO_GL env is set
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst  # noqa: E402

Gst.init(None)

# teleop_sdk is the Agora RTC/RTM wrapper shipped as a wheel under telearms/
import teleop_sdk  # noqa: E402

# Local
from roboeval.data_collection.keyboard_input import (  # noqa: E402
    Countdown,
    KeyboardTeleop,
)
from roboeval.data_collection.webrtc_config import (  # noqa: E402
    TaskConfig,
    get_random_task,
    get_task_by_name,
)
from roboeval.demonstrations.demo import TERMINATION_STEPS  # noqa: E402
from roboeval.roboeval_env import RoboEvalEnv  # noqa: E402
from telearms.ws_client import InstanceStatus, TeleArmsWSClient  # noqa: E402
from tools.shared.utils import ENVIRONMENTS  # noqa: E402


# ==================== Environment config ====================
WS_SERVER_URL = os.environ.get("WS_SERVER_URL", "ws://localhost:8080")
TELEARMS_BACKEND_URL = os.environ.get("TELEARMS_BACKEND_URL", "http://localhost:3000")
TELEARMS_BACKEND_API_KEY = os.environ.get("TELEARMS_BACKEND_API_KEY", "")

WIDTH, HEIGHT = 1280, 720
FPS = 30

PIPELINE = (
    f'appsrc name=source is-live=true block=true format=TIME '
    f'caps=video/x-raw,format=RGB,width={WIDTH},height={HEIGHT},framerate={FPS}/1 ! '
    f'videoconvert ! '
    f'x264enc speed-preset=ultrafast bframes=0 tune=zerolatency byte-stream=true key-int-max=30 ! '
    f'video/x-h264,stream-format=byte-stream ! '
    f'appsink name=appsink emit-signals=true sync=false max-buffers=100 drop=true'
)

# Frontend RTM messages sometimes include the same keys used for local
# record/save shortcuts — ignore those so only backend commands can trigger
# recording lifecycle transitions.
_IGNORED_FRONTEND_KEYS = {KeyboardTeleop.RECORD_KEY, KeyboardTeleop.SAVE_KEY}

# Module-level ref so atexit/signal handlers can tear the instance down.
_active_teleop_instance: Optional["TelearmsTeleop"] = None


@dataclass
class TelearmsTeleopStats:
    is_recording: bool = False
    time: float = 0.0
    reward: float = 0.0
    demos_counter: int = 0


class TelearmsTeleop(KeyboardTeleop):
    """Telearms (Agora + WS) teleoperation driver for RoboEval envs.

    Reuses KeyboardTeleop's arm/gripper/IK logic via inheritance; replaces
    the local pynput listener with RTM-based keyboard/gamepad events and
    streams frames over Agora RTC instead of rendering to a local window.
    """

    # Gamepad axes/buttons come straight from the browser Gamepad API.
    GAMEPAD_MOVEMENT_SCALE = 0.008
    GAMEPAD_ROTATION_SCALE = 0.008
    GAMEPAD_DEADZONE = 0.15

    def __init__(
        self,
        env_cls: Optional[Type[RoboEvalEnv]],
        action_mode,
        resolution: tuple[int, int] = (WIDTH, HEIGHT),
        demo_directory: Optional[Union[str, Path]] = None,
        robot_cls: Optional[Type[Any]] = None,
        config: Optional[dict] = None,
        port: int = 8080,
    ):
        # NOTE: intentionally do NOT call super().__init__() — the parent
        # starts a pynput listener and opens a human-visible renderer. We
        # replicate the parts we need below and skip those two.
        from roboeval.demonstrations.demo_recorder import DemoRecorder

        self.config = config
        self._width, self._height = resolution
        self._demo_recorder = DemoRecorder(demo_directory)
        self.port = port
        self._env_lock = threading.RLock()

        self._action_mode = action_mode
        self._robot_cls = robot_cls

        # Resolve initial task
        self._current_task = self._resolve_initial_task(env_cls)
        env_cls = ENVIRONMENTS.get(self._current_task.name)
        print(f"[telearms] Loading environment: {self._current_task.name}")

        # Build env with WebRTC wrapper (rgb_array so render returns frames)
        webrtc_env_cls = self._create_webrtc_env(env_cls)
        self._env = webrtc_env_cls(
            render_mode="rgb_array", action_mode=action_mode, robot_cls=robot_cls
        )
        self._env.mojo.model.vis.global_.offwidth = self._width
        self._env.mojo.model.vis.global_.offheight = self._height
        self._env.reset()

        # State
        self._stats = TelearmsTeleopStats()
        self._stop_countdown: Optional[Countdown] = None
        self._current_action = np.zeros(self._env.action_space.shape)
        self._ik = self._env.inverse_kinematics

        # Input state — no pynput listener; keys come via RTM
        self.pressed_keys: set = set()

        # Gamepad state
        self.gamepad_axes = [0.0, 0.0, 0.0, 0.0]
        self.gamepad_buttons = [0.0] * 17
        self._prev_gamepad_buttons = [0.0] * 17
        self._gamepad_active = False

        # Control settings
        self.control_mode = self.MODE_POSITION
        self.toggle_gripper_mode = True

        self._setup_grippers()
        self._init_arm_state()
        self._save_initial_positions()

        self._gripper_toggle_state = {"left": 0.0, "right": 0.0}
        self._prev_key_state = {
            self.GRIPPER_KEYS["left"]: False,
            self.GRIPPER_KEYS["right"]: False,
        }

        # Scene-control channel (reset_scene / change_task) coming from RTM
        self.scene_control_queue: pyqueue.Queue = pyqueue.Queue()
        self.scene_control_handlers = {
            "reset_scene": self._handle_reset_scene,
            "change_task": self._handle_change_task,
        }

        # Task lifecycle state
        self._env_start_recording_event = threading.Event()
        self._task_start_time: Optional[float] = None
        self._frames_recorded = 0
        self._shutdown_done = False
        self._video_paused = False

        # Agora init — callback pair is registered now; tokens are set later
        # in _handle_start_command when the backend tells us which channel.
        teleop_sdk.init(
            message_received_callback=self._on_message_received,
            on_event_callback=self._on_event_cb,
        )

        # GStreamer pipeline
        self.pipeline = Gst.parse_launch(PIPELINE)
        self.appsrc = self.pipeline.get_by_name("source")
        # Let GStreamer stamp PTS on every pushed buffer from the pipeline's
        # running clock. Without this, x264enc sees buffers with pts=NONE/0
        # and can emit NAL units the decoder rejects as "no reference frame",
        # which most decoders render as a solid green field.
        self.appsrc.set_property("do-timestamp", True)
        self.appsink = self.pipeline.get_by_name("appsink")
        self.appsink.connect("new-sample", self._on_new_sample)

        # WS client
        self._ws_client: Optional[TeleArmsWSClient] = None
        self._init_ws_client()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------
    def _resolve_initial_task(self, env_cls: Optional[Type[RoboEvalEnv]]) -> TaskConfig:
        """Pick the TaskConfig to boot with, preferring the user-specified env_cls."""
        if env_cls is None:
            return get_random_task()

        for name, cls in ENVIRONMENTS.items():
            if cls == env_cls:
                task_config = get_task_by_name(name)
                if task_config is None:
                    # Env exists but no TaskConfig registered — synthesize one
                    task_config = TaskConfig(name=name, description=f"{name} task", enabled=True)
                return task_config

        print(f"[telearms] Warning: env_cls {env_cls} not in ENVIRONMENTS, falling back to random task")
        return get_random_task()

    def _save_initial_positions(self):
        """Capture arm pose at boot for _reset_arms()."""
        self.left_arm_initial_position = (
            np.array(self.gripper_l.wrist_position) if self.gripper_l else np.zeros(3)
        )
        self.right_arm_initial_position = (
            np.array(self.gripper_r.wrist_position) if self.gripper_r else np.zeros(3)
        )
        self.left_arm_initial_orientation = (
            Quaternion(self.gripper_l.wrist_orientation) if self.gripper_l else Quaternion()
        )
        self.right_arm_initial_orientation = (
            Quaternion(self.gripper_r.wrist_orientation) if self.gripper_r else Quaternion()
        )

    def _reset_arms_to_initial(self):
        """Restore both arms to the pose captured at boot."""
        if self.gripper_l:
            self.left_arm_target = self.left_arm_initial_position.copy()
            self.left_arm_orientation = Quaternion(self.left_arm_initial_orientation)
        if self.gripper_r:
            self.right_arm_target = self.right_arm_initial_position.copy()
            self.right_arm_orientation = Quaternion(self.right_arm_initial_orientation)

    def _init_ws_client(self):
        """Wire up TeleArmsWSClient and register command handlers."""
        instance_id = os.environ.get("INSTANCE_ID", f"roboplayground-{uuid.uuid4().hex[:8]}")
        self._ws_client = TeleArmsWSClient(
            ws_url=WS_SERVER_URL,
            instance_id=instance_id,
            capabilities=["simulation", "recording", "teleop"],
            on_connected=self._on_ws_connected,
            on_disconnected=self._on_ws_disconnected,
        )
        self._ws_client.register_command_handler("start", self._handle_start_command)
        self._ws_client.register_command_handler("stop", self._handle_stop_command)
        self._ws_client.register_command_handler("reset", self._handle_reset_command)
        self._ws_client.register_command_handler("preload", self._handle_preload_command)

        self._ws_client.start()
        if self._ws_client.wait_for_connection(timeout=10.0):
            print(f"[telearms] Connected to WS server as {instance_id}")
        else:
            print("[telearms] Warning: could not connect to WS server (will keep retrying)")

        global _active_teleop_instance
        _active_teleop_instance = self
        atexit.register(self._cleanup_on_exit)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # ------------------------------------------------------------------
    # WebRTC env wrapper
    # ------------------------------------------------------------------
    def _create_webrtc_env(self, env_cls: Type[RoboEvalEnv]) -> Type[RoboEvalEnv]:
        """Subclass env_cls so step() logs to demo_recorder and render() pushes H.264."""
        get_demo_recorder = lambda: self._demo_recorder
        get_on_render = lambda: self._on_render

        class TelearmsRoboEvalEnv(env_cls, ABC):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._demo_recorder = get_demo_recorder()
                self._on_render = get_on_render()

            def step(self, action: ActType, fast: bool = True):
                super().step(action, fast)
                timestep = ({}, self.reward, False, False, {})
                self._demo_recorder.add_timestep(timestep, action)
                return timestep

            @property
            def task_name(self) -> str:
                return self.__class__.__base__.__name__

        return TelearmsRoboEvalEnv

    # ------------------------------------------------------------------
    # WS command handlers
    # ------------------------------------------------------------------
    def _fetch_agora_token(self, task_id: str) -> Optional[dict]:
        url = f"{TELEARMS_BACKEND_URL}/api/v1/internal/tasks/{task_id}/agora_token"
        headers = {
            "Content-Type": "application/json",
            "X-Internal-Api-Key": TELEARMS_BACKEND_API_KEY,
        }
        try:
            print(f"[telearms] Fetching Agora token: {url}")
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            print(f"[telearms] Token fetch failed: HTTP {resp.status_code} — {resp.text}")
            return None
        except requests.RequestException as e:
            print(f"[telearms] Token fetch error: {e}")
            return None

    def _handle_start_command(self, params: dict) -> dict:
        task_id = params.get("task_id")
        config = params.get("config", {})
        # Accept the mission name under any of the historical key names the
        # backend has used; the canonical one in RoboEval is config.mission_name
        # but we've seen config.mission / config.env / config.env_name in logs.
        mission_name = (
            config.get("mission_name")
            or config.get("mission")
            or config.get("env")
            or config.get("env_name")
            or params.get("mission_name")
            or params.get("mission")
        )
        print(
            f"[telearms] start command: task_id={task_id} mission_name={mission_name} "
            f"param_keys={list(params.keys())} config_keys={list(config.keys())}"
        )

        if not task_id:
            print("[telearms] start rejected: missing task_id")
            return {"accepted": False, "error": "task_id is required"}
        if not mission_name:
            print(f"[telearms] start rejected: no mission in params/config → params={params}")
            return {"accepted": False, "error": "mission_name is required"}

        # Guard against double-start drift
        if (
            self._ws_client
            and self._ws_client.current_task_id
            and self._ws_client.current_task_id != task_id
        ):
            return {
                "accepted": False,
                "error": f"busy_with_other_task: current={self._ws_client.current_task_id}",
            }

        # (Re)load environment. We always recreate on start to avoid MuJoCo
        # stale-pointer segfaults between sessions.
        if mission_name != self._current_task.name:
            env_cls = ENVIRONMENTS.get(mission_name)
            if env_cls is None:
                available = ", ".join(sorted(ENVIRONMENTS.keys()))
                print(
                    f"[telearms] start rejected: unknown mission '{mission_name}'. "
                    f"Available missions in this playground build:\n  {available}"
                )
                return {"accepted": False, "error": f"Unknown mission_name: {mission_name}"}
            task_config = get_task_by_name(mission_name) or TaskConfig(
                name=mission_name, description=f"{mission_name} task", enabled=True
            )
            self._current_task = task_config

        try:
            self._setup_environment()
        except Exception as e:
            print(f"[telearms] start rejected: setup_environment failed: {e}")
            traceback.print_exc()
            return {"accepted": False, "error": f"setup_environment failed: {e}"}

        # Clear input state carried over from previous session
        self.pressed_keys.clear()
        self._gamepad_active = False

        # Agora handshake
        creds = self._fetch_agora_token(task_id)
        if not creds:
            return {"accepted": False, "error": "failed to fetch Agora credentials"}

        agora_channel = creds.get("agora_channel")
        agora_uid = creds.get("agora_uid")
        rtc_token = creds.get("agora_rtc_token")
        rtm_token = creds.get("agora_rtm_token")

        if not (agora_channel and rtc_token and rtm_token):
            return {"accepted": False, "error": "incomplete Agora credentials"}

        try:
            self._video_paused = True
            # Full teardown before reconnecting so the native SDK drops stale state
            try:
                teleop_sdk.disconnect()
            except Exception:
                pass
            try:
                teleop_sdk.deinit()
            except Exception:
                pass

            teleop_sdk.init(
                message_received_callback=self._on_message_received,
                on_event_callback=self._on_event_cb,
            )
            rtm_user_id = f"{agora_channel}-{agora_uid}"
            teleop_sdk.set_token(agora_channel, agora_uid, rtc_token, rtm_user_id, rtm_token)
            teleop_sdk.connect()
            self._video_paused = False
            print(f"[telearms] teleop_sdk connected → channel={agora_channel}, rtm_user={rtm_user_id}")
        except Exception as e:
            self._video_paused = False
            traceback.print_exc()
            return {"accepted": False, "error": f"teleop_sdk init failed: {e}"}

        self._task_start_time = time.time()
        self._frames_recorded = 0
        self._env_start_recording_event.set()
        return {"accepted": True, "task_id": task_id}

    def _handle_stop_command(self, params: dict) -> dict:
        task_id = params.get("task_id")
        reason = params.get("reason", "stop_requested")

        if (
            self._ws_client
            and self._ws_client.current_task_id
            and task_id != self._ws_client.current_task_id
        ):
            return {"accepted": False, "error": "stale_stop_command"}

        print(f"[telearms] Stop task={task_id} reason={reason}")

        self.pressed_keys.clear()
        self._gamepad_active = False

        # Save in-flight recording
        if self._demo_recorder and (
            self._demo_recorder.is_recording or self._demo_recorder._demo is not None
        ):
            try:
                self._save_recording()
                print(f"[telearms] Recording saved (frames={self._frames_recorded})")
            except Exception as e:
                print(f"[telearms] Failed to save recording: {e}")

        # Notify backend
        terminal_reasons = {"user_disconnect", "timeout", "error", "cancelled"}
        if self._ws_client and task_id and reason in terminal_reasons:
            self._ws_client.send_quit(task_id=task_id, reason=reason)

        # Drop Agora connection so we're ready for the next mission
        try:
            teleop_sdk.disconnect()
        except Exception as e:
            print(f"[telearms] Agora disconnect warning: {e}")

        self._task_start_time = None
        self._frames_recorded = 0

        try:
            with self._env_lock:
                self._env.reset()
            self._init_arm_state()
            self._save_initial_positions()
            self._gripper_toggle_state = {"left": 0.0, "right": 0.0}
        except Exception as e:
            print(f"[telearms] Env reset warning: {e}")

        return {"accepted": True}

    def _handle_reset_command(self, params: dict) -> dict:
        with self._env_lock:
            self._env.reset()
        self._init_arm_state()
        self._save_initial_positions()
        self._gripper_toggle_state = {"left": 0.0, "right": 0.0}
        return {"accepted": True}

    def _handle_preload_command(self, params: dict) -> dict:
        mission_name = params.get("mission_name")
        if not mission_name:
            return {"accepted": False, "error": "mission_name is required"}
        if mission_name == self._current_task.name:
            return {"accepted": True, "already_loaded": True}
        if self._ws_client and self._ws_client.current_task_id:
            return {"accepted": False, "error": "busy_with_active_task"}

        env_cls = ENVIRONMENTS.get(mission_name)
        if env_cls is None:
            return {"accepted": False, "error": f"Unknown mission_name: {mission_name}"}

        task_config = get_task_by_name(mission_name) or TaskConfig(
            name=mission_name, description=f"{mission_name} task", enabled=True
        )
        previous = self._current_task
        self._current_task = task_config
        try:
            self._setup_environment()
            return {"accepted": True}
        except Exception as e:
            self._current_task = previous
            try:
                self._setup_environment()
            except Exception:
                pass
            return {"accepted": False, "error": f"preload failed: {e}"}

    def _setup_environment(self):
        """Rebuild the env for the currently-selected task."""
        env_cls = ENVIRONMENTS.get(self._current_task.name)
        if env_cls is None:
            raise ValueError(f"Unknown env: {self._current_task.name}")

        with self._env_lock:
            # Close old env before replacing it — frees MuJoCo resources.
            try:
                if hasattr(self._env, "close"):
                    self._env.close()
            except Exception:
                pass

            webrtc_env_cls = self._create_webrtc_env(env_cls)
            self._env = webrtc_env_cls(
                render_mode="rgb_array",
                action_mode=self._action_mode,
                robot_cls=self._robot_cls,
            )
            self._env.mojo.model.vis.global_.offwidth = self._width
            self._env.mojo.model.vis.global_.offheight = self._height
            self._env.reset()

        self._stats = TelearmsTeleopStats()
        self._stop_countdown = None
        self._current_action = np.zeros(self._env.action_space.shape)
        self._ik = self._env.inverse_kinematics
        self._setup_grippers()
        self._init_arm_state()
        self._save_initial_positions()
        self._gripper_toggle_state = {"left": 0.0, "right": 0.0}
        self._prev_key_state = {
            self.GRIPPER_KEYS["left"]: False,
            self.GRIPPER_KEYS["right"]: False,
        }

    # ------------------------------------------------------------------
    # Scene-control RTM handlers
    # ------------------------------------------------------------------
    def _handle_reset_scene(self, data: dict):
        with self._env_lock:
            self._env.reset()
        self._init_arm_state()
        self._save_initial_positions()

    def _handle_change_task(self, data: dict):
        name = data.get("mission_name")
        if not name:
            return
        if self._ws_client and self._ws_client.current_task_id:
            print(f"[telearms] Ignoring change_task while {self._ws_client.current_task_id} active")
            return
        task_config = get_task_by_name(name)
        if task_config is None:
            print(f"[telearms] change_task: unknown {name}")
            return
        self._current_task = task_config
        self._setup_environment()

    # ------------------------------------------------------------------
    # GStreamer / Agora video out
    # ------------------------------------------------------------------
    def _on_ws_connected(self):
        # Start the pipeline as soon as WS is up. teleop_sdk.send_video() is
        # safe to no-op while disconnected from Agora (will just drop frames).
        self.pipeline.set_state(Gst.State.PLAYING)

    def _on_ws_disconnected(self):
        if self._demo_recorder and (
            self._demo_recorder.is_recording or self._demo_recorder._demo is not None
        ):
            try:
                self._save_recording()
                print("[telearms] Recording auto-saved on disconnect")
            except Exception as e:
                print(f"[telearms] Auto-save on disconnect failed: {e}")

    def _on_new_sample(self, sink):
        if self._video_paused:
            return Gst.FlowReturn.OK
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if ok:
            try:
                data = mapinfo.data
                # First few frames: log the raw NAL prefix to verify the
                # start-code layout we're stripping is actually 6 bytes.
                if self._frames_recorded < 3:
                    head = data[:12].hex() if len(data) >= 12 else data.hex()
                    print(f"[telearms] x264 out #{self._frames_recorded}: "
                          f"len={len(data)} prefix={head}")
                # Skip the 6-byte Annex-B start code header that GStreamer
                # prepends; teleop_sdk.send_video expects raw NAL units.
                teleop_sdk.send_video(data[6:])
                self._frames_recorded += 1
                if self._frames_recorded % 150 == 0:
                    print(f"[telearms] pushed {self._frames_recorded} frames to Agora")
            finally:
                buf.unmap(mapinfo)
        return Gst.FlowReturn.OK

    def _on_render(self, frame):
        if not hasattr(self, "_render_push_count"):
            self._render_push_count = 0

        # The pipeline's appsrc was created with caps saying 1280x720 RGB.
        # If the MuJoCo renderer produces a different size (gym's renderer
        # caches width/height at construction time and ignores later
        # model.vis.global_ tweaks) the x264enc reads the raw bytes with the
        # wrong stride and the browser decodes only green. Resize the frame
        # to match the pipeline caps before pushing.
        import numpy as _np
        expected_h, expected_w = HEIGHT, WIDTH
        if frame.shape[:2] != (expected_h, expected_w):
            if self._render_push_count < 3:
                print(f"[telearms] frame shape {frame.shape} != "
                      f"pipeline {expected_h}x{expected_w}, resizing")
            try:
                from PIL import Image as _Image
                pil = _Image.fromarray(frame)
                pil = pil.resize((expected_w, expected_h), _Image.BILINEAR)
                frame = _np.asarray(pil)
            except Exception as e:
                print(f"[telearms] resize failed: {e}; skipping frame")
                return

        if not frame.flags["C_CONTIGUOUS"]:
            frame = _np.ascontiguousarray(frame)

        data = frame.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        ret = self.appsrc.emit("push-buffer", buf)
        self._render_push_count += 1
        if self._render_push_count <= 3 or self._render_push_count % 150 == 0:
            print(f"[telearms] pushed render #{self._render_push_count} "
                  f"shape={frame.shape} bytes={len(data)} appsrc={ret}")

    def _on_event_cb(self, event, payload):
        print(f"[telearms] teleop_sdk event: {event} {payload}")

    # ------------------------------------------------------------------
    # Agora RTM input
    # ------------------------------------------------------------------
    def _on_message_received(self, msg: str):
        """Callback from teleop_sdk — every RTM message from the frontend."""
        # Ignore stale messages that arrive after task end but before UI disconnect
        if not self._ws_client or not self._ws_client.current_task_id:
            return
        try:
            action = json.loads(msg)
        except json.JSONDecodeError:
            return

        t = action.get("type")
        data = action.get("data", {})
        if t == "keyboard":
            self._gamepad_active = False
            new_pressed = {k for k, v in data.items() if v and k not in _IGNORED_FRONTEND_KEYS}
            backend_keys = self.pressed_keys & _IGNORED_FRONTEND_KEYS
            self.pressed_keys = new_pressed | backend_keys
        elif t == "gamepad":
            active = False
            axes = data.get("axes") or []
            for i, v in enumerate(axes[:4]):
                self.gamepad_axes[i] = float(v)
                if abs(self.gamepad_axes[i]) > 0.01:
                    active = True
            buttons = data.get("buttons") or []
            for i, v in enumerate(buttons):
                if i >= len(self.gamepad_buttons):
                    break
                self.gamepad_buttons[i] = float(v)
                if self.gamepad_buttons[i] > 0.01:
                    active = True
            self._gamepad_active = active
        elif t == "scene_control":
            # frontend command channel (reset_scene / change_task)
            self.scene_control_queue.put((data.get("action"), data))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, exit_event: Optional[threading.Event] = None,
            on_running_event: Optional[threading.Event] = None):
        if exit_event is None:
            exit_event = threading.Event()
        if on_running_event is None:
            on_running_event = threading.Event()
        on_running_event.set()

        try:
            while not exit_event.is_set():
                with self._env_lock:
                    if self._env is None:
                        pass  # mid-swap; sleep below
                    else:
                        self._handle_input()
                        self._apply_gamepad_deltas()
                        action = self._get_action()
                        for _ in range(self.STEPS_COUNT_FACTOR):
                            self._env.step(action, fast=True)
                        # In telearms mode the backend owns the recording
                        # lifecycle, so we do NOT run the reward-triggered
                        # termination countdown (which saves local demos).

                if self._env is None:
                    time.sleep(0.01)
                    continue

                self._render_frame()
        finally:
            self.shutdown()

    def _handle_input(self):
        """Drain scene-control queue and honor recording start signal."""
        if self._env_start_recording_event.is_set():
            self._start_recording_for_task()
            self._env_start_recording_event.clear()

        while not self.scene_control_queue.empty():
            try:
                action_name, data = self.scene_control_queue.get_nowait()
                handler = self.scene_control_handlers.get(action_name)
                if handler:
                    try:
                        handler(data)
                    except Exception as e:
                        print(f"[telearms] scene_control handler error ({action_name}): {e}")
                self.scene_control_queue.task_done()
            except pyqueue.Empty:
                break

    def _render_frame(self):
        """Render to offscreen buffer and push through the GStreamer pipeline."""
        self._update_stats()
        frame = self._env.render()
        if frame is not None:
            try:
                self._on_render(frame)
            except Exception as e:
                print(f"[telearms] render push failed: {e}")

    def _apply_gamepad_deltas(self):
        """Turn stick axes into pressed_keys deltas so the IK pipeline sees them."""
        if not self._gamepad_active:
            return

        # Map axes → virtual keys. Left stick drives left arm XY, right stick
        # drives right arm XY. Triggers (6/7) are Z up/down.
        dz = self.GAMEPAD_DEADZONE
        virtual = set()

        lx, ly = self.gamepad_axes[0], self.gamepad_axes[1]
        rx, ry = self.gamepad_axes[2], self.gamepad_axes[3]

        if lx < -dz:
            virtual.add(self.LEFT_ARM_KEYS["x_minus"])
        elif lx > dz:
            virtual.add(self.LEFT_ARM_KEYS["x_plus"])
        if ly > dz:
            virtual.add(self.LEFT_ARM_KEYS["y_minus"])
        elif ly < -dz:
            virtual.add(self.LEFT_ARM_KEYS["y_plus"])

        if rx < -dz:
            virtual.add(self.RIGHT_ARM_KEYS["x_minus"])
        elif rx > dz:
            virtual.add(self.RIGHT_ARM_KEYS["x_plus"])
        if ry > dz:
            virtual.add(self.RIGHT_ARM_KEYS["y_minus"])
        elif ry < -dz:
            virtual.add(self.RIGHT_ARM_KEYS["y_plus"])

        # L/R triggers (buttons 6/7) → Z movement
        if self.gamepad_buttons[6] > 0.5:
            virtual.add(self.LEFT_ARM_KEYS["z_minus"])
            virtual.add(self.RIGHT_ARM_KEYS["z_minus"])
        if self.gamepad_buttons[7] > 0.5:
            virtual.add(self.LEFT_ARM_KEYS["z_plus"])
            virtual.add(self.RIGHT_ARM_KEYS["z_plus"])

        # Shoulder buttons LB/RB (4/5) → gripper toggle edge-trigger
        for idx, side in ((4, "left"), (5, "right")):
            was_down = self._prev_gamepad_buttons[idx] > 0.5
            is_down = self.gamepad_buttons[idx] > 0.5
            if is_down and not was_down:
                virtual.add(self.GRIPPER_KEYS[side])

        self._prev_gamepad_buttons = list(self.gamepad_buttons)
        self.pressed_keys |= virtual

    # ------------------------------------------------------------------
    # Recording hooks
    # ------------------------------------------------------------------
    def _start_recording_for_task(self):
        """Start recording without touching the env — the backend already reset it."""
        if self._demo_recorder.is_recording:
            self._demo_recorder.stop()
        self._stop_countdown = None
        self._gripper_toggle_state = {"left": 0.0, "right": 0.0}
        self._prev_key_state = {
            self.GRIPPER_KEYS["left"]: False,
            self.GRIPPER_KEYS["right"]: False,
        }
        self._demo_recorder.record(self._env, lightweight_demo=True)

    def _save_recording(self, task_id: Optional[str] = None):
        if self._demo_recorder.is_recording:
            self._demo_recorder.stop()
        if self._demo_recorder.save_demo():
            self._stats.demos_counter += 1

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def _signal_handler(self, signum, frame):
        print(f"[telearms] Signal {signum}, shutting down")
        self.shutdown()
        os._exit(0)

    def _cleanup_on_exit(self):
        self.shutdown()

    def shutdown(self):
        if self._shutdown_done:
            return

        # Save any open recording first so we don't lose data.
        if self._demo_recorder and (
            self._demo_recorder.is_recording or self._demo_recorder._demo is not None
        ):
            try:
                self._save_recording()
            except Exception as e:
                print(f"[telearms] save-on-exit failed: {e}")

        self._video_paused = True

        if self._ws_client:
            try:
                self._ws_client.stop()
            except Exception:
                pass

        try:
            self.pipeline.set_state(Gst.State.NULL)
        except Exception:
            pass

        try:
            teleop_sdk.disconnect()
        except Exception:
            pass
        try:
            teleop_sdk.deinit()
        except Exception:
            pass

        self._shutdown_done = True
        print("[telearms] Shutdown complete")
