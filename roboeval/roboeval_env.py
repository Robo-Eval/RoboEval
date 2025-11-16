"""Core RoboEval environment functionality."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type
from yaml import safe_load

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from mojo import Mojo
from mojo.elements import Geom, Camera, Site
from mojo.elements.consts import SiteType
from scipy.spatial.transform import Rotation

from roboeval.action_modes import ActionMode
from roboeval.const import WORLD_MODEL
from roboeval.envs.arenas.arena import Arena
from roboeval.envs.props.preset import Preset
from roboeval.robots.configs.panda import BimanualPanda
from roboeval.robots.robot import Robot
from roboeval.roboeval_renderer import RoboEvalRenderer
from roboeval.utils.callables_cache import CallablesCache
from roboeval.utils.env_health import EnvHealth
from roboeval.utils.observation_config import ObservationConfig
from roboeval.utils.spawn_boundary import SpawnBoundary

from roboeval.ik.base_ik import GenericUpperBodyIK

CONTROL_FREQUENCY_MAX = 500
CONTROL_FREQUENCY_MIN = 20

PHYSICS_DT = 0.002

MAX_DISTANCE_FROM_ORIGIN = 10
SPARSE_REWARD_FACTOR = 1


class RoboEvalEnv(gym.Env):
    """Core RoboEval environment which loads in common robot across all tasks.
    
    This environment provides a standardized interface for robot control tasks
    with support for various observation modes, action modes, and rendering options.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 1 / PHYSICS_DT,
    }

    _ENV_CAMERAS = ["external"]

    _MODEL_PATH: Path = WORLD_MODEL
    _PRESET_PATH: Optional[Path] = None
    _FLOOR = "floor"

    DEFAULT_ROBOT = BimanualPanda
    
    SUCCESS_EVAL_METRICS = False

    RESET_ROBOT_POS = np.array([0, 0, 0])
    RESET_ROBOT_QUAT = np.array([1, 0, 0, 0])

    def __init__(
        self,
        action_mode: ActionMode,
        observation_config: ObservationConfig = ObservationConfig(),
        render_mode: Optional[str] = None,
        start_seed: Optional[int] = None,
        control_frequency: int = CONTROL_FREQUENCY_MAX,
        robot_cls: Optional[Type[Robot]] = None,
        **kwargs,
    ):
        """Initialize the environment.

        :param action_mode: The action mode of the robot. Use this to configure how
            you plan to control the robot. E.g. joint position, delta ee pose, etc.
        :param observation_config: Observations configuration. Use this to configure
            collected data.
        :param render_mode: The render mode for mujoco. Options are
            "human", "rgb_array" or "depth_array". If None, the default render mode
            will be used.
        :param start_seed: The seed to start the environment with. If None, a random
            seed will be used.
        :param control_frequency: Control loop frequency, 500 Hz by default.
        :param robot_cls: Environment robot class override.
        """
        self._parse_kwargs(kwargs or {})
        # Tracks physics simulation stability
        self._env_health = EnvHealth()
        # Caches results valid for one environment step
        self._step_cache = CallablesCache()

        self._observation_config = observation_config
        self.action_mode = action_mode

        if start_seed is None:
            start_seed = np.random.randint(2**32)
        if not isinstance(start_seed, int):
            raise ValueError("Expected start_seed to be an integer.")
        self._next_seed = start_seed
        self._current_seed = None

        assert CONTROL_FREQUENCY_MIN <= control_frequency <= CONTROL_FREQUENCY_MAX, (
            f"Control frequency must be in "
            f"{CONTROL_FREQUENCY_MIN}-{CONTROL_FREQUENCY_MAX} range."
        )
        self._control_frequency = control_frequency
        self._sub_steps_count = int(
            np.round(CONTROL_FREQUENCY_MAX / self._control_frequency)
        )

        self._spawns_info = []
        self._robots_info = {}

        if self._PRESET_PATH is not None:
            with open(self._PRESET_PATH) as fhandle:
                config = safe_load(fhandle)
                # Check if the robot position is given in the presets YAML file
                if "robot" in config:
                    for robot in config["robot"]:
                        self._robots_info[robot["name"]] = dict(
                            position=robot.get("position", [0.0, 0.0, 0.0]),
                            euler=robot.get("euler", [0.0, 0.0, 0.0]),
                        )
                # Check if any spawn regions are defined in the YAML file
                if "spawns" in config:
                    for spawn in config["spawns"]:
                        self._spawns_info.append(
                            dict(
                                name=spawn["name"],
                                size=spawn["size"],
                                position=spawn["position"],
                                euler=spawn["euler"],
                            )
                        )

        self._mojo = Mojo(str(self._MODEL_PATH), timestep=PHYSICS_DT)

        self._mojo.root_element.mjcf.size.memory = str(50_000_000)  # 50 MB

        self._robot = (robot_cls or self.DEFAULT_ROBOT)(self.action_mode, self._mojo)
        self._preset = Preset(self._mojo, self._PRESET_PATH)
        self._arena = Arena(self._mojo, self._PRESET_PATH)

        if self._robot.config.model_name in self._robots_info:
            robot_info = self._robots_info[self._robot.config.model_name]
            self.RESET_ROBOT_POS = np.array(robot_info["position"])
            self.RESET_ROBOT_QUAT = Rotation.from_euler(
                "xyz", robot_info["euler"], degrees=True
            ).as_quat(scalar_first=True)

        self._spawns: Dict[str, Optional[SpawnBoundary]] = {}
        for spawn_info in self._spawns_info:
            # Create the site for this spawn area
            spawn_site = Site.create(
                self._mojo,
                parent=None,
                size=np.array(spawn_info["size"]),
                position=np.array(spawn_info["position"]),
                quaternion=Rotation.from_euler(
                    "xyz", spawn_info["euler"], degrees=True
                ).as_quat(scalar_first=True),
                color=np.array([0, 1, 1, 0.25]),
                site_type=SiteType.BOX,
                group=3,
            )
            # Build the spawn boundary object
            spawn_boundary = SpawnBoundary(self._mojo, spawn_site)
            self._spawns[spawn_info["name"]] = spawn_boundary

        self._initialize_env()
        self._floor = Geom.get(self._mojo, self._FLOOR)

        self.action_space: spaces.Box = self.action_mode.action_space(
            action_scale=self._sub_steps_count, seed=self._next_seed
        )
        self._action: np.ndarray = np.zeros_like(self.action_space.low)

        self.observation_space: spaces.Space = self.get_observation_space()

        assert self.metadata["render_modes"] == [
            "human",
            "rgb_array",
            "depth_array",
        ], self.metadata["render_modes"]

        self.render_mode = render_mode

        # Validate cameras configuration
        if type(self._robot.config.cameras) is dict:
            available_cameras = set(self._ENV_CAMERAS + list(self._robot.config.cameras.keys()))
        elif type(self._robot.config.cameras) is list:
            available_cameras = set(self._ENV_CAMERAS + self._robot.config.cameras)
        else:
            raise ValueError("Invalid cameras configuration.")
        
        for camera_config in self._observation_config.cameras:
            assert camera_config.name in available_cameras

        # Mapping original camera names to full identifiers
        self._cameras_map = self._initialize_cameras()

        self.mujoco_renderer: Optional[RoboEvalRenderer] = None
        self.obs_renderers: Optional[dict[tuple[int, int], mujoco.Renderer]] = {}
        self._initialize_renderers()

        self.inverse_kinematics = GenericUpperBodyIK(self, self._robot.ik_config)

    def _parse_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Parse additional keyword arguments.
        
        This method can be overridden by subclasses to handle custom arguments.
        
        Args:
            kwargs: Dictionary of additional arguments
        """
        pass

    @property
    def task_name(self) -> str:
        """Return the class name of the environment."""
        return self.__class__.__name__

    @property
    def _use_pixels(self) -> bool:
        """Return True if the environment uses pixel observations."""
        return len(self.observation_config.cameras) > 0

    @property
    def seed(self) -> Optional[int]:
        """Return the initial seed of the environment."""
        return self._current_seed

    @property
    def success(self) -> bool | Any:
        """Check if current step is successful."""
        if not self.SUCCESS_EVAL_METRICS:
            return bool(self._step_cache.get(self._success))
        else:
            return self._step_cache.get(self._success)

    @property
    def fail(self) -> bool:
        """Check if current step has failed."""
        return bool(self._step_cache.get(self._fail))

    @property
    def reward(self) -> float:
        """Get current step reward."""
        return float(self._step_cache.get(self._reward))

    @property
    def terminate(self) -> bool:
        """Get current step termination condition."""
        return bool(self.success or self.fail)

    @property
    def truncate(self) -> bool:
        """Get current step truncation condition."""
        return bool(not self.is_healthy)

    @property
    def is_healthy(self) -> bool:
        """Check if the simulation is currently healthy."""
        return bool(self._env_health.is_healthy)

    @property
    def observation_config(self) -> ObservationConfig:
        """Get the observation configuration."""
        return self._observation_config

    @property
    def control_frequency(self) -> int:
        """Return the control frequency of the environment."""
        return self._control_frequency

    @property
    def robot(self) -> Robot:
        """Get the robot instance."""
        return self._robot

    @property
    def floor(self) -> Geom:
        """Get environment floor geometry."""
        return self._floor

    @property
    def mojo(self) -> Mojo:
        """Get the Mojo instance."""
        return self._mojo

    @property
    def action(self) -> np.ndarray:
        """Get the last executed action."""
        return self._action.copy()

    def get_spawn_boundary(self, name: str) -> Optional[SpawnBoundary]:
        """Get the spawn boundary with the given name.
        
        Args:
            name: The name of the spawn boundary
            
        Returns:
            The spawn boundary object or None if not found
        """
        return self._spawns.get(name, None)

    def _initialize_renderers(self) -> None:
        """Initialize rendering objects for the environment."""
        self._close_renderers()
        self.mujoco_renderer: RoboEvalRenderer = RoboEvalRenderer(self._mojo)
        for camera_config in self._observation_config.cameras:
            resolution = camera_config.resolution
            if resolution in self.obs_renderers:
                continue
            self.obs_renderers[resolution] = mujoco.Renderer(
                self._mojo.model, resolution[0], resolution[1]
            )

    def _initialize_cameras(self) -> dict[str, tuple[int, Camera]]:
        """Initialize cameras for the environment.
        
        Returns:
            Dictionary mapping camera names to their IDs and Camera objects
        """
        cameras_map: dict[str, tuple[int, Camera]] = {}
        for camera_name in self._ENV_CAMERAS:
            camera: Camera = Camera.get(
                self._mojo, camera_name, self._mojo.root_element
            )
            cameras_map[camera_name] = (camera.id, camera)
        for robot_camera in self._robot.cameras:
            cameras_map[robot_camera.mjcf.name] = (
                robot_camera.id,
                robot_camera,
            )

        for camera_config in self._observation_config.cameras:
            _, camera = cameras_map[camera_config.name]
            if camera_config.pos is not None:
                camera.set_position(np.array(camera_config.pos))
            if camera_config.quat is not None:
                camera.set_quaternion(np.array(camera_config.quat))
        return cameras_map
    
    def _safe_close_renderer(self, renderer) -> None:
        """Safely close a renderer object, handling possible exceptions.
        
        Args:
            renderer: The renderer object to close
        """
        if renderer is not None:
            try:
                # In some cases renderer.close() will itself fail because underlying C resources are dead
                close_fn = getattr(renderer, 'close', None)
                if callable(close_fn):
                    close_fn()
            except Exception as e:
                print(f"Warning: Renderer close failed: {e}")

    def _close_renderers(self) -> None:
        """Close all renderers used by the environment."""
        self._safe_close_renderer(self.mujoco_renderer)
        for renderer in self.obs_renderers.values():
            self._safe_close_renderer(renderer)
        self.mujoco_renderer = None
        self.obs_renderers.clear()

    def _initialize_env(self) -> None:
        """Initialize environment-specific components.
        
        This method can be overridden by subclasses to add task-specific items.
        """
        pass

    def get_observation_space(self) -> spaces.Space:
        """Get the observation space for the environment.
        
        Returns:
            A gymnasium Space object defining the observation space
        """
        obs_dict = {}
        if self._observation_config.proprioception:
            obs_dict = {
                "proprioception": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(len(self._robot.qpos) + len(self._robot.qvel),),
                    dtype=np.float32,
                ),
                "proprioception_grippers": spaces.Box(
                    low=0,
                    high=1,
                    shape=(len(self.robot.qpos_grippers),),
                    dtype=np.float32,
                ),
            }
            if self.robot.floating_base:
                obs_dict.update(
                    {
                        "proprioception_floating_base": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(len(self.robot.floating_base.qpos),),
                            dtype=np.float32,
                        ),
                        "proprioception_floating_base_actions": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(
                                len(self.robot.floating_base.get_accumulated_actions),
                            ),
                            dtype=np.float32,
                        ),
                    }
                )
        if self._use_pixels:
            for camera in self.observation_config.cameras:
                if camera.rgb:
                    obs_dict[f"rgb_{camera.name}"] = spaces.Box(
                        low=0, high=255, shape=(3, *camera.resolution), dtype=np.uint8
                    )
                if camera.depth:
                    obs_dict[f"depth_{camera.name}"] = spaces.Box(
                        low=0,
                        high=1,
                        shape=camera.resolution,
                        dtype=np.float32,
                    )
        if self._observation_config.privileged_information:
            obs_dict.update(self._get_task_privileged_obs_space())
        return spaces.Dict(obs_dict)

    def _get_task_privileged_obs_space(self) -> dict[str, Any]:
        """Get the task-specific privileged observation space.
        
        This method should be overridden by subclasses to define task-specific
        privileged observation spaces.
        
        Returns:
            Dictionary of privileged observation spaces
        """
        return {}

    def get_observation(self) -> dict[str, np.ndarray]:
        """Get the current observation.
        
        Returns:
            Dictionary containing all observation components
        """
        obs = {}
        if self._observation_config.proprioception:
            obs |= self._get_proprioception_obs()
        if self._use_pixels:
            obs |= self._get_visual_obs()
        if self._observation_config.privileged_information:
            obs |= self._get_task_privileged_obs()
        return obs

    def _get_task_info(self) -> dict[str, Any]:
        """Get the task-specific info dictionary.
        
        This method should be overridden by subclasses to provide task-specific info.
        
        Returns:
            Dictionary of task-specific info
        """
        return {}

    def get_info(self) -> dict[str, Any]:
        """Get the info dictionary.
        
        Returns:
            Dictionary containing information about the current state
        """
        info = self._get_task_info()
        info.update({"task_success": float(self.success)})
        return info

    def _get_proprioception_obs(self) -> dict[str, Any]:
        """Get proprioception observations.
        
        Returns:
            Dictionary of proprioception observations
        """
        obs = {
            "proprioception": np.concatenate(
                [self._robot.qpos, self._robot.qvel]
            ).astype(np.float32),
            "proprioception_grippers": np.array(self.robot.qpos_grippers).astype(
                np.float32
            ),
        }
        if self.robot.floating_base:
            obs["proprioception_floating_base"] = np.array(
                self.robot.floating_base.qpos
            ).astype(np.float32)
            obs["proprioception_floating_base_actions"] = np.array(
                self.robot.floating_base.get_accumulated_actions
            ).astype(np.float32)
        return obs

    def _get_visual_obs(self) -> dict[str, Any]:
        """Get visual observations from cameras.
        
        Returns:
            Dictionary of visual observations
        """
        obs = {}
        for camera_config in self._observation_config.cameras:
            obs_renderer = self.obs_renderers[camera_config.resolution]
            obs_renderer.update_scene(
                self._mojo.data, self._cameras_map[camera_config.name][0]
            )
            if camera_config.rgb:
                rgb = obs_renderer.render()
                obs[f"rgb_{camera_config.name}"] = np.moveaxis(rgb, -1, 0)
            if camera_config.depth:
                obs_renderer.enable_depth_rendering()
                obs[f"depth_{camera_config.name}"] = obs_renderer.render()
                obs_renderer.disable_depth_rendering()
        return obs

    def _get_task_privileged_obs(self) -> dict[str, Any]:
        """Get task-specific privileged observations.
        
        This method should be overridden by subclasses to provide task-specific
        privileged observations.
        
        Returns:
            Dictionary of privileged observations
        """
        return {}

    def _update_seed(self, override_seed=None) -> None:
        """Update the seed for the environment.

        Args:
            override_seed: If not None, the next seed will be set to this value.
        """
        if override_seed is not None:
            if not isinstance(override_seed, int):
                logging.warning(
                    "Expected override_seed to be an integer. Casting to int."
                )
                override_seed = int(override_seed)
            self._next_seed = override_seed
            self.action_space = self.action_mode.action_space(
                action_scale=self._sub_steps_count, seed=override_seed
            )
        self._current_seed = self._next_seed
        assert self._current_seed is not None
        self._next_seed = np.random.randint(2**32)
        np.random.seed(self._current_seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment.

        Args:
            seed: If not None, the environment will be reset with this seed
            options: Additional information to specify how the environment is reset
        
        Returns:
            Tuple of (observation, info)
        """
        self._env_health.reset()
        self._update_seed(override_seed=seed)
        self._mojo.physics.reset()
        self._action = np.zeros_like(self._action)
        self._robot.set_pose(self.RESET_ROBOT_POS, self.RESET_ROBOT_QUAT)
        self._robot.reset_pose()
        self._on_reset()
        return self.get_observation(), self.get_info()

    def _on_reset(self) -> None:
        """Custom environment reset behavior.
        
        This method should be overridden by subclasses to provide task-specific
        reset behavior.
        """
        pass

    def _on_step(self) -> None:
        """Custom environment behavior after stepping.
        
        This method should be overridden by subclasses to provide task-specific
        step behavior.
        """
        pass

    def render(self):
        """Render a frame of the simulation.
        
        Returns:
            Rendered frame based on the specified render mode
        """
        return self.mujoco_renderer.render(self.render_mode)

    def step(
        self, action: Optional[np.ndarray] = None, fast: bool = False
    ) -> tuple[Any, float, bool, bool, dict]:
        """Step the environment.

        Args:
            action: Action to take. If None, only physics simulation will be applied
                without any robot actions (useful for letting objects fall due to gravity).
            fast: If True, perform the environment step without processing observations
                and return default values. Useful when performance is crucial,
                but observations are not required, e.g., demo collection in VR.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        if self.mujoco_renderer:
            if hasattr(self.mujoco_renderer.viewer, "request_reset"):
                if self.mujoco_renderer.viewer.request_reset:
                    self.mujoco_renderer.viewer.request_reset = False
                    self.reset()
            if hasattr(self.mujoco_renderer.viewer, "task_success"):
                self.mujoco_renderer.viewer.task_success = self.success
            if hasattr(self.mujoco_renderer.viewer, "task_fail"):
                self.mujoco_renderer.viewer.task_fail = self.fail
        self._step_cache.clean()
        
        if action is None:
            # Physics-only step (no robot action)
            with self._env_health.track():
                for _ in range(self._sub_steps_count):
                    self._mojo.step()
                    mujoco.mj_rnePostConstraint(self._mojo.model, self._mojo.data)
            self._action = np.zeros_like(self.action_space.low)
        else:
            # Normal step with robot action
            self._step_mujoco_simulation(action)
            self._action = action
            
        with self._env_health.track():
            self._on_step()
            
        if fast:
            return {}, 0, False, False, {}
        else:
            return (
                self.get_observation(),
                self.reward,
                self.terminate,
                self.truncate,
                self.get_info(),
            )

    def _step_mujoco_simulation(self, action: np.ndarray) -> None:
        """Step the MuJoCo simulation with the given action.
        
        Args:
            action: Action to apply
            
        Raises:
            ValueError: If action shape doesn't match the action space
        """
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Action shape mismatch: "
                f"expected {self.action_space.shape}, but got {action.shape}."
            )
        if np.any(action < self.action_space.low) or np.any(
            action > self.action_space.high
        ):
            clipped_action = np.clip(
                action, self.action_space.low, self.action_space.high
            )
            print(
                f"Action {action} is out of the action space bounds. "
                f"Overhead: {action - clipped_action}"
            )

        with self._env_health.track():
            for i in range(self._sub_steps_count):
                if i == 0:
                    self.action_mode.step(action, self.inverse_kinematics)
                else:
                    self._mojo.step()
                mujoco.mj_rnePostConstraint(self._mojo.model, self._mojo.data)

    def _success(self):
        """Check if the episode is successful.
        
        This method should be overridden by subclasses to define success criteria.
        
        Returns:
            True if successful, False otherwise
        """
        return False

    def _fail(self) -> bool:
        """Check if the episode has failed.
        
        Returns:
            True if failed, False otherwise
        """
        try:
            return (
                np.linalg.norm(self._robot.pelvis.get_position()) > MAX_DISTANCE_FROM_ORIGIN
            )
        except:
            return True

    def _reward(self) -> float:
        """Get current episode reward.
        
        Returns:
            Reward value
        """
        return float(self.success) * SPARSE_REWARD_FACTOR

    def close(self):
        """Close environment and free resources."""
        self._close_renderers()
