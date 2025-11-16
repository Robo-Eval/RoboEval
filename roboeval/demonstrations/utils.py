"""Demonstration collection utils."""
from __future__ import annotations
import datetime
import json
import logging

import uuid
from dataclasses import dataclass, field, asdict, fields
from enum import Enum
from typing import Optional, Type

from pathlib import Path
import importlib.util
import importlib.metadata

from gymnasium import spaces
from safetensors import safe_open

import roboeval.action_modes as action_modes_module
import roboeval.envs as envs_module
import roboeval.robots.configs as robots_module

from roboeval.action_modes import (
    JointPositionActionMode,
    ActionMode,
    PelvisDof,
    DEFAULT_DOFS,
)
from roboeval.roboeval_env import RoboEvalEnv
from roboeval.robots.robot import Robot
from roboeval.utils.observation_config import ObservationConfig
from roboeval.utils.shared import find_class_in_module

from roboeval.demonstrations.const import TRACKED_PACKAGES


class ObservationMode(Enum):
    """Observation mode enum."""

    State = "state"
    Pixel = "pixel"
    Lightweight = "lightweight"


@dataclass
class Metadata:
    """RoboEval demonstration metadata."""

    observation_mode: ObservationMode
    environment_data: EnvData
    seed: int
    package_versions: dict[str, str]
    date: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
    )
    uuid: str = field(default_factory=lambda: uuid.uuid4().hex)

    @classmethod
    def from_safetensors(cls, demo_path: Path):
        """Get metadata from a safetensor file."""
        with safe_open(demo_path, framework="np", device="cpu") as f:
            metadata = f.metadata() or {}
        metadata = decode_safetensors_metadata(metadata)
        metadata["observation_mode"] = ObservationMode(metadata["observation_mode"])
        metadata["environment_data"] = EnvData.from_safetensors_metadata(
            metadata["environment_data"]
        )
        obj = cls(**clean_metadata(metadata, cls))
        obj._check_package_versions()
        return obj

    @classmethod
    def from_env(cls, env: RoboEvalEnv, is_lightweight: bool = False):
        """Create metadata from a RoboEval environment."""
        if is_lightweight:
            observation_mode = ObservationMode.Lightweight
        elif env.observation_config.cameras:
            observation_mode = ObservationMode.Pixel
        else:
            observation_mode = ObservationMode.State
        package_versions = {}
        for package in TRACKED_PACKAGES:
            package_versions[package] = get_package_version(package)
        return cls(
            seed=env.seed,
            observation_mode=observation_mode,
            environment_data=EnvData.from_env(env),
            package_versions=package_versions,
        )

    @classmethod
    def from_env_cls(
        cls,
        env_cls: type[RoboEvalEnv],
        action_mode: type[ActionMode],
        floating_base: bool = True,
        floating_dofs: Optional[list[str]] = None,
        obs_mode: ObservationMode = ObservationMode.Lightweight,
        observation_config: ObservationConfig = ObservationConfig(),
        action_mode_absolute: Optional[bool] = True,
        end_effector_mode: Optional[bool] = False,
    ):
        """Create metadata based on environment class."""
        floating_dofs = DEFAULT_DOFS if floating_dofs is None else floating_dofs
        if obs_mode == ObservationMode.Pixel:
            if not observation_config.cameras:
                raise ValueError("Pixel observation mode requires cameras.")
        return cls(
            seed=0,
            observation_mode=obs_mode,
            environment_data=EnvData(
                env_name=env_cls.__name__,
                action_mode_name=action_mode.__name__,
                floating_base=floating_base,
                floating_dofs=[dof.value for dof in floating_dofs],
                observation_config=observation_config,
                action_mode_absolute=action_mode_absolute,
                end_effector_mode=end_effector_mode,
                reset_positions=[],
                robot_name=env_cls.DEFAULT_ROBOT.__name__,
            ),
            package_versions={},
        )

    def ready_for_safetensors(self) -> dict:
        """Prepare metadata for safetensors."""
        return {
            "seed": json.dumps(self.seed),
            "observation_mode": json.dumps(self.observation_mode.value),
            "environment_data": json.dumps(asdict(self.environment_data)),
            "package_versions": json.dumps(self.package_versions),
            "date": json.dumps(self.date),
            "uuid": json.dumps(self.uuid),
        }

    @property
    def env_name(self) -> str:
        """Get environment name."""
        return self.environment_data.env_name

    @property
    def filename(self) -> str:
        """Create file name."""
        return f"{self.uuid}.safetensors"

    @property
    def floating_dof_count(self) -> int:
        """Count of floating DOFs."""
        if not self.environment_data.floating_base:
            return 0
        return len(self.environment_data.floating_dofs)

    @property
    def env_cls(self) -> Type[RoboEvalEnv]:
        """Get environment class."""
        return self.environment_data.env_cls

    @property
    def robot_cls(self) -> Type[Robot]:
        """Get robot class."""
        return self.environment_data.robot_cls

    @property
    def action_mode_cls(self) -> Type[ActionMode]:
        """Get action mode class."""
        return self.environment_data.action_mode_cls

    def _check_package_versions(self):
        """Check if the package versions are consistent with the current environment."""
        for package, saved_version in self.package_versions.items():
            installed_version = get_package_version(package)
            if saved_version != installed_version:
                logging.warning(
                    f"Installed version of {package}: {installed_version} doesn't "
                    f"match version stored in demo file: {saved_version}. Demo replay "
                    "could be unstable."
                )

    def get_env(
        self, control_frequency: int, render_mode: Optional[str] = None
    ) -> RoboEvalEnv:
        """Get environment based on metadata."""
        return self.env_cls(
            action_mode=self.get_action_mode(),
            observation_config=self.environment_data.observation_config,
            render_mode=render_mode,
            control_frequency=control_frequency,
            robot_cls=self.robot_cls,
        )

    def get_action_mode(self) -> ActionMode:
        """Get action mode based on metadata.

        Notes:
            - The action mode is not completely initialized
              until `ActionMode.bind_robot(robot)` is called.
        """
        action_mode_cls = self.action_mode_cls
        if issubclass(action_mode_cls, JointPositionActionMode):
            action_mode = action_mode_cls(
                absolute=self.environment_data.action_mode_absolute,
                floating_base=self.environment_data.floating_base,
                floating_dofs=[
                    PelvisDof(dof) for dof in self.environment_data.floating_dofs
                ],
                ee=self.environment_data.end_effector_mode,
            )
        else:
            action_mode = action_mode_cls(
                floating_base=self.environment_data.floating_base,
                floating_dofs=[
                    PelvisDof(dof) for dof in self.environment_data.floating_dofs
                ],
            )
        return action_mode

    def get_action_space(self, action_scale: float) -> spaces.Box:
        """Get action space based on metadata."""
        # ToDo: get rid of slow Robot instantiation
        robot = self.get_robot()
        return robot.action_mode.action_space(action_scale)

    def get_robot(self) -> Robot:
        """Get robot based on metadata."""
        return self.robot_cls(self.get_action_mode())


@dataclass
class EnvData:
    """RoboEval environment data."""

    env_name: str
    action_mode_name: str
    floating_base: bool
    observation_config: ObservationConfig
    end_effector_mode: bool = False
    action_mode_absolute: Optional[bool] = None
    floating_dofs: Optional[list[str]] = None
    reset_positions: Optional[list[float]] = None
    robot_name: Optional[str] = None

    def __post_init__(self):
        """Recovering missing data to support legacy demos."""
        if self.floating_dofs is None:
            self.floating_dofs = [dof.value for dof in DEFAULT_DOFS]
        if self.reset_positions is None:
            self.reset_positions = []
        if self.robot_name is None:
            self.robot_name = self.env_cls.DEFAULT_ROBOT.__name__

    @classmethod
    def from_safetensors_metadata(cls, metadata: dict):
        """Get metadata from a safetensor file."""
        metadata["observation_config"] = ObservationConfig.from_safetensors_metadata(
            metadata["observation_config"]
        )
        return cls(**clean_metadata(metadata, cls))

    @classmethod
    def from_env(cls, env: RoboEvalEnv):
        """Get data about RoboEval environment."""
        env_name = env.task_name
        action_mode_name = type(env.action_mode).__name__
        robot_config_name = type(env.robot).__name__
        absolute = (
            env.action_mode.absolute
            if isinstance(env.action_mode, JointPositionActionMode)
            else None
        )
        ee = (
            env.action_mode.ee
            if isinstance(env.action_mode, JointPositionActionMode)
            else None
        )
        floating_base = env.action_mode.floating_base
        floating_dofs = [dof.value for dof in env.action_mode.floating_dofs]
        reset_positions = (env.robot.qpos_actuated * 0).tolist()
        observation_config = env.observation_config
        return cls(
            env_name=env_name,
            action_mode_name=action_mode_name,
            action_mode_absolute=absolute,
            floating_base=floating_base,
            floating_dofs=floating_dofs,
            observation_config=observation_config,
            reset_positions=reset_positions,
            robot_name=robot_config_name,
            end_effector_mode=ee,
        )

    @property
    def action_mode_description(self) -> str:
        """Get unified description of the action mode."""
        parts = [self.action_mode_name]
        if self.floating_base:
            parts.append("floating")
            if self.floating_dofs:
                parts.extend(self.floating_dofs)
        if self.action_mode_absolute is not None:
            parts.append("absolute" if self.action_mode_absolute else "delta")
        if self.end_effector_mode is not None:
            parts.append("end_effector" if self.end_effector_mode else "joint")
        return "_".join(parts)

    @property
    def camera_description(self) -> str:
        """Get unified description of the cameras."""
        if not self.observation_config.cameras:
            return ""
        return "_".join(
            [camera.to_string() for camera in self.observation_config.cameras]
        )

    @property
    def env_cls(self) -> Type[RoboEvalEnv]:
        """Get environment class."""
        env_cls = find_class_in_module(envs_module, self.env_name)
        if env_cls is None:
            raise ValueError(f"Invalid environment name: {self.env_name}")
        if not issubclass(env_cls, RoboEvalEnv):
            raise ValueError(f"Invalid environment class: {env_cls}")
        return env_cls

    @property
    def action_mode_cls(self) -> Type[ActionMode]:
        """Get action mode class."""
        action_mode_cls = find_class_in_module(
            action_modes_module, self.action_mode_name
        )
        if action_mode_cls is None:
            raise ValueError(f"Invalid action mode name: {self.action_mode_name}")
        if not issubclass(action_mode_cls, ActionMode):
            raise ValueError(f"Invalid action mode class: {action_mode_cls}")
        return action_mode_cls

    @property
    def robot_cls(self) -> Type[Robot]:
        """Get robot config."""
        robot = find_class_in_module(robots_module, self.robot_name)
        if robot is None:
            raise ValueError(f"Invalid robot name: {self.robot_name}")
        if not issubclass(robot, Robot):
            raise ValueError(f"Invalid robot class: {robot}")
        return robot


def get_package_version(package_name: str) -> Optional[str]:
    """Get version of installed package."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def decode_safetensors_metadata(metadata: dict) -> dict:
    """Load metadata from a safetensor metadata dict recursively.

    Args:
        metadata (dict): Dictionary with metadata strings.

    Returns:
        dict: Dictionary with metadata.
    """
    for key, val in metadata.items():
        if isinstance(val, str):
            try:
                metadata[key] = json.loads(val)
            except ValueError:
                pass
        if isinstance(metadata[key], dict):
            metadata[key] = decode_safetensors_metadata(metadata[key])
    return metadata


def clean_metadata(metadata: dict, cls: Type[dataclass]) -> dict:
    """Remove unexpected fields from metadata dictionary."""
    field_names = {f.name for f in fields(cls)}
    return {k: v for k, v in metadata.items() if k in field_names}
