"""Submission bundle: metadata.yaml, the policy interface, and env construction.

The evaluator builds every environment from the ``interface`` block of
``metadata.yaml``. What a team declares is therefore exactly what runs, and the
validator can rebuild the same environment to replay logged actions.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import yaml

from roboeval.challenge import spec

REQUIRED_PATHS = (
    "results.json",
    "report.pdf",
    "metadata.yaml",
    "checkpoint",
    "policy/policy.py",
    "policy/requirements.txt",
    "logs",
)

REQUIRED_METADATA_FIELDS = (
    "team_name",
    "method_name",
    "contact_email",
    "affiliation",
    "members",
    "roboeval_release",
    "interface",
    "policy_seed",
    "training_data",
    "training_gpu_hours",
    "open_source",
)

INSTRUCTIONS_DIR = Path(__file__).resolve().parent.parent / "envs" / "instructions"


# ─────────────────────────────── interface ───────────────────────────────
@dataclass(frozen=True)
class CameraSpec:
    name: str
    resolution: tuple = (128, 128)
    rgb: bool = True
    depth: bool = False


@dataclass(frozen=True)
class Interface:
    """Declared action/observation interface; drives environment construction."""

    absolute: bool
    ee: bool
    control_frequency: int = spec.DEFAULT_CONTROL_FREQUENCY
    proprioception: bool = True
    cameras: tuple = ()

    @property
    def observation_mode(self) -> str:
        """Leaderboard grouping: image-based vs. proprioception-only."""
        return "vision" if self.cameras else "state"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Interface":
        action = d.get("action_mode", {})
        obs = d.get("observation", {})
        cameras = tuple(
            CameraSpec(
                name=c["name"],
                resolution=tuple(c.get("resolution", (128, 128))),
                rgb=bool(c.get("rgb", True)),
                depth=bool(c.get("depth", False)),
            )
            for c in obs.get("cameras", []) or []
        )
        return cls(
            absolute=bool(action["absolute"]),
            ee=bool(action["ee"]),
            control_frequency=int(d.get("control_frequency", spec.DEFAULT_CONTROL_FREQUENCY)),
            proprioception=bool(obs.get("proprioception", True)),
            cameras=cameras,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_mode": {"absolute": self.absolute, "ee": self.ee},
            "control_frequency": self.control_frequency,
            "observation": {
                "proprioception": self.proprioception,
                "cameras": [
                    {"name": c.name, "resolution": list(c.resolution), "rgb": c.rgb, "depth": c.depth}
                    for c in self.cameras
                ],
            },
        }

    def problems(self) -> List[str]:
        """Return human-readable reasons this interface cannot be evaluated."""
        out = []
        cf = self.control_frequency
        if not 20 <= cf <= 500:
            out.append(f"control_frequency {cf} Hz is outside the supported 20-500 Hz range")
        elif 500 % cf:
            out.append(
                f"control_frequency {cf} Hz does not divide the 500 Hz physics rate; "
                f"the simulator would actually run at {500 / round(500 / cf):.2f} Hz"
            )
        for c in self.cameras:
            if c.name not in spec.AVAILABLE_CAMERAS:
                out.append(f"unknown camera '{c.name}' (available: {', '.join(spec.AVAILABLE_CAMERAS)})")
            if len(c.resolution) != 2:
                out.append(f"camera '{c.name}' resolution must be [height, width]")
            if not (c.rgb or c.depth):
                out.append(f"camera '{c.name}' requests neither rgb nor depth")
        if not self.proprioception and not self.cameras:
            out.append("observation is empty: enable proprioception or at least one camera")
        return out


def load_metadata(submission_dir: Path) -> Dict[str, Any]:
    with open(Path(submission_dir) / "metadata.yaml") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("metadata.yaml must be a mapping")
    return data


def load_interface(submission_dir: Path) -> Interface:
    return Interface.from_dict(load_metadata(submission_dir)["interface"])


# ───────────────────────────── environments ──────────────────────────────
def make_env(family: str, variation: str, interface: Interface):
    """Construct the challenge environment for one task variation."""
    from roboeval.action_modes import JointPositionActionMode
    from roboeval.robots.configs.panda import BimanualPanda
    from roboeval.utils.observation_config import CameraConfig, ObservationConfig

    observation_config = ObservationConfig(
        cameras=[
            CameraConfig(name=c.name, rgb=c.rgb, depth=c.depth, resolution=tuple(c.resolution))
            for c in interface.cameras
        ],
        proprioception=interface.proprioception,
    )
    env_cls = spec.env_class(family, variation)
    return env_cls(
        action_mode=JointPositionActionMode(
            floating_base=True,
            floating_dofs=[],
            absolute=interface.absolute,
            ee=interface.ee,
        ),
        observation_config=observation_config,
        render_mode=None,
        control_frequency=interface.control_frequency,
        robot_cls=BimanualPanda,
    )


def task_instruction(family: str) -> str:
    """Canonical language instruction for a task family (first listed phrasing)."""
    path = INSTRUCTIONS_DIR / spec.FAMILIES_BY_KEY[family].instruction_file
    with open(path) as f:
        return json.load(f)["instructions"][0]


# ──────────────────────────────── policy ─────────────────────────────────
@dataclass
class EpisodeSpec:
    """Passed to ``Policy.reset`` at the start of every evaluation episode."""

    family: str
    variation: str
    env_name: str
    instruction: str
    episode_index: int
    seed: int
    policy_seed: int
    control_frequency: int
    max_steps: int
    action_space: Any = field(repr=False, default=None)


class Policy(Protocol):
    """What ``policy/policy.py:make_policy`` must return."""

    def reset(self, episode: EpisodeSpec) -> None:
        ...

    def act(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        ...


def load_policy(submission_dir: Path, metadata: Optional[Dict[str, Any]] = None) -> Policy:
    """Import ``policy/policy.py`` and call its ``make_policy``.

    ``make_policy(checkpoint_dir: Path, metadata: dict) -> Policy``
    """
    submission_dir = Path(submission_dir).resolve()
    policy_dir = submission_dir / "policy"
    metadata = metadata if metadata is not None else load_metadata(submission_dir)
    # Let policy.py import sibling modules from policy/.
    if str(policy_dir) not in sys.path:
        sys.path.insert(0, str(policy_dir))
    module_spec = importlib.util.spec_from_file_location("submission_policy", policy_dir / "policy.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    if not hasattr(module, "make_policy"):
        raise AttributeError("policy/policy.py must define make_policy(checkpoint_dir, metadata)")
    policy = module.make_policy(submission_dir / "checkpoint", metadata)
    for method in ("reset", "act"):
        if not callable(getattr(policy, method, None)):
            raise TypeError(f"make_policy() returned an object without a callable .{method}()")
    return policy
