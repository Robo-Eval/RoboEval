"""Class to hold demo."""
from __future__ import annotations
import logging
import numpy as np
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from safetensors import safe_open
from safetensors.numpy import save_file
from typing import Optional, Any, Union, Iterable

from gymnasium.core import ActType

from roboeval.roboeval_env import RoboEvalEnv, CONTROL_FREQUENCY_MAX

from roboeval.demonstrations.utils import (
    Metadata,
    ObservationMode,
)

from roboeval.demonstrations.const import (
    ACTION_KEY,
    VISUAL_OBSERVATIONS_PREFIX,
    SAFETENSORS_SUFFIX,
    SAFETENSORS_OBSERVATION_PREFIX,
    SAFETENSORS_INFO_PREFIX,
    GYM_OBSERVATION_KEY,
    GYM_REWARD_KEY,
    GYM_TERMINATIION_KEY,
    GYM_TRUNCATION_KEY,
    GYM_INFO_KEY,
)

# Additional steps after termination/truncation
TERMINATION_STEPS = CONTROL_FREQUENCY_MAX * 2


@dataclass
class DemoStep:
    """Class to hold a time step."""

    observation: dict[str, Any]
    reward: float
    termination: bool
    truncation: bool
    info: dict[str, Any]

    def __init__(self, observation, reward, termination, truncation, info, action):
        """Init.

        Args:
            observation: A dictionary containing time step information.
            reward: The reward of the time step.
            termination: Whether the episode terminated.
            truncation: Whether the episode was truncated.
            info: A dictionary containing additional information.
            action: The action taken to reach the time step.
        """
        self.observation = observation
        self.reward = reward
        self.termination = termination
        self.truncation = truncation
        self.info = info
        self.set_executed_action(action)

    @property
    def executed_action(self) -> ActType:
        """Executed action."""
        return self.info.get(ACTION_KEY, None)

    def set_executed_action(self, action):
        """Set the executed action."""
        # Convert to float64 to reduce compounding errors
        self.info[ACTION_KEY] = np.float64(action)

    @property
    def has_visual_observations(self) -> bool:
        """Check if this timestep has visual observations."""
        return any(
            key.lower().startswith(VISUAL_OBSERVATIONS_PREFIX)
            for key in self.observation
        )

    @property
    def visual_observations(self) -> dict[str, np.ndarray]:
        """Get all visual observations of the current timestep."""
        visual_observations: dict[str, np.ndarray] = {}
        for key, observation in self.observation.items():
            if key.lower().startswith(VISUAL_OBSERVATIONS_PREFIX):
                visual_observations[key] = observation
        return visual_observations


class Demo:
    """Class to hold demo."""

    def __init__(
        self,
        metadata: Metadata,
        timesteps: Optional[list[DemoStep]] = None,
    ):
        """Init.

        Args:
            metadata: Metadata demo information.
            timesteps: A list of time steps.
        """
        self._metadata: Metadata = metadata
        if timesteps is not None:
            self._steps: list[DemoStep] = timesteps
        else:
            self._steps: list[DemoStep] = []

    @classmethod
    def from_safetensors(
        cls, demo_path: Path, override_metadata: Optional[Metadata] = None
    ) -> Optional[Demo]:
        """Load demo from a safetensors file.

        Args:
            demo_path(Path): Path to safetensors file.
            override_metadata(Metadata): Optional metadata override.

        Returns:
            A Demo object.
        """
        if isinstance(demo_path, str):
            demo_path = Path(demo_path)
        if not demo_path.suffix == SAFETENSORS_SUFFIX:
            demo_path = demo_path.with_suffix(SAFETENSORS_SUFFIX)
        if not demo_path.exists():
            logging.error(f"File {demo_path} does not exist.")
            return None
        metadata = override_metadata or Metadata.from_safetensors(demo_path)
        if metadata.observation_mode == ObservationMode.Lightweight:
            return LightweightDemo.from_safetensors(demo_path, override_metadata)
        demo = cls.load_timesteps_from_safetensors(demo_path)
        timesteps = [DemoStep(*step, step[-1][ACTION_KEY]) for step in demo]
        return cls(
            metadata=metadata,
            timesteps=timesteps,
        )

    @classmethod
    def from_env(cls, env: RoboEvalEnv) -> Demo:
        """Create a demo from an environment.

        Args:
            env: The environment to record.

        Returns:
            A Demo object.
        """
        return cls(
            metadata=Metadata.from_env(env),
        )

    @staticmethod
    def load_timesteps_from_safetensors(
        demo_path: Path,
    ):
        """Load timesteps from a safetensors file.

        Args:
            demo_path(Path): Path to safetensors file.

        Returns:
            List[Tuple(Dict[str, np.ndarray])]: a list of time steps.
        """
        demo_dict = {
            GYM_OBSERVATION_KEY: {},
            GYM_REWARD_KEY: None,
            GYM_TERMINATIION_KEY: None,
            GYM_TRUNCATION_KEY: None,
            GYM_INFO_KEY: {},
        }
        demo_path = Path(demo_path)
        logging.debug(f"Processing demo {demo_path}")
        with safe_open(demo_path, framework="np", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                t = f.get_tensor(key)
                if key.startswith(SAFETENSORS_OBSERVATION_PREFIX):
                    demo_dict[GYM_OBSERVATION_KEY][
                        key.removeprefix(SAFETENSORS_OBSERVATION_PREFIX)
                    ] = t
                elif key.startswith(SAFETENSORS_INFO_PREFIX):
                    demo_dict[GYM_INFO_KEY][
                        key.removeprefix(SAFETENSORS_INFO_PREFIX)
                    ] = t
                elif key in demo_dict:
                    demo_dict[key] = t
                else:
                    demo_dict[GYM_INFO_KEY][key] = t

        demo_length = len(demo_dict[GYM_INFO_KEY][ACTION_KEY])

        # Convert demo_dict
        #   from:   Dict[Dict[str, List[np.ndarray]]]
        #   to:     List[Tuple(Dict[str, np.ndarray])]
        demo = []

        def is_iterable(variable):
            return isinstance(variable, Iterable) and not isinstance(variable, str)

        for step_id in range(demo_length):
            demo_step_dict = {}
            for key, value in demo_dict.items():
                if isinstance(value, dict):
                    sub_dict = {}
                    for sub_key, sub_value in value.items():
                        sub_dict[sub_key] = (
                            sub_value[step_id]
                            if is_iterable(sub_value) and len(sub_value) == demo_length
                            else None
                        )
                    demo_step_dict[key] = sub_dict
                elif is_iterable(value) and len(value) > 0:
                    demo_step_dict[key] = value[step_id]
                else:
                    demo_step_dict[key] = value
            demo.append(tuple(demo_step_dict.values()))
        return demo

    @property
    def _saving_format(self):
        """Saving format.

        Returns:
            A dictionary containing timesteps ready to be saved.
        """
        to_save = {
            GYM_OBSERVATION_KEY: defaultdict(list),
            GYM_REWARD_KEY: [],
            GYM_TERMINATIION_KEY: [],
            GYM_TRUNCATION_KEY: [],
            GYM_INFO_KEY: defaultdict(list),
        }
        for step in self._steps:
            for key, val in step.observation.items():
                to_save[GYM_OBSERVATION_KEY][key].append(val)
            to_save[GYM_REWARD_KEY].append(step.reward)
            to_save[GYM_TERMINATIION_KEY].append(step.termination)
            to_save[GYM_TRUNCATION_KEY].append(step.truncation)
            for key, val in step.info.items():
                to_save[GYM_INFO_KEY][key].append(val)
        return to_save
    
    def debug_safetensors_entries(self,d):
        import tempfile, numpy as np
        from safetensors.numpy import save_file

        def _summ(v):
            import numpy as np
            t = type(v).__name__
            if hasattr(v, "detach") and hasattr(v, "cpu") and hasattr(v, "numpy"):
                try:
                    vv = v.detach().cpu().numpy()
                    return f"torch.Tensor -> np.ndarray shape={vv.shape} dtype={vv.dtype}"
                except Exception as e:
                    return f"torch.Tensor (error converting: {e})"
            if isinstance(v, np.ndarray):
                return f"np.ndarray shape={v.shape} dtype={v.dtype}"
            if isinstance(v, (list, tuple)):
                try:
                    arr = np.array(v)
                    return f"{t} -> np.array shape={arr.shape} dtype={arr.dtype}"
                except Exception as e:
                    return f"{t} (error np.array: {e})"
            return f"{t}"

        print("\n[Safetensors debug] Inspecting entries:")
        for k, v in d.items():
            print(f"  - {k}: {_summ(v)}")

        print("\n[Safetensors debug] Probing keys one-by-one:")
        bad = []
        for k, v in d.items():
            try:
                x = v
                # normalize torch -> numpy (no dtype changes)
                if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
                    x = x.detach().cpu().numpy()
                # allow lists/tuples so we can surface 'dtype=object' if it happens
                if isinstance(x, (list, tuple)):
                    x = np.array(x)  # if ragged, this becomes dtype=object (what we want to catch)

                # must be ndarray for numpy backend
                if not isinstance(x, np.ndarray):
                    raise TypeError(f"value is {type(v).__name__}, not np.ndarray")

                # try saving this single tensor
                with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=True) as tmp:
                    save_file({k: x}, tmp.name, {})  # will raise on unsupported dtype (e.g., object)
            except Exception as e:
                bad.append((k, str(e)))

        if not bad:
            print("\n[Safetensors debug] All individual keys saved fine. The issue may be in metadata types.")
        else:
            print("\n[Safetensors debug] Offending entries:")
            for k, err in bad:
                print(f"  * {k}: {err}")

    def save(self, path: Union[str, Path], debug: bool = False) -> Path:
        """Save a gymnasium demo to a file."""
        import json
        import numpy as np
        from pathlib import Path

        # ---------- helpers ----------
        def _union_keys_dict_series(seq):
            return list(sorted({k for d in seq if isinstance(d, dict) for k in d.keys()}))

        def _densify_bool_dict_series(seq, keys):
            idx = {k: i for i, k in enumerate(keys)}
            out = np.zeros((len(seq), len(keys)), dtype=np.bool_)
            for i, d in enumerate(seq):
                if not isinstance(d, dict):
                    continue
                for k, v in d.items():
                    out[i, idx[k]] = bool(v)
            return np.ascontiguousarray(out)

        def _densify_numeric_dict_series(seq, keys, dtype=np.float32, fill=np.nan):
            idx = {k: i for i, k in enumerate(keys)}
            out = np.full((len(seq), len(keys)), fill, dtype=dtype)
            for i, d in enumerate(seq):
                if not isinstance(d, dict):
                    continue
                for k, v in d.items():
                    out[i, idx[k]] = float(v)
            return np.ascontiguousarray(out)

        def _is_all_bool_values(seq):
            for d in seq:
                if isinstance(d, dict):
                    for v in d.values():
                        if not isinstance(v, (bool, np.bool_)):
                            return False
            return True

        def _is_all_int_values(seq):
            for d in seq:
                if isinstance(d, dict):
                    for v in d.values():
                        if not isinstance(v, (int, np.integer, bool, np.bool_)):
                            return False
            return True

        # ---------- build tensors ----------
        if isinstance(path, str):
            path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)

        float_dtype = np.float64  # keep your chosen precision
        timesteps = self._saving_format

        demo_dict = {
            f"{SAFETENSORS_OBSERVATION_PREFIX}{key}": (
                np.asarray(val, dtype=float_dtype)
                if np.issubdtype(np.asarray(val).dtype, np.floating)
                else np.asarray(val)
            )
            for key, val in timesteps[GYM_OBSERVATION_KEY].items()
        }
        demo_dict[GYM_REWARD_KEY] = np.asarray(timesteps[GYM_REWARD_KEY], dtype=float_dtype)
        demo_dict[GYM_TERMINATIION_KEY] = np.asarray(timesteps[GYM_TERMINATIION_KEY])
        demo_dict[GYM_TRUNCATION_KEY] = np.asarray(timesteps[GYM_TRUNCATION_KEY])
        demo_dict |= {
            f"{SAFETENSORS_INFO_PREFIX}{key}": (
                np.asarray(val, dtype=float_dtype)
                if np.issubdtype(np.asarray(val).dtype, np.floating)
                else np.asarray(val)
            )
            for key, val in timesteps[GYM_INFO_KEY].items()
        }

        # ---------- fix dtype=object dict-series ----------
        meta = dict(self.safetensor_metadata or {})
        for k, v in list(demo_dict.items()):
            if isinstance(v, np.ndarray) and v.dtype == object:
                # Normalize to Python list for inspection
                seq = v.tolist()
                # If this is a series of dicts, densify
                if any(isinstance(x, dict) for x in seq):
                    keys = _union_keys_dict_series(seq)
                    # choose dtype based on values
                    if _is_all_bool_values(seq):
                        dense = _densify_bool_dict_series(seq, keys)
                    else:
                        # integers -> int64 (fill 0), otherwise float32 (fill NaN)
                        if _is_all_int_values(seq):
                            dense = _densify_numeric_dict_series(seq, keys, dtype=np.int64, fill=0)
                        else:
                            dense = _densify_numeric_dict_series(seq, keys, dtype=np.float32, fill=np.nan)
                    demo_dict[k] = dense
                    # record columns; safetensors metadata values must be strings
                    meta[f"{k}_columns_json"] = json.dumps(keys)
                else:
                    # Not a dict-series (e.g., ragged lists). Move to sidecar JSON or handle separately.
                    # Here we keep it simple: drop to metadata sidecar.
                    meta.setdefault("sidecar_json_fields", "[]")
                    side_fields = json.loads(meta["sidecar_json_fields"])
                    side_fields.append(k)
                    meta["sidecar_json_fields"] = json.dumps(sorted(set(side_fields)))
                    # Remove from tensors — caller can write sidecar externally if desired
                    demo_dict.pop(k, None)

        if debug:
            self.debug_safetensors_entries(demo_dict)

        # ---------- ensure metadata is str->str ----------
        meta = {str(mk): (mv if isinstance(mv, str) else json.dumps(mv)) for mk, mv in meta.items()}

        # ---------- save ----------
        save_file(demo_dict, str(path), metadata=meta)
        logging.info(f"Saved {path}")
        return path

    @property
    def timesteps(self) -> list[DemoStep]:
        """Time steps."""
        return self._steps.copy()

    @property
    def duration(self) -> int:
        """Amount of demo steps."""
        return len(self._steps)

    @property
    def seed(self):
        """Seed of demo."""
        return self._metadata.seed

    @property
    def metadata(self) -> Metadata:
        """Metadata."""
        return self._metadata

    @property
    def uuid(self):
        """UUID."""
        return self._metadata.uuid

    @property
    def safetensor_metadata(self):
        """Metadata in safetensors format."""
        return self.metadata.ready_for_safetensors()

    def add_timestep(self, observation, reward, termination, truncation, info, action):
        """Add a time step to the recording.

        Args:
            observation: A dictionary containing time step information.
            reward: The reward of the time step.
            termination: Whether the episode terminated.
            truncation: Whether the episode was truncated.
            info: A dictionary containing additional information.
            action: The action taken to reach the time step.
        """
        timestep = DemoStep(observation, reward, termination, truncation, info, action)
        self._steps.append(timestep)

    def add_termination_steps(self, steps_count: int):
        """Duplicate last step multiple times."""
        steps = [deepcopy(self._steps[-1])] * steps_count
        self._steps.extend(steps)


class LightweightDemo(Demo):
    """Class to hold a lightweight demo."""

    def __init__(
        self,
        metadata: Metadata,
        timesteps=None,
    ):
        """Init.

        Args:
            metadata(dict): Metadata demo information.
            timesteps(list): A list of time steps.
        """
        super().__init__(metadata, timesteps)
        self._metadata.observation_mode = ObservationMode.Lightweight
        if timesteps is not None:
            self._steps: list[DemoStep] = LightweightDemo.lighten_timesteps(timesteps)
        else:
            self._steps: list[DemoStep] = []

    @classmethod
    def from_safetensors(
        cls, demo_path: Path, override_metadata: Optional[Metadata] = None
    ) -> Optional[Demo]:
        """Load demo from a safetensors file.

        Args:
            demo_path(Path): Path to safetensors file.
            override_metadata(Metadata): Optional metadata override.

        Returns:
            A Demo object.
        """
        if isinstance(demo_path, str):
            demo_path = Path(demo_path)
        if not demo_path.suffix == SAFETENSORS_SUFFIX:
            demo_path = demo_path.with_suffix(SAFETENSORS_SUFFIX)
        if not demo_path.exists():
            logging.error(f"File {demo_path} does not exist.")
            return None
        metadata = override_metadata or Metadata.from_safetensors(demo_path)
        if not metadata.observation_mode == ObservationMode.Lightweight:
            raise ValueError(
                f"Demo {demo_path} is not a lightweight demo. "
                "Use `Demo.from_safetensors` instead."
            )
        demo = cls.load_timesteps_from_safetensors(demo_path)
        timesteps = [DemoStep(*step, step[-1][ACTION_KEY]) for step in demo]
        return cls(
            metadata=metadata,
            timesteps=timesteps,
        )

    @classmethod
    def from_demo(cls, demo: Demo) -> LightweightDemo:
        """Create a lightweight demo from a demo.

        Args:
            demo: The demo to lighten.

        Returns:
            A LightweightDemo object.
        """
        return cls(
            metadata=deepcopy(demo.metadata),
            timesteps=deepcopy(demo.timesteps),
        )

    @classmethod
    def from_env(cls, env: RoboEvalEnv) -> Demo:
        """Create a demo from an environment.

        Args:
            env: The environment to record.

        Returns:
            A Demo object.
        """
        return cls(
            metadata=Metadata.from_env(env),
        )

    @property
    def _saving_format(self):
        """Saving format.

        Returns:
            A dictionary containing timesteps ready to be saved.
        """
        to_save = {
            GYM_OBSERVATION_KEY: defaultdict(list),
            GYM_REWARD_KEY: [],
            GYM_TERMINATIION_KEY: [],
            GYM_TRUNCATION_KEY: [],
            GYM_INFO_KEY: defaultdict(list),
        }
        for step in self._steps:
            to_save[GYM_TERMINATIION_KEY].append(step.termination)
            to_save[GYM_TRUNCATION_KEY].append(step.truncation)
            to_save[GYM_INFO_KEY][ACTION_KEY].append(step.executed_action)
        return to_save

    def add_timestep(self, observation, reward, termination, truncation, info, action):
        """Add a time step to the recording.

        Args:
            observation: A dictionary containing time step information.
            reward: The reward of the time step.
            termination: Whether the episode terminated.
            truncation: Whether the episode was truncated.
            info: A dictionary containing additional information.
            action: The action taken to reach the time step.
        """
        timestep = DemoStep({}, None, termination, truncation, {}, action)
        self._steps.append(timestep)

    @staticmethod
    def lighten_timesteps(timesteps: list[DemoStep]) -> list[DemoStep]:
        """Lighten the timesteps.

        Args:
            timesteps(list): A list of time steps.

        Returns:
            A list of time steps.
        """
        return [
            DemoStep(
                {}, None, step.termination, step.truncation, {}, step.executed_action
            )
            for step in timesteps
        ]
