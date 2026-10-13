"""Template policy for the RoboEval CoRL 2026 challenge.

The evaluator imports this file and calls ``make_policy(checkpoint_dir, metadata)``
once per worker process. Replace ``HoldPolicy`` with your model; keep the
``reset``/``act`` signatures.

This placeholder keeps both arms still (zero end-effector deltas) and holds the
grippers where they are. It scores ~0 success but exercises the full
evaluate -> validate -> package pipeline, so run it once before plugging in
your own model.
"""
from pathlib import Path
from typing import Any, Dict

import numpy as np


class HoldPolicy:
    def __init__(self, checkpoint_dir: Path, metadata: Dict[str, Any]):
        # Load your weights from checkpoint_dir here, e.g.
        #   self.model = torch.load(checkpoint_dir / "model.pt", map_location="cuda")
        self.checkpoint_dir = checkpoint_dir
        self.action_space = None

    def reset(self, episode) -> None:
        """Called before every episode.

        ``episode`` carries family, variation, env_name, instruction (language
        goal), seed, policy_seed, control_frequency, max_steps and action_space.
        """
        self.action_space = episode.action_space

    def act(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Map one observation dict to one action of shape action_space.shape.

        Keys: 'proprioception' (joint qpos+qvel), 'proprioception_grippers', and
        'rgb_<camera>' (C, H, W uint8) / 'depth_<camera>' for each declared camera.
        """
        action = np.zeros(self.action_space.shape, dtype=np.float32)
        # ee-delta layout: [left xyz, left rotvec, right xyz, right rotvec, left grip, right grip]
        action[-2:] = observation["proprioception_grippers"]
        return action


def make_policy(checkpoint_dir: Path, metadata: Dict[str, Any]) -> HoldPolicy:
    return HoldPolicy(checkpoint_dir, metadata)
