"""Fast smoke tests: the package imports and one env per task family runs.

Deliberately lightweight and network-free so it can gate every push/PR:
builds a representative env from each task family, reset/step/get_info, and
confirms the core modules import. Heavier demo-replay regression lives in
``examples/replay_sample_check_success.py`` (see the CI workflow).
"""
import importlib

import numpy as np
import pytest

from roboeval.action_modes import JointPositionActionMode
from roboeval.robots.configs.panda import BimanualPanda

CORE_MODULES = [
    "roboeval",
    "roboeval.action_modes",
    "roboeval.roboeval_env",
    "roboeval.demonstrations.demo",
    "roboeval.demonstrations.demo_converter",
    "roboeval.demonstrations.demo_store",
    "roboeval.utils.metric_rollout",
    "roboeval.robots.configs.panda",
]

# One representative env per task family.
ENV_SPECS = [
    ("roboeval.envs.lift_pot", "LiftPot"),
    ("roboeval.envs.manipulation", "CubeHandover"),
    ("roboeval.envs.manipulation", "StackTwoBlocks"),
    ("roboeval.envs.stack_books", "PickSingleBookFromTable"),
    ("roboeval.envs.pack_objects", "PackBox"),
    ("roboeval.envs.lift_tray", "LiftTray"),
    ("roboeval.envs.rotate_utility_objects", "RotateValve"),
]


@pytest.mark.parametrize("module", CORE_MODULES)
def test_core_module_imports(module):
    importlib.import_module(module)


@pytest.mark.parametrize("module,cls_name", ENV_SPECS, ids=[s[1] for s in ENV_SPECS])
def test_env_builds_resets_and_steps(module, cls_name):
    cls = getattr(importlib.import_module(module), cls_name)
    env = cls(
        action_mode=JointPositionActionMode(
            floating_base=True, absolute=True, floating_dofs=[]
        ),
        render_mode=None,
        control_frequency=20,
        robot_cls=BimanualPanda,
    )
    try:
        obs, info = env.reset(seed=0)
        assert obs is not None
        action = np.zeros_like(env.action_space.sample())
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step(action)
        # get_info finalizes metrics; must not raise.
        env.get_info()
    finally:
        env.close()
