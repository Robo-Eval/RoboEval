"""Tool to replay demonstrations."""
from __future__ import annotations

from typing import Optional

from roboeval.roboeval_env import CONTROL_FREQUENCY_MAX, RoboEvalEnv
from roboeval.demonstrations.demo import Demo, DemoStep
from roboeval.demonstrations.demo_converter import DemoConverter

import copy


class DemoPlayer:
    """Tool to replay demonstrations."""

    @staticmethod
    def replay(
        demo: Demo,
        control_frequency: int,
        demo_frequency: int = CONTROL_FREQUENCY_MAX,
        render_mode: Optional[str] = None,
    ):
        """Replay demonstration in original environment."""
        env = demo.metadata.get_env(control_frequency, render_mode=render_mode)
        DemoPlayer.replay_in_env(demo, env, demo_frequency)

    @staticmethod
    def replay_in_env(
        demo: Demo, env: RoboEvalEnv, demo_frequency: int = CONTROL_FREQUENCY_MAX
    ):
        """Replay demonstration in environment."""
        timesteps = DemoPlayer._get_timesteps_for_replay(demo, env, demo_frequency)
        env.reset(seed=demo.seed)
        for step in timesteps:
            action = step.executed_action
            env.step(action, fast=True)
            if env.render_mode:
                env.render()
        # env.close()
    
    @staticmethod
    def replay_in_env_save_position_orientation(
        demo: Demo, env: RoboEvalEnv, demo_frequency: int = CONTROL_FREQUENCY_MAX
    ):
        """Replay demonstration in environment."""
        timesteps = DemoPlayer._get_timesteps_for_replay(demo, env, demo_frequency)
        env.reset(seed=demo.seed)
        position_lefts = []
        position_rights = []
        orientations_left = []
        orientations_right = []
        
        for step in timesteps:
            action = step.executed_action
            env.step(action, fast=True)
            if env.render_mode:
                env.render()


            

            for hand_side, site_obj in env.robot._wrist_sites.items():
                site_id = site_obj.id
                site_name = site_obj.mjcf.name

                site_position = env.mojo.data.site_xpos[site_id]
                site_orientation = env.mojo.data.site_xmat[site_id]
                
                if 'left' in site_name:
                    position_lefts.append(copy.deepcopy(site_position))
                    orientations_left.append(copy.deepcopy(site_orientation))
                elif 'right' in site_name:
                    position_rights.append(copy.deepcopy(site_position))
                    orientations_right.append(copy.deepcopy(site_orientation))
                else:
                    raise ValueError(f"Unknown wrist site name: {site_name}")
                
        return position_lefts, position_rights, orientations_left, orientations_right

    @staticmethod
    def validate(
        demo: Demo,
        control_frequency: int,
        demo_frequency: int = CONTROL_FREQUENCY_MAX,
    ) -> bool:
        """Replay demonstration in original environment."""
        env = demo.metadata.get_env(control_frequency)
        return DemoPlayer.validate_in_env(demo, env, demo_frequency)

    @staticmethod
    def validate_in_env(
        demo: Demo, env: RoboEvalEnv, demo_frequency: int = CONTROL_FREQUENCY_MAX
    ) -> bool:
        """Check if demonstration is successful in environment."""
        timesteps = DemoPlayer._get_timesteps_for_replay(demo, env, demo_frequency)
        env.reset(seed=demo.seed)
        is_successful = False
        for step in timesteps:
            action = step.executed_action
            env.step(action, fast=True)
            if env.reward > 0:
                is_successful = True
                break
        env.close()
        return is_successful

    @staticmethod
    def _get_timesteps_for_replay(
        demo: Demo, env: RoboEvalEnv, demo_frequency: int = CONTROL_FREQUENCY_MAX
    ) -> list[DemoStep]:
        if env.control_frequency != demo_frequency:
            timesteps = DemoConverter.decimate(
                demo,
                target_freq=env.control_frequency,
                original_freq=demo_frequency,
                robot=env.robot,
            ).timesteps
        else:
            timesteps = demo.timesteps
        return timesteps
