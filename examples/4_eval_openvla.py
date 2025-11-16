"""
OpenVLA Evaluation Script for Bimanual Robot Tasks

This script evaluates OpenVLA models on bimanual robot manipulation tasks,
with support for both demonstration replay and model inference modes.

Usage Examples:
    # Model inference mode:
    python 4_eval_openvla.py --ckpt_path /path/to/model/checkpoint

    # Demonstration replay mode:
    python 4_eval_openvla.py --ckpt_path /path/to/model/checkpoint \
                             --use_demos --dataset_path /path/to/demo/dataset

    # Custom configuration:
    python 4_eval_openvla.py --ckpt_path /path/to/model/checkpoint \
                             --instruction "pick up the red object" \
                             --num_episodes 10 --max_steps 300 \
                             --output_dir /path/to/output/videos

Required Arguments:
    --ckpt_path: Path to the OpenVLA model checkpoint directory

Optional Arguments:
    --dataset_path: Path to demonstration dataset (required if --use_demos is set)
    --use_demos: Use demonstration replay instead of model inference
    --instruction: Task instruction for the robot
    --device: Device for model inference (default: cuda:0)
    --downsample_rate: Control frequency downsampling factor (default: 25)
    --max_steps: Maximum steps per episode (default: 200)
    --num_episodes: Number of episodes to run (default: 5)
    --fps: FPS for output videos (default: 5)
    --output_dir: Output directory for videos (default: checkpoint directory)
"""

import argparse
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from roboeval.action_modes import JointPositionActionMode
from roboeval.demonstrations.demo_converter import DemoConverter
from roboeval.demonstrations.demo_store import DemoStore
from roboeval.demonstrations.utils import Metadata
from roboeval.envs.lift_pot import LiftPot
from roboeval.roboeval_env import CONTROL_FREQUENCY_MAX
from roboeval.robots.configs.panda import BimanualPanda
from roboeval.utils.observation_config import CameraConfig, ObservationConfig

try:
    from moviepy.editor import VideoClip
except ImportError:
    raise ImportError("Install moviepy for video preview: pip install moviepy pygame")


# Configuration constants
DEFAULT_DEVICE = "cuda:0"
DEFAULT_DOWNSAMPLE_RATE = 25
DEFAULT_MAX_STEPS = 200
DEFAULT_NUM_EPISODES = 5
DEFAULT_FPS = 5
CAMERA_RESOLUTION = (256, 256)
MIN_REWARD_THRESHOLD = 0.25


@dataclass
class EvaluationConfig:
    """Configuration for OpenVLA evaluation."""
    ckpt_path: str = None
    dataset_path: str = None
    use_demos: bool = False
    instruction: str = "reach the red sphere with the left hand and the green sphere with the right hand"
    device: str = DEFAULT_DEVICE
    downsample_rate: int = DEFAULT_DOWNSAMPLE_RATE
    max_steps: int = DEFAULT_MAX_STEPS
    num_episodes: int = DEFAULT_NUM_EPISODES
    fps: int = DEFAULT_FPS
    output_dir: str = None  # Optional output directory for videos

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'EvaluationConfig':
        """Create config from command line arguments."""
        return cls(
            ckpt_path=args.ckpt_path,
            dataset_path=args.dataset_path,
            use_demos=args.use_demos,
            instruction=args.instruction,
            device=args.device,
            downsample_rate=args.downsample_rate,
            max_steps=args.max_steps,
            num_episodes=args.num_episodes,
            fps=args.fps,
            output_dir=args.output_dir
        )

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.ckpt_path is None:
            raise ValueError("ckpt_path must be specified")
        if self.use_demos and self.dataset_path is None:
            raise ValueError("dataset_path must be specified when use_demos=True")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate OpenVLA models on bimanual robot tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        required=True,
        help="Path to the OpenVLA model checkpoint directory"
    )
    
    # Optional arguments
    parser.add_argument(
        "--dataset_path", 
        type=str,
        help="Path to the demonstration dataset (required if --use_demos is set)"
    )
    
    parser.add_argument(
        "--use_demos", 
        action="store_true",
        help="Use demonstration replay instead of model inference"
    )
    
    parser.add_argument(
        "--instruction", 
        type=str,
        default="reach the red sphere with the left hand and the green sphere with the right hand",
        help="Task instruction for the robot"
    )
    
    parser.add_argument(
        "--device", 
        type=str,
        default=DEFAULT_DEVICE,
        help="Device for model inference"
    )
    
    parser.add_argument(
        "--downsample_rate", 
        type=int,
        default=DEFAULT_DOWNSAMPLE_RATE,
        help="Control frequency downsampling factor"
    )
    
    parser.add_argument(
        "--max_steps", 
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum number of steps per episode"
    )
    
    parser.add_argument(
        "--num_episodes", 
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help="Number of episodes to run"
    )
    
    parser.add_argument(
        "--fps", 
        type=int,
        default=DEFAULT_FPS,
        help="FPS for output videos"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str,
        help="Output directory for videos (defaults to checkpoint directory)"
    )
    
    return parser.parse_args()


def register_openvla() -> None:
    """Register OpenVLA components with the Transformers library."""
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def load_vla_model(ckpt_path: str, device: str = DEFAULT_DEVICE) -> Tuple[AutoProcessor, AutoModelForVision2Seq]:
    """
    Load OpenVLA model and processor from checkpoint.
    
    Args:
        ckpt_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        Tuple of (processor, model)
        
    Raises:
        FileNotFoundError: If checkpoint path or dataset statistics file doesn't exist
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")
    
    stats_path = os.path.join(ckpt_path, "dataset_statistics.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Dataset statistics file not found: {stats_path}")
    
    processor = AutoProcessor.from_pretrained(ckpt_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        ckpt_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)

    with open(stats_path, "r") as f:
        model.norm_stats = json.load(f)

    return processor, model


def setup_liftpot_env(downsample_rate: int = DEFAULT_DOWNSAMPLE_RATE) -> LiftPot:
    """
    Set up the LiftPot environment with bimanual robot configuration.
    
    Args:
        downsample_rate: Control frequency downsampling factor
        
    Returns:
        Configured LiftPot environment
    """
    return LiftPot(
        action_mode=JointPositionActionMode(
            floating_base=True, 
            absolute=True, 
            block_until_reached=False, 
            ee=True, 
            floating_dofs=[]
        ),
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig(name="head", rgb=True, depth=False, resolution=CAMERA_RESOLUTION),
                CameraConfig(name="left_wrist", rgb=True, depth=False, resolution=CAMERA_RESOLUTION),
                CameraConfig(name="right_wrist", rgb=True, depth=False, resolution=CAMERA_RESOLUTION),
            ]
        ),
        render_mode=None,
        robot_cls=BimanualPanda,
        control_frequency=CONTROL_FREQUENCY_MAX // downsample_rate,
    )


def get_successful_demos(
    env: LiftPot, 
    dataset_path: str, 
    downsample_rate: int, 
    num_demos: int = 1
) -> List:
    """
    Load successful demonstrations from the dataset.
    
    Args:
        env: Environment instance for metadata
        dataset_path: Path to the demonstration dataset
        downsample_rate: Frequency downsampling factor
        num_demos: Number of demonstrations to load
        
    Returns:
        List of successful demonstrations (reward > threshold)
        
    Raises:
        FileNotFoundError: If dataset path doesn't exist
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    demo_store = DemoStore()
    demos = demo_store.get_demos_from_folder(
        dataset_path,
        Metadata.from_env(env),
        amount=num_demos,
        frequency=CONTROL_FREQUENCY_MAX // downsample_rate,
    )
    
    successful_demos = [
        copy.deepcopy(demo)
        for demo in demos
        if sum(step.reward for step in demo.timesteps) > MIN_REWARD_THRESHOLD
    ]
    
    print(f"Found {len(successful_demos)} successful demos out of {len(demos)} total")
    return successful_demos


def run_inference_loop(
    env: LiftPot,
    processor: AutoProcessor,
    model: AutoModelForVision2Seq,
    instruction: str,
    use_demos: bool = False,
    demo: Optional[object] = None,
    device: str = DEFAULT_DEVICE,
    max_steps: int = DEFAULT_MAX_STEPS
) -> List[np.ndarray]:
    """
    Run the agent in a closed loop using either demo replay or OpenVLA predictions.
    
    Args:
        env: Environment instance
        processor: Model processor for input preparation
        model: OpenVLA model for action prediction
        instruction: Task instruction string
        use_demos: Whether to use demonstration actions instead of model predictions
        demo: Demonstration object (required if use_demos=True)
        device: Device for model inference
        max_steps: Maximum number of steps to run
        
    Returns:
        List of RGB frames from the episode
    """
    obs = env.reset(seed=demo.seed if (use_demos and demo) else None)
    images = []
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"

    episode_length = min(max_steps, len(demo.timesteps) if demo else max_steps)
    
    for step_idx in range(episode_length):
        # Collect observation image
        if step_idx == 0:
            images.append(copy.deepcopy(obs[0]["rgb_head"]))
        else:
            images.append(obs["rgb_head"])

        # Get action from demo or model
        if use_demos and demo:
            action = demo.timesteps[step_idx].executed_action
        else:
            image = Image.fromarray(np.moveaxis(images[-1], 0, -1))
            inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
            action = model.predict_action(**inputs, do_sample=False)

        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode ended at step {step_idx + 1} (terminated: {terminated}, truncated: {truncated})")
            break

    return images


def visualize_frames(frames: List[np.ndarray], fps: int = DEFAULT_FPS) -> None:
    """
    Preview frames as a video using moviepy.
    
    Args:
        frames: List of RGB frames (C, H, W format)
        fps: Frames per second for playback
    """
    frames = np.moveaxis(np.array(frames), 1, -1)  # (T, C, H, W) → (T, H, W, C)
    duration = len(frames) / fps
    clip = VideoClip(make_frame=lambda t: frames[min(int(t * fps), len(frames) - 1)], duration=duration)
    clip.preview()


def save_frames_to_video(
    frames: List[np.ndarray], 
    output_dir: str, 
    filename: str = "rollout.mp4", 
    fps: int = DEFAULT_FPS
) -> None:
    """
    Save rollout frames to a video file.
    
    Args:
        frames: List of RGB frames (C, H, W format)
        output_dir: Output directory path
        filename: Output video filename
        fps: Frames per second for the output video
    """
    output_path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    
    frames = np.moveaxis(np.array(frames), 1, -1)  # (T, C, H, W) → (T, H, W, C)
    duration = len(frames) / fps
    clip = VideoClip(make_frame=lambda t: frames[min(int(t * fps), len(frames) - 1)], duration=duration)
    clip.write_videofile(output_path, fps=fps, codec="libx264")
    print(f"Saved rollout video to: {output_path}")



def main() -> None:
    """Main evaluation function."""
    # Parse arguments and create configuration
    args = parse_arguments()
    config = EvaluationConfig.from_args(args)
    
    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        return
    
    # Set output directory
    if config.output_dir is None:
        config.output_dir = config.ckpt_path
    
    print("=== OpenVLA Evaluation Configuration ===")
    for field_name in config.__dataclass_fields__:
        value = getattr(config, field_name)
        print(f"{field_name}: {value}")
    print("=" * 40)

    try:
        # Register OpenVLA components
        print("Registering OpenVLA components...")
        register_openvla()

        # Load model
        print(f"Loading VLA model from: {config.ckpt_path}")
        processor, model = load_vla_model(config.ckpt_path, config.device)

        # Setup environment
        print("Setting up environment...")
        env = setup_liftpot_env(downsample_rate=config.downsample_rate)

        # Load demonstrations if needed
        demos = []
        if config.use_demos:
            print("Loading demonstration data...")
            demos = get_successful_demos(
                env, 
                config.dataset_path, 
                config.downsample_rate, 
                num_demos=config.num_episodes
            )
            
            if len(demos) < config.num_episodes:
                print(f"Warning: Only found {len(demos)} successful demos, "
                      f"reducing episodes to {len(demos)}")
                config.num_episodes = len(demos)

        # Run evaluation episodes
        print(f"\nRunning {config.num_episodes} episodes...")
        for ep in range(config.num_episodes):
            mode = "demo" if config.use_demos else "inference"
            print(f"\nEpisode {ep + 1}/{config.num_episodes} ({mode} mode)")
            
            demo = DemoConverter.joint_to_ee(demos[ep]) if config.use_demos else None
            
            # Run episode
            frames = run_inference_loop(
                env=env,
                processor=processor,
                model=model,
                instruction=config.instruction,
                use_demos=config.use_demos,
                demo=demo,
                device=config.device,
                max_steps=config.max_steps
            )
            
            # Save video
            filename = f"rollout_{mode}_ep{ep + 1}.mp4"
            save_frames_to_video(
                frames, 
                config.output_dir, 
                filename=filename, 
                fps=config.fps
            )
            print(f"Episode {ep + 1} completed with {len(frames)} frames")

        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise
    finally:
        # Cleanup
        if 'env' in locals():
            env.close()
            print("Environment closed.")



if __name__ == "__main__":
    main()