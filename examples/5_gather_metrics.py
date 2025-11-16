#!/usr/bin/env python3
"""
Demo Loading and Metrics Computation Script

This script provides comprehensive functionality for:
1. Loading demonstrations from various sources (files, DemoStore, directories)
2. Computing metrics from demonstrations (trajectory efficiency, coordination quality, etc.)
3. Replaying demonstrations in environments to collect rollout metrics
4. Analyzing and reporting metrics with statistical summaries

Usage Examples:
    # Load demos from a directory and compute metrics
    python 5_gather_metrics.py --demos_dir /path/to/demos --env_name LiftPot

    # Load demos from DemoStore and analyze metrics
    python 5_gather_metrics.py --use_demo_store --env_name LiftPot --amount 50

    # Compute metrics from specific demo files
    python 5_gather_metrics.py --demo_files demo1.safetensors demo2.safetensors

    # Generate detailed metric reports
    python 5_gather_metrics.py --demos_dir /path/to/demos --output_report metrics_report.json

Dependencies:
    - roboeval package with all demonstration and environment components
    - numpy, pandas for data analysis
    - tqdm for progress tracking
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings
from collections import defaultdict
import time
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# RoboEval imports
from roboeval.demonstrations.demo import Demo, DemoStep
from roboeval.demonstrations.demo_store import DemoStore, DemoNotFoundError
from roboeval.demonstrations.demo_player import DemoPlayer
from roboeval.demonstrations.demo_converter import DemoConverter
from roboeval.demonstrations.utils import Metadata, ObservationMode
from roboeval.roboeval_env import RoboEvalEnv, CONTROL_FREQUENCY_MAX
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.const import HandSide

# Import environment classes for instantiation
from tools.shared.utils import ENVIRONMENTS


class DemoMetricsComputer:
    """Comprehensive demonstration loading and metrics computation utility."""

    def __init__(self, verbose: bool = True):
        """Initialize the metrics computer.
        
        Args:
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self.demo_store = DemoStore()
        self.logger = self._setup_logger()
        self.env = None  # Will be created when needed
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO if self.verbose else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def create_environment(self, env_name: str, 
                          frequency: int = CONTROL_FREQUENCY_MAX,
                          robot_name: str = "BimanualPanda",
                          action_mode_name: str = "JointPositionActionMode") -> Optional[RoboEvalEnv]:
        """Create and cache an environment instance for use across all methods.
        
        Args:
            env_name: Name of the environment
            frequency: Control frequency for the environment
            robot_name: Name of the robot
            action_mode_name: Name of the action mode
            
        Returns:
            Environment instance or None if creation fails
        """
        if self.env is not None:
            self.logger.info(f"Using cached environment: {self.env.__class__.__name__}")
            return self.env
            
        try:
            # Validate environment name
            if env_name not in ENVIRONMENTS:
                self.logger.error(f"Unknown environment: {env_name}")
                self.logger.error(f"Available environments: {list(ENVIRONMENTS.keys())}")
                return None
            
            # Create the environment class
            env_class = ENVIRONMENTS[env_name]
            
            # Set up action mode
            if action_mode_name == "JointPositionActionMode":
                from roboeval.action_modes import JointPositionActionMode
                action_mode = JointPositionActionMode(floating_base=True, absolute=True, floating_dofs=[])
            else:
                # Default fallback
                from roboeval.action_modes import JointPositionActionMode
                action_mode = JointPositionActionMode(floating_base=True, absolute=True)
            
            # Set up robot class
            if robot_name == "BimanualPanda":
                from roboeval.robots.configs.panda import BimanualPanda
                robot_cls = BimanualPanda
            else:
                # Default fallback
                from roboeval.robots.configs.panda import BimanualPanda
                robot_cls = BimanualPanda
            
            # Create environment instance
            self.env = env_class(
                action_mode=action_mode,
                control_frequency=frequency,
                robot_cls=robot_cls,
            )
            
            self.logger.info(f"Created environment: {self.env.__class__.__name__} (supports metrics: {isinstance(self.env, MetricRolloutEval)})")
            return self.env
            
        except Exception as e:
            self.logger.error(f"Error creating environment {env_name}: {e}")
            return None

    def cleanup_environment(self):
        """Clean up the cached environment."""
        if self.env and hasattr(self.env, 'close'):
            self.env.close()
            self.logger.debug("Environment closed")
        self.env = None

    def load_demos_from_files(self, demo_files: List[Union[str, Path]], 
                             frequency: int = CONTROL_FREQUENCY_MAX) -> List[Demo]:
        """Load demonstrations from individual files.
        
        Args:
            demo_files: List of paths to demonstration files
            frequency: Control frequency for demo processing
            
        Returns:
            List of loaded Demo objects
        """
        demos = []
        self.logger.info(f"Loading {len(demo_files)} demo files...")
        
        for demo_file in tqdm(demo_files, desc="Loading demos", disable=not self.verbose):
            demo_path = Path(demo_file)
            if not demo_path.exists():
                self.logger.warning(f"Demo file not found: {demo_path}")
                continue
                
            try:
                demo = Demo.from_safetensors(demo_path)
                if demo is not None:
                    demos.append(demo)
                    self.logger.debug(f"Loaded demo: {demo.uuid}")
                else:
                    self.logger.warning(f"Failed to load demo: {demo_path}")
            except Exception as e:
                self.logger.error(f"Error loading demo {demo_path}: {e}")
                
        # Apply frequency conversion if needed
        if frequency != CONTROL_FREQUENCY_MAX and demos:
            self.logger.info(f"Converting demos to frequency {frequency}Hz...")
            converted_demos = []
            for demo in demos:
                try:
                    if hasattr(demo.metadata, 'get_robot'):
                        robot = demo.metadata.get_robot()
                        converted_demo = DemoConverter.decimate(
                            demo, frequency, CONTROL_FREQUENCY_MAX, robot=robot
                        )
                        converted_demos.append(converted_demo)
                    else:
                        converted_demos.append(demo)
                except Exception as e:
                    self.logger.warning(f"Failed to convert demo {demo.uuid} frequency: {e}")
                    converted_demos.append(demo)
            demos = converted_demos
                
        self.logger.info(f"Successfully loaded {len(demos)} demonstrations")
        return demos

    def load_demos_from_directory(self, demos_dir: Union[str, Path], 
                                   amount: int = -1, frequency: int = CONTROL_FREQUENCY_MAX,
                                   robot_name: str = "BimanualPanda",
                                   action_mode_name: str = "JointPositionActionMode",
                                   pattern: str = "*.safetensors") -> List[Demo]:
        """Load demonstrations from a directory using DemoStore.get_demos_from_folder.
        
        Args:
            demos_dir: Directory containing demonstration files
            amount: Maximum number of demos to load (-1 for all)
            frequency: Control frequency for demo processing
            robot_name: Name of the robot for metadata creation
            action_mode_name: Name of the action mode for metadata creation
            pattern: File pattern to match (default: *.safetensors)
            
        Returns:
            List of loaded Demo objects
        """
        demos_dir = Path(demos_dir)
        if not demos_dir.exists():
            self.logger.error(f"Demos directory not found: {demos_dir}")
            self.logger.error("To download demonstration data, please run:")
            self.logger.error("  python examples/1_data_replay.py")
            self.logger.error("Or check the README for data download instructions.")
            return []
            
        demo_files = list(demos_dir.glob(pattern))
        if not demo_files:
            self.logger.error(f"No demonstration files found in {demos_dir} matching pattern '{pattern}'")
            self.logger.error("To download demonstration data, please run:")
            self.logger.error("  python examples/1_data_replay.py")
            return []
            
        try:
            # Create a basic metadata object for get_demos_from_folder
            # We'll use the first demo file to extract proper metadata
            first_demo = Demo.from_safetensors(demo_files[0])
            if first_demo is None:
                self.logger.error("Failed to load first demo to extract metadata")
                return []
                
            metadata = first_demo.metadata
            
            # Use DemoStore's existing get_demos_from_folder method
            self.logger.info(f"Loading demos from directory using DemoStore: {demos_dir}")
            demos = self.demo_store.get_demos_from_folder(
                demos_dir=demos_dir,
                metadata=metadata,
                amount=amount,
                frequency=frequency
            )
            
            self.logger.info(f"Successfully loaded {len(demos)} demonstrations from directory")
            return demos
            
        except Exception as e:
            self.logger.error(f"Error loading demos from directory: {e}")
            self.logger.error("Falling back to simple file loading...")
            return self.load_demos_from_files(demo_files, frequency=frequency)

    def load_demos_from_store(self, env_name: str, 
                             amount: int = -1, frequency: int = CONTROL_FREQUENCY_MAX,
                             robot_name: str = "BimanualPanda",
                             action_mode_name: str = "JointPositionActionMode",
                             observation_mode = None) -> List[Demo]:
        """Load demonstrations from DemoStore.
        
        Args:
            env_name: Name of the environment
            amount: Number of demos to load (-1 for all available)
            frequency: Control frequency for the demos
            robot_name: Name of the robot
            action_mode_name: Name of the action mode
            observation_mode: Type of observations (Lightweight or Full)
            
        Returns:
            List of loaded Demo objects
        """
        try:
            # Create environment using centralized method
            env = self.create_environment(env_name, frequency, robot_name, action_mode_name)
            if env is None:
                return []
            
            # Get metadata from environment (following 1_data_replay.py pattern)
            metadata = Metadata.from_env(env)
            
            self.logger.info(f"Loading demos from DemoStore for {env_name}...")
            demos = self.demo_store.get_demos(metadata, amount=amount, frequency=frequency)
            self.logger.info(f"Successfully loaded {len(demos)} demonstrations from DemoStore")
            
            return demos
            
        except DemoNotFoundError as e:
            self.logger.error(f"No demos found in DemoStore: {e}")
            self.logger.error("To download demonstration data, please run:")
            self.logger.error("  python examples/1_data_replay.py")
            self.logger.error("Or check the README for data download instructions.")
            return []
        except Exception as e:
            self.logger.error(f"Error loading demos from DemoStore: {e}")
            return []

    def compute_basic_demo_metrics(self, demo: Demo) -> Dict[str, Any]:
        """Compute basic metrics from a demonstration without environment replay.
        
        Args:
            demo: Demonstration to analyze
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {
            'demo_id': demo.uuid,
            'duration_steps': demo.duration,
            'total_timesteps': len(demo.timesteps),
            'environment': demo.metadata.environment_data.env_name,
            'robot': demo.metadata.environment_data.robot_name,
            'action_mode': demo.metadata.environment_data.action_mode_name,
            'observation_mode': demo.metadata.observation_mode.value,
        }
        
        if demo.timesteps:
            # Extract action sequences
            actions = np.array([step.executed_action for step in demo.timesteps])
            
            # Basic action statistics
            metrics.update({
                'action_dim': actions.shape[1] if len(actions.shape) > 1 else 1,
                'action_mean': np.mean(actions, axis=0).tolist() if len(actions.shape) > 1 else float(np.mean(actions)),
                'action_std': np.std(actions, axis=0).tolist() if len(actions.shape) > 1 else float(np.std(actions)),
                'action_range': (np.min(actions, axis=0).tolist(), np.max(actions, axis=0).tolist()) if len(actions.shape) > 1 else (float(np.min(actions)), float(np.max(actions))),
            })
            
            # Action smoothness (as proxy for trajectory quality)
            if len(actions) > 1:
                action_diffs = np.diff(actions, axis=0)
                action_smoothness = np.mean(np.linalg.norm(action_diffs, axis=1)) if len(actions.shape) > 1 else np.mean(np.abs(action_diffs))
                metrics['action_smoothness'] = float(action_smoothness)
                
            # Reward statistics
            rewards = [step.reward for step in demo.timesteps]
            if rewards:
                metrics.update({
                    'total_reward': sum(rewards),
                    'mean_reward': np.mean(rewards),
                    'final_reward': rewards[-1],
                    'max_reward': max(rewards),
                })
                
            # Termination analysis
            terminations = [step.termination for step in demo.timesteps]
            if any(terminations):
                metrics['terminated_early'] = True
                metrics['termination_step'] = next(i for i, term in enumerate(terminations) if term)
            else:
                metrics['terminated_early'] = False
                metrics['termination_step'] = len(terminations)
                
        return metrics

    def replay_demo_with_metrics(self, demo: Demo, env: Optional[RoboEvalEnv] = None,
                                demo_frequency: int = CONTROL_FREQUENCY_MAX) -> Dict[str, Any]:
        """Replay demonstration in environment and collect comprehensive metrics.
        
        Args:
            demo: Demonstration to replay
            env: Environment to replay in (if None, creates from demo metadata)
            demo_frequency: Frequency of the demonstration
            
        Returns:
            Dictionary of collected metrics including rollout metrics
        """
        # Create environment if not provided
        if env is None:
            try:
                env = demo.metadata.get_env(demo.metadata.environment_data.control_frequency or CONTROL_FREQUENCY_MAX)
            except Exception as e:
                self.logger.error(f"Failed to create environment from demo metadata: {e}")
                return {'error': str(e)}
        
        # Get basic metrics first
        metrics = self.compute_basic_demo_metrics(demo)
        
        try:
            # Check if environment supports metric collection
            supports_metrics = isinstance(env, MetricRolloutEval)
            
            if supports_metrics:
                # Initialize metrics tracking
                env._metric_init(
                    track_cartesian_jerk=True,
                    track_joint_jerk=True,
                    track_cartesian_path_length=True,
                    track_joint_path_length=True,
                    track_orientation_path_length=True,
                    track_vel_sync=True,
                    track_vertical_sync=True,
                    track_collisions=True,
                    robot=env.robot
                )
            
            # Replay the demonstration
            timesteps = DemoPlayer._get_timesteps_for_replay(demo, env, demo_frequency)
            env.reset(seed=demo.seed)

            
            replay_metrics = {
                'replay_success': False,
                'replay_duration': 0,
                'steps_completed': 0,
                'final_success': False,
            }
            
            start_time = time.time()
            
            for i, step in enumerate(timesteps):
                        
                action = step.executed_action
                observation, reward, termination, truncation, info = env.step(action)
                    
                # Track step completion
                replay_metrics['steps_completed'] = i + 1
                    
                if termination or truncation:
                    break
            
            replay_metrics['replay_duration'] = time.time() - start_time
            replay_metrics['replay_success'] = True

            # Check final success if environment supports it
            if hasattr(env, '_success'):
                replay_metrics['final_success'] = env._success()
            elif hasattr(env, 'reward') and env.reward > 0:
                replay_metrics['final_success'] = True
                
            # Collect task-specific metrics if available
            if hasattr(env, '_get_task_info'):
                try:
                    task_metrics = env._get_task_info()
                    if task_metrics:
                        metrics.update({'task_metrics': task_metrics})
                except Exception as e:
                    self.logger.warning(f"Error collecting task metrics: {e}")
            
            metrics.update(replay_metrics)
            
        except Exception as e:
            self.logger.error(f"Error during demo replay: {e}")
            metrics.update({
                'replay_success': False,
                'replay_error': str(e)
            })
        finally:
            # Clean up environment
            if hasattr(env, 'close'):
                env.close()
                
        return metrics

    def compute_metrics_for_demos(self, demos: List[Demo], 
                                 replay_in_env: bool = True,
                                 env_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Compute metrics for a list of demonstrations.
        
        Args:
            demos: List of demonstrations to analyze
            replay_in_env: Whether to replay demos in environment for detailed metrics
            env_name: Environment name to use for replay (if different from demo metadata)
            
        Returns:
            List of metric dictionaries, one per demonstration
        """
        if not demos:
            self.logger.warning("No demonstrations provided for metrics computation")
            return []
            
        all_metrics = []
        
        # Use cached environment if available, or create from demo metadata if needed
        env = self.env
        if replay_in_env and env is None:
            try:
                if env_name and env_name in ENVIRONMENTS:
                    # Create environment using centralized method
                    env = self.create_environment(env_name)
                else:
                    # Use environment from first demo
                    env = demos[0].metadata.get_env(
                        demos[0].metadata.environment_data.control_frequency or CONTROL_FREQUENCY_MAX
                    )
                    # Cache it for future use
                    self.env = env
                    
                if env:
                    self.logger.info(f"Created environment for replay: {env.__class__.__name__}")
            except Exception as e:
                self.logger.warning(f"Failed to create environment, skipping replay: {e}")
                replay_in_env = False
        
        self.logger.info(f"Computing metrics for {len(demos)} demonstrations...")
        
        for demo in tqdm(demos, desc="Processing demos", disable=not self.verbose):
            try:
                if replay_in_env and env is not None:
                    metrics = self.replay_demo_with_metrics(demo, env, demo_frequency=env.control_frequency)
                else:
                    metrics = self.compute_basic_demo_metrics(demo)
                    
                all_metrics.append(metrics)
                
            except Exception as e:
                self.logger.error(f"Error processing demo {demo.uuid}: {e}")
                # Add basic error info
                all_metrics.append({
                    'demo_id': demo.uuid,
                    'error': str(e),
                    'processing_failed': True
                })
        
        self.logger.info(f"Successfully computed metrics for {len(all_metrics)} demonstrations")
        return all_metrics

    def analyze_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze collected metrics and compute summary statistics.
        
        Args:
            metrics_list: List of metric dictionaries from demonstrations
            
        Returns:
            Dictionary containing analysis results and summary statistics
        """
        if not metrics_list:
            return {'error': 'No metrics provided for analysis'}
        
        # Filter out failed processing
        valid_metrics = [m for m in metrics_list if not m.get('processing_failed', False)]
        failed_count = len(metrics_list) - len(valid_metrics)


        
        analysis = {
            'summary': {
                'total_demos': len(metrics_list),
                'successfully_processed': len(valid_metrics),
                'processing_failures': failed_count,
                'processing_success_rate': len(valid_metrics) / len(metrics_list) if metrics_list else 0
            }
        }
        
        if not valid_metrics:
            analysis['summary']['note'] = 'No valid metrics to analyze'
            return analysis
        
        # Extract task_metrics subdictionaries if available
        for i, metric_entry in enumerate(valid_metrics):
            if 'task_metrics' in metric_entry and isinstance(metric_entry['task_metrics'], dict):
                valid_metrics[i].update(metric_entry['task_metrics'])

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(valid_metrics)

        # Basic statistics for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number, np.float64, np.float32, np.int64, np.int32]).columns
        if len(numeric_columns) > 0:
            analysis['numeric_statistics'] = {}
            for col in numeric_columns:
                analysis['numeric_statistics'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'median': float(df[col].median()),
                    'count': int(df[col].count())
                }
        
        # Success rate analysis
        if 'final_success' in df.columns:
            success_rate = df['final_success'].mean()
            analysis['task_performance'] = {
                'success_rate': float(success_rate),
                'successful_demos': int(df['final_success'].sum()),
                'failed_demos': int((~df['final_success']).sum())
            }
        
        # Environment and configuration distribution
        categorical_fields = ['environment', 'robot', 'action_mode', 'observation_mode']
        for field in categorical_fields:
            if field in df.columns:
                analysis[f'{field}_distribution'] = df[field].value_counts().to_dict()
        
        # Trajectory quality metrics
        trajectory_metrics = ['completion_time', 'slip_count', 'env_collision_count',
       'self_collision_count', 'subtask_progress', 'object_pose_error',
       'success', 'total_cartesian_path_length', 'avg_cartesian_path_length',
       'total_joint_path_length', 'avg_joint_path_length',
       'total_orientation_path_length', 'avg_orientation_path_length',
       'bimanual_arm_velocity_difference',
       'bimanual_gripper_vertical_difference', 'overall_avg_cartesian_jerk',
       'overall_avg_joint_jerk']
        available_traj_metrics = [m for m in trajectory_metrics if m in df.columns]
        if available_traj_metrics:
            analysis['trajectory_quality'] = {}
            for metric in available_traj_metrics:
                analysis['trajectory_quality'][metric] = {
                    'mean': float(df[metric].mean()),
                    'std': float(df[metric].std()),
                    'percentiles': {
                        '25th': float(df[metric].quantile(0.25)),
                        '50th': float(df[metric].quantile(0.50)),
                        '75th': float(df[metric].quantile(0.75)),
                        '90th': float(df[metric].quantile(0.90))
                    }
                }
        
        return analysis

    def save_metrics_report(self, metrics_list: List[Dict[str, Any]], 
                           analysis: Dict[str, Any], output_file: Union[str, Path]):
        """Save comprehensive metrics report to file.
        
        Args:
            metrics_list: Raw metrics data
            analysis: Analysis results
            output_file: Path to save the report
        """
        output_path = Path(output_file)
        
        report = {
            'metadata': {
                'generation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_demonstrations': len(metrics_list),
                'metrics_version': '1.0'
            },
            'analysis': analysis,
            'raw_metrics': metrics_list
        }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        self.logger.info(f"Metrics report saved to: {output_path}")
        
        # Also save a CSV summary if there are valid metrics
        valid_metrics = [m for m in metrics_list if not m.get('processing_failed', False)]
        if valid_metrics:
            csv_path = output_path.with_suffix('.csv')
            df = pd.DataFrame(valid_metrics)
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Metrics CSV saved to: {csv_path}")


def main():
    """Main function demonstrating comprehensive demo loading and metrics computation."""
    parser = argparse.ArgumentParser(
        description="Load demonstrations and compute comprehensive metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Load demos from directory with frequency conversion
    python 5_gather_metrics.py --demos_dir /path/to/demos --frequency 20 --amount 5
    
    # Load demos from DemoStore with replay metrics
    python 5_gather_metrics.py --use_demo_store --env_name LiftPot --amount 20 --replay
    
    # Analyze specific demo files with custom robot/action mode
    python 5_gather_metrics.py --demo_files demo1.safetensors demo2.safetensors --robot_name BimanualPanda --action_mode JointPositionActionMode --replay
    
    # Generate comprehensive report with all options
    python 5_gather_metrics.py --demos_dir /path/to/demos --amount 50 --frequency 10 --output_report results.json --replay
        """
    )
    
    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--demos_dir', type=str,
                           help='Directory containing demonstration files')
    input_group.add_argument('--demo_files', nargs='+', type=str,
                           help='Specific demonstration files to analyze')
    input_group.add_argument('--use_demo_store', action='store_true',
                           help='Load demonstrations from DemoStore')
    
    # DemoStore options
    parser.add_argument('--env_name', type=str, default='LiftPot',
                       help='Environment name for DemoStore loading (default: LiftPot)')
    parser.add_argument('--robot_name', type=str, default='BimanualPanda',
                       help='Robot name for demo loading (default: BimanualPanda)')
    parser.add_argument('--action_mode', type=str, default='JointPositionActionMode',
                       help='Action mode for demo loading (default: JointPositionActionMode)')
    parser.add_argument('--amount', type=int, default=10,
                       help='Number of demos to load (default: 10, -1 for all)')
    
    # Processing options
    parser.add_argument('--replay', action='store_true',
                       help='Replay demonstrations in environment for detailed metrics')
    parser.add_argument('--frequency', type=int, default=CONTROL_FREQUENCY_MAX,
                       help=f'Control frequency for demo processing (default: {CONTROL_FREQUENCY_MAX})')
    
    # Output options
    parser.add_argument('--output_report', type=str,
                       help='Path to save comprehensive metrics report (JSON format)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Initialize metrics computer
    computer = DemoMetricsComputer(verbose=args.verbose)
    
    try:
        # Load demonstrations based on input source
        demos = []
        
        if args.demos_dir:
            demos = computer.load_demos_from_directory(
                demos_dir=args.demos_dir,
                amount=args.amount,
                frequency=args.frequency,
                robot_name=args.robot_name,
                action_mode_name=args.action_mode
            )
        elif args.demo_files:
            demos = computer.load_demos_from_files(
                demo_files=args.demo_files,
                frequency=args.frequency
            )
        elif args.use_demo_store:
            demos = computer.load_demos_from_store(
                env_name=args.env_name,
                amount=args.amount,
                frequency=args.frequency,
                robot_name=args.robot_name,
                action_mode_name=args.action_mode
            )
        
        if not demos:
            print("No demonstrations loaded. Exiting.")
            print("\nTo download demonstration data, please run:")
            print("  python examples/1_data_replay.py")
            print("Or check the README for data download instructions.")
            return 1
        
        print(f"\nLoaded {len(demos)} demonstrations")
        print(f"Sample demo info:")
        for i, demo in enumerate(demos[:3]):
            print(f"  Demo {i+1}: {demo.uuid} ({demo.duration} steps, {demo.metadata.environment_data.env_name})")
        if len(demos) > 3:
            print(f"  ... and {len(demos) - 3} more")
        
        # Compute metrics
        print(f"\nComputing metrics (replay mode: {args.replay})...")
        metrics_list = computer.compute_metrics_for_demos(
            demos, 
            replay_in_env=args.replay,
            env_name=args.env_name if args.use_demo_store else None
        )
        
        if not metrics_list:
            print("No metrics computed. Exiting.")
            print("\nThis could be due to:")
            print("1. No valid demonstration files found")
            print("2. Missing dependencies for demonstration replay")
            print("3. Incompatible demonstration format")
            print("\nTo download proper demonstration data, please run:")
            print("  python examples/1_data_replay.py")
            return 1
        
        # Analyze metrics
        print("\nAnalyzing metrics...")
        analysis = computer.analyze_metrics(metrics_list)
        
        # Print summary
        print("\n" + "="*60)
        print("METRICS SUMMARY")
        print("="*60)
        
        summary = analysis.get('summary', {})
        print(f"Total demonstrations: {summary.get('total_demos', 0)}")
        print(f"Successfully processed: {summary.get('successfully_processed', 0)}")
        print(f"Processing failures: {summary.get('processing_failures', 0)}")
        print(f"Success rate: {summary.get('processing_success_rate', 0):.2%}")
        
        # Task performance
        if 'task_performance' in analysis:
            perf = analysis['task_performance']
            print(f"\nTask Performance:")
            print(f"  Success rate: {perf['success_rate']:.2%}")
            print(f"  Successful demos: {perf['successful_demos']}")
            print(f"  Failed demos: {perf['failed_demos']}")
        
        # Trajectory quality
        if 'trajectory_quality' in analysis:
            print(f"\nTrajectory Quality Metrics:")
            for metric, stats in analysis['trajectory_quality'].items():
                print(f"  {metric}: {stats['mean']:.3f} ± {stats['std']:.3f}")
        
        # Rollout metrics
        if 'rollout_metrics' in analysis:
            print(f"\nRollout Metrics (top 5):")
            rollout_items = list(analysis['rollout_metrics'].items())[:5]
            for metric, stats in rollout_items:
                print(f"  {metric}: {stats['mean']:.3f} ± {stats['std']:.3f}")
            if len(analysis['rollout_metrics']) > 5:
                print(f"  ... and {len(analysis['rollout_metrics']) - 5} more metrics")
        
        # Environment distribution
        if 'environment_distribution' in analysis:
            print(f"\nEnvironment Distribution:")
            for env, count in analysis['environment_distribution'].items():
                print(f"  {env}: {count} demos")
        
        # Save report if requested
        if args.output_report:
            computer.save_metrics_report(metrics_list, analysis, args.output_report)
            print(f"\nDetailed report saved to: {args.output_report}")
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        
        return 0
        
    finally:
        # Always cleanup environment resources
        computer.cleanup_environment()


if __name__ == "__main__":
    exit(main())
