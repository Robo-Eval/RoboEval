from abc import ABC
from roboeval.roboeval_env import RoboEvalEnv
from roboeval.utils.metric_rollout import MetricRolloutEval


class EmptyEnv(RoboEvalEnv, ABC, MetricRolloutEval):
    """Base Environment for RoboEval environments.
    
    A minimal implementation that can be extended for specific tasks.
    """
    
    _final_metrics = {}

    def _initialize_env(self):
        """Initialize environment components.
        
        Override this method in subclasses to create specific environments.
        """
        pass
        
    def _get_task_info(self):
        """Expose metrics for the task.
        
        Returns:
            dict: Metrics collected during the episode.
        """
        return getattr(self, "_final_metrics", {})
