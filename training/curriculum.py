"""
Curriculum learning callback for gradually removing hold incentives.
"""

from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback

from shared.logging_config import get_logger

logger = get_logger(__name__)


class CurriculumCallback(BaseCallback):
    """
    Linearly decreases the environment's hold_reward during training.

    The callback updates every step using the model's progress_redmaining
    (1.0 -> 0.0) so that hold_reward transitions from start_hold_reward
    to end_hold_reward over the course of training.
    """

    def __init__(
        self,
        total_timesteps: Optional[int],
        start_hold_reward: float = 0.05,
        end_hold_reward: float = 0.0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.total_timesteps = max(int(total_timesteps or 1), 1)
        self.start_hold_reward = start_hold_reward
        self.end_hold_reward = end_hold_reward

    def _set_hold_reward(self, value: float) -> None:
        """Propagate the hold_reward to underlying environments."""
        envs = []
        try:
            if hasattr(self.training_env, "envs"):
                envs = self.training_env.envs
            elif hasattr(self.training_env, "venv") and hasattr(self.training_env.venv, "envs"):
                envs = self.training_env.venv.envs  # type: ignore[attr-defined]
            else:
                envs = [self.training_env]
        except Exception:
            envs = [self.training_env]

        for env in envs:
            base_env = getattr(env, "env", env)
            # Unwrap one more level if needed
            base_env = getattr(base_env, "env", base_env)
            if hasattr(base_env, "hold_reward"):
                base_env.hold_reward = value

    def _on_step(self) -> bool:
        """Update hold_reward based on remaining training progress."""
        progress_remaining = getattr(self.model, "_current_progress_remaining", None)
        if progress_remaining is None:
            # Fallback using timesteps if SB3 internal attribute is unavailable
            progress_remaining = 1.0 - min(self.num_timesteps / self.total_timesteps, 1.0)

        hold_reward = self.end_hold_reward + (self.start_hold_reward - self.end_hold_reward) * max(
            progress_remaining, 0.0
        )
        self._set_hold_reward(hold_reward)
        self.logger.record("curriculum/hold_reward", hold_reward)
        return True

