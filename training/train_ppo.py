"""
PPO Training Script for Stock Trading Agent.

This script trains a PPO agent to trade stocks using a custom environment
and neural network architecture.

Usage:
    python train_ppo.py [--timesteps N] [--seed S] [--device D]
"""

import argparse
from datetime import datetime
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from shared.config import get_config
from shared.logging_config import setup_logging, get_logger
from training.environment import StockTradingEnv
from training.policies import (
    MultiScaleActorCriticPolicy,
    IdentityFeaturesExtractor,
)
from training.callbacks import (
    InfoLoggerCallback,
    GradientMonitoringCallback,
    TradingMetricsCallback,
    StockIdUpdateCallback,
    EpisodeMetricsCallback,
)
from training.curriculum import CurriculumCallback
from training.lr_schedules import warmup_exponential_schedule

logger = get_logger(__name__)


def create_env(
    config,
    mode: str = "train",
    action_space_type: Optional[str] = None,
) -> StockTradingEnv:
    """
    Create and configure a trading environment.

    Args:
        config: Configuration object
        mode: 'train' or 'test'
        action_space_type: Deprecated - always uses Discrete(3) now

    Returns:
        Configured StockTradingEnv instance
    """
    return StockTradingEnv(
        mode=mode,
        test_train_split=config.TEST_TRAIN_SPLIT,
        initial_balance=config.INITIAL_BALANCE,
        window_size=config.WINDOW_SIZE,
        transaction_cost_pct=config.TRANSACTION_COST,
        stamp_duty_pct=0.005,
        trade_penalty=config.TRADE_PENALTY,
        risk_factor=0.5,
        sharpe_weight=config.SHARPE_WEIGHT,
        loss_weight=config.LOSS_WEIGHT,
        gain_weight=config.GAIN_WEIGHT,
        hold_reward=config.HOLD_REWARD,
        aritificial_decay=config.ARTIFICIAL_DECAY,
        seed=config.SEED,
        ticker_limit=config.TICKER_LIMIT,
        risk_aversion=config.RISK_AVERSION,
        action_space_type="discrete",  # Always discrete now (Discrete(3))
    )


def train(
    total_timesteps: Optional[int] = None,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    n_envs: int = 1,
    action_space_type: Optional[str] = None,
    trend_window: int = 5,
):
    """
    Train a PPO agent for stock trading.

    Args:
        total_timesteps: Total training timesteps
        seed: Random seed
        device: Device to use (cuda/cpu)
        n_envs: Number of parallel environments
    """
    config = get_config()

    # Override with arguments if provided
    total_timesteps = total_timesteps or config.TOTAL_TIMESTEPS
    seed = seed if seed is not None else config.SEED
    device = device or config.DEVICE
    # Note: action_space_type is deprecated - always uses Discrete(3) now
    if action_space_type:
        logger.warning(
            f"Action space type '{action_space_type}' is deprecated. "
            "Using Discrete(3) action space (Hold/Buy/Sell)."
        )

    # Check device availability
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    logger.info("=" * 60)
    logger.info("STARTING PPO TRAINING")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Total timesteps: {total_timesteps:,}")
    logger.info(f"Seed: {seed}")
    logger.info(f"N envs: {n_envs}")
    logger.info(f"Action space: Discrete(3) - Hold/Buy/Sell")
    logger.info(f"Trading metrics trend window (batches): {trend_window}")

    # Create run identifier
    run_dt = datetime.now().strftime("%Y%m%d_%H%M")

    # Create environment
    if n_envs > 1:
        logger.info(f"Creating {n_envs} parallel environments")
        env = make_vec_env(
            lambda: create_env(config, action_space_type=action_space_type),
            n_envs=n_envs,
            vec_env_cls=SubprocVecEnv
        )
        # VecNormalize: norm_obs=True (critical for neural network), norm_reward=False (reward already scaled)
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    else:
        env = create_env(config, action_space_type=action_space_type)
        check_env(env, warn=True)
        # Wrap single environment in VecNormalize
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    def _unwrap_env(e):
        # VecNormalize wraps a VecEnv in `venv`
        while hasattr(e, "venv"):
            e = e.venv
        # VecEnv stores underlying envs in `envs`
        if hasattr(e, "envs") and len(e.envs) > 0:
            return e.envs[0]
        return e

    # Unwrap environment to get base env for metadata
    try:
        base_env = _unwrap_env(env)
    except Exception:
        # Fallback if unwrapping fails
        base_env = env
    
    # Get num_stocks from environment
    num_stocks = getattr(base_env, 'num_stocks', 1)
    embedding_dim = 16  # Embedding dimension for stock IDs
    logger.info(f"Number of stocks: {num_stocks}, Embedding dimension: {embedding_dim}")

    # Get observation space info
    if isinstance(env.observation_space, gym.spaces.Box):
        # New observation space is flattened: (obs_window_size * 9,)
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 1:
            # Flattened observation space
            input_dim = obs_shape[0]
            window_size = None  # Not a 2D window anymore
        else:
            # 2D observation space (backward compatibility)
            window_size = obs_shape[0]
            input_dim = obs_shape[-1]
    else:
        input_dim = gym.spaces.utils.flatdim(env.observation_space)
        window_size = None

    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = int(env.action_space.n)
        # New action space: Discrete(3) - Hold, Buy, Sell
        if action_dim == 3:
            action_names = ["hold", "buy", "sell"]
        else:
            # Fallback for backward compatibility
            action_names = ["hold", "buy", "sell"][:action_dim]
    else:
        raise ValueError(
            f"Unsupported action space: {type(env.action_space)}. "
            "Expected Discrete(3)."
        )

    logger.info(
        "Input dim: %s, Window size: %s, Actions: %s",
        input_dim,
        window_size,
        action_dim,
    )

    # Action space is always Discrete(3) now
    is_continuous = False
    
    # Entropy coefficient from config (high exploration to prevent freezing)
    ent_coef = config.ENTROPY_COEF

    # Network hyperparameters
    hidden_dim = 128
    sequence_lengths = [16, 64, 256]
    num_lstm_layers = 2

    # Learning rate schedule - warmup (start low, go high) then exponential decay
    lr_schedule = warmup_exponential_schedule(
        initial_value=config.LR_WARMUP_START,  # Start low
        peak_value=config.LR_PEAK,  # Go high
        final_value=config.LEARNING_RATE_END,  # Decay to final
        warmup_fraction=config.LR_WARMUP_FRACTION,
        decay_fraction=config.LR_DECAY_FRACTION,
    )

    # Create PPO model with updated hyperparameters
    model = PPO(
        policy=MultiScaleActorCriticPolicy,
        env=env,
        verbose=1,
        device=device,
        learning_rate=lr_schedule,  # Warmup then exponential decay schedule
        n_steps=2048,  # Stable gradients
        batch_size=128,  # Stable gradients
        n_epochs=config.EPOCHS,
        clip_range=0.2,
        ent_coef=ent_coef,  # From config
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=str(config.TB_LOG_DIR),
        gamma=0.99,
        gae_lambda=0.95,
        seed=seed,
        policy_kwargs=dict(
            features_extractor_class=IdentityFeaturesExtractor,
            input_dim=9,  # Features per timestep (not total flattened size)
            obs_window_size=config.OBS_WINDOW_SIZE,  # For policy reshaping
            n_features=9,  # For policy reshaping
            actor_hidden_dim=hidden_dim,
            critic_hidden_dim=hidden_dim,
            action_dim=action_dim,
            sequence_lengths=sequence_lengths,
            num_lstm_layers=num_lstm_layers,
            num_stocks=num_stocks,
            embedding_dim=embedding_dim,
        ),
    )

    # Calculate total timesteps
    try:
        epoch_timesteps = env.epoch_timesteps()
    except AttributeError:
        epoch_timesteps = total_timesteps

    actual_timesteps = min(
        int(epoch_timesteps * config.EPOCHS),
        total_timesteps,
    )
    logger.info(f"Training for {actual_timesteps:,} timesteps")

    # Create callbacks
    stock_id_updater = StockIdUpdateCallback(verbose=0)
    info_logger = InfoLoggerCallback(
        verbose=1,
        save_path=str(config.DATA_DIR / f"training_info_{run_dt}.csv"),
    )
    gradient_monitor = GradientMonitoringCallback(verbose=0)
    trading_metrics = TradingMetricsCallback(
        verbose=1,
        log_interval=1000,
        action_names=action_names,
        trend_window=trend_window,
    )
    curriculum = CurriculumCallback(
        total_timesteps=actual_timesteps,
        start_hold_reward=0.05,
        end_hold_reward=0.0,
    )
    # Custom metrics callback for episode-level metrics
    episode_metrics = EpisodeMetricsCallback(verbose=1)
    # Curriculum fee callback to set total training steps for dynamic fee calculation
    from training.callbacks import CurriculumFeeCallback
    curriculum_fee = CurriculumFeeCallback(
        total_timesteps=actual_timesteps,
        verbose=1
    )
    callbacks = [curriculum_fee, stock_id_updater, info_logger, gradient_monitor, trading_metrics, curriculum, episode_metrics]

    # Train
    try:
        model.learn(
            total_timesteps=actual_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")

    # Save model
    models_dir = config.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"ppo_stock_trading_{run_dt}"
    model.save(str(model_path))
    logger.info(f"Model saved to {model_path}")

    if n_envs > 1:
        vec_norm_path = models_dir / f"vec_normalize_{run_dt}.pkl"
        env.save(str(vec_norm_path))
        logger.info(f"VecNormalize saved to {vec_norm_path}")

    # Return training info for analysis
    return model, info_logger.info_history


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train PPO stock trading agent"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        help="Total training timesteps",
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Device",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of environments",
    )
    parser.add_argument(
        "--action-space-type",
        type=str,
        choices=["discrete", "continuous"],
        help="Action space type for the environment/policy",
    )
    parser.add_argument(
        "--trend-window",
        type=int,
        default=5,
        help="Number of recent metric batches to trend over for TensorBoard trading/* scalars",
    )
    args = parser.parse_args()

    # Setup logging
    config = get_config()
    setup_logging(level=config.LOG_LEVEL, log_dir=config.LOGS_DIR)

    # Run training
    train(
        total_timesteps=args.timesteps,
        seed=args.seed,
        device=args.device,
        n_envs=args.n_envs,
        action_space_type=args.action_space_type,
        trend_window=args.trend_window,
    )


if __name__ == "__main__":
    main()
