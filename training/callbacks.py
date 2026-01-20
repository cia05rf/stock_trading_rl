"""
Custom callbacks for training monitoring.

This module provides callbacks for Stable-Baselines3 training that enable
logging, checkpointing, and training monitoring.
"""

from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import pandas as pd
import gymnasium as gym
import torch
from stable_baselines3.common.callbacks import BaseCallback

from shared.logging_config import get_logger

logger = get_logger(__name__)


class CurriculumFeeCallback(BaseCallback):
    """
    Callback to set total training steps in the environment for curriculum learning.
    
    This enables the environment to calculate dynamic transaction fees based on
    training progress (starting at 0.0 and ramping to TARGET_TRANSACTION_FEE).
    """
    
    def __init__(self, total_timesteps: Optional[int] = None, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps_set = False
        self.total_timesteps = total_timesteps
    
    def _on_training_start(self) -> None:
        """Set total training steps when training starts."""
        # We'll set it in _on_step when we have access to locals
        pass
    
    def _on_step(self) -> bool:
        """Set total training steps on first step if not already set."""
        if not self.total_timesteps_set:
            # Try multiple ways to get total_timesteps
            total_timesteps = self.total_timesteps
            
            # Method 1: Use the value passed during initialization
            if total_timesteps is None:
                # Method 2: From locals (passed from learn() method)
                if hasattr(self, 'locals') and self.locals:
                    total_timesteps = self.locals.get('total_timesteps', None)
            
            if total_timesteps is not None:
                self._set_total_timesteps(total_timesteps)
        
        return True
    
    def _set_total_timesteps(self, total_timesteps: int) -> None:
        """Helper method to set total timesteps in environment."""
        # Get the base environment (unwrap VecEnv if needed)
        env = self.training_env
        if hasattr(env, 'envs') and len(env.envs) > 0:
            # VecEnv - set for all sub-environments
            for sub_env in env.envs:
                if hasattr(sub_env, 'set_total_training_steps'):
                    sub_env.set_total_training_steps(total_timesteps)
        elif hasattr(env, 'set_total_training_steps'):
            # Single environment
            env.set_total_training_steps(total_timesteps)
        
        self.total_timesteps_set = True
        logger.info(
            f"Set total training steps to {total_timesteps:,} "
            f"for curriculum learning"
        )


class StockIdUpdateCallback(BaseCallback):
    """
    Callback to update stock_id in the policy from environment info.
    
    This ensures the policy's stock_id is synchronized with the environment's
    current stock_id for proper embedding lookup.
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        """Update stock_id from environment info."""
        infos = self.locals.get('infos', [])
        
        # Update policy's stock_id from the first environment's info
        if infos and len(infos) > 0:
            info = infos[0] if isinstance(infos, list) else infos
            if isinstance(info, dict) and 'stock_id' in info:
                stock_id = info['stock_id']
                # Update policy if it has set_stock_id method
                if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'set_stock_id'):
                    self.model.policy.set_stock_id(stock_id)
        
        return True


class InfoLoggerCallback(BaseCallback):
    """
    Callback for logging training info from the environment.

    Collects information from the environment's info dict at each step
    and optionally saves to CSV at the end of training.

    Args:
        verbose: Verbosity level
        save_path: Path to save training info CSV (optional)
        log_interval: Steps between logging summaries
    """

    def __init__(
        self,
        verbose: int = 0,
        save_path: Optional[str] = None,
        log_interval: int = 10000,
    ):
        super().__init__(verbose)
        self.info_history: List[Dict[str, Any]] = []
        self.save_path = save_path
        self.log_interval = log_interval
        self._step_count = 0

    def _on_step(self) -> bool:
        """Called at each step."""
        infos = self.locals.get('infos', [])

        for info in infos:
            if info:
                self.info_history.append(info)

        self._step_count += 1

        # Log summary periodically
        if self.verbose > 0 and self._step_count % self.log_interval == 0:
            self._log_summary()

        return True

    def _log_summary(self) -> None:
        """Log a summary of recent training."""
        if not self.info_history:
            return

        recent = self.info_history[-self.log_interval:]

        rewards = [i.get('reward', 0) for i in recent]
        net_worths = [
            i.get('net_worth', 0) for i in recent if 'net_worth' in i
        ]

        logger.info(
            f"Step {self._step_count}: "
            f"Reward mean={np.mean(rewards):.4f}, "
            f"Net worth mean={np.mean(net_worths):.2f}"
        )

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if self.save_path and self.info_history:
            df = pd.DataFrame(self.info_history)
            save_path = Path(self.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            logger.info(f"Saved training info to {save_path}")


class GradientMonitoringCallback(BaseCallback):
    """
    Callback for monitoring gradient health during training.

    Checks for NaN/Inf gradients which can indicate training instability.

    Args:
        verbose: Verbosity level
        check_interval: Steps between gradient checks
    """

    def __init__(self, verbose: int = 0, check_interval: int = 100):
        super().__init__(verbose)
        self.check_interval = check_interval
        self._step_count = 0

    def _on_step(self) -> bool:
        """Check gradients periodically."""
        self._step_count += 1

        if self._step_count % self.check_interval != 0:
            return True

        bad_nan: List[str] = []
        bad_inf: List[str] = []
        bad_large: List[str] = []

        for name, param in self.model.policy.named_parameters():
            grad = param.grad
            if grad is None:
                continue

            if torch.isnan(grad).any():
                bad_nan.append(name)
                continue

            if torch.isinf(grad).any():
                bad_inf.append(name)
                continue

            grad_mean = grad.mean().item()
            # Avoid PyTorch warning when grad.numel() <= 1
            grad_std = grad.float().std(unbiased=False).item()
            if abs(grad_mean) > 100 or grad_std > 100:
                bad_large.append(
                    f"{name}(mean={grad_mean:.2f},std={grad_std:.2f})"
                )

        # Clamp log_std defensively (continuous mode) if present.
        for name, param in self.model.policy.named_parameters():
            if name.endswith("log_std"):
                with torch.no_grad():
                    param.data = torch.nan_to_num(
                        param.data,
                        nan=-2.0,
                        posinf=2.0,
                        neginf=-20.0,
                    )
                    param.data.clamp_(-20.0, 2.0)

        if bad_nan:
            logger.warning("NaN gradient(s) detected in: %s", ", ".join(bad_nan[:8]))
        if bad_inf:
            logger.warning("Inf gradient(s) detected in: %s", ", ".join(bad_inf[:8]))
        if bad_large:
            logger.warning("Large gradient(s) detected in: %s", ", ".join(bad_large[:4]))

        return True


class PerformanceCallback(BaseCallback):
    """
    Callback for tracking training performance metrics.

    Tracks rewards, episode lengths, and other metrics over time
    for analysis and visualization.

    Args:
        verbose: Verbosity level
        log_interval: Steps between logging
    """

    def __init__(self, verbose: int = 0, log_interval: int = 1000):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.current_rewards: List[float] = []
        self._step_count = 0

    def _on_step(self) -> bool:
        """Track metrics at each step."""
        self._step_count += 1

        # Get rewards from current step
        rewards = self.locals.get('rewards', [])
        if len(rewards) > 0:
            self.current_rewards.extend(rewards)

        # Check for episode end
        dones = self.locals.get('dones', [])
        for i, done in enumerate(dones):
            if done:
                if self.current_rewards:
                    self.episode_rewards.append(sum(self.current_rewards))
                    self.episode_lengths.append(len(self.current_rewards))
                    self.current_rewards = []

        # Log periodically
        if self.verbose > 0 and self._step_count % self.log_interval == 0:
            self._log_metrics()

        return True

    def _log_metrics(self) -> None:
        """Log current metrics."""
        if not self.episode_rewards:
            return
        
        recent_rewards = self.episode_rewards[-100:]
        recent_lengths = self.episode_lengths[-100:]
        
        logger.info(
            f"Episodes: {len(self.episode_rewards)}, "
            f"Avg reward: {np.mean(recent_rewards):.4f}, "
            f"Avg length: {np.mean(recent_lengths):.1f}"
        )

    def get_metrics(self) -> Dict[str, List]:
        """Get all tracked metrics."""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
        }


class TradingMetricsCallback(BaseCallback):
    """
    Callback for logging trading-specific metrics to TensorBoard.

    Tracks domain-specific metrics that are more meaningful for trading:
    - Sharpe ratio
    - Win rate
    - Action distribution
    - Net worth trajectory
    - Cumulative returns

    Args:
        verbose: Verbosity level
        log_interval: Steps between TensorBoard logging
        action_names: List of action names for distribution logging
    """

    def __init__(
        self,
        verbose: int = 0,
        log_interval: int = 1000,
        action_names: Optional[List[str]] = None,
        trend_window: int = 5,
    ):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.action_names = action_names or [
            "buy25", "buy50", "buy75", "buy100",
            "sell25", "sell50", "sell75", "sell100",
            "hold"
        ]
        self._step_count = 0
        self.trend_window = max(1, int(trend_window))
        
        # Tracking metrics
        self.returns: List[float] = []
        self.net_worths: List[float] = []
        self.actions: List[int] = []
        self.action_groups: List[str] = []  # buy/sell/hold/waiting
        self.trade_profits: List[float] = []
        self.fill_rates: List[float] = []
        self.entry_discounts: List[float] = []
        self.initial_balance = 10000.0  # Will be updated from env
        self.recent_run_means: List[float] = []
        self._is_continuous: bool = False

        # Indices to compute "batch" deltas since last log
        self._last_returns_idx: int = 0
        self._last_net_worths_idx: int = 0
        self._last_actions_idx: int = 0
        self._last_action_groups_idx: int = 0
        self._last_trade_profits_idx: int = 0
        self._last_recent_run_means_idx: int = 0
        self._last_fill_rates_idx: int = 0
        self._last_entry_discounts_idx: int = 0

        # Rolling window buffers over the last N log batches
        self._trend_buffers: Dict[str, Deque[float]] = {}
        # (wins, trades) per logged batch (rolling window)
        self._trend_win_trades: Deque[Tuple[int, int]] = deque(
            maxlen=self.trend_window
        )

    def _trend_push(self, key: str, value: float) -> float:
        """
        Push a batch value into the rolling window and return the current trend value.
        Default trend aggregation is mean over the last N batches.
        """
        if key not in self._trend_buffers:
            self._trend_buffers[key] = deque(maxlen=self.trend_window)
        buf = self._trend_buffers[key]
        buf.append(float(value))
        return float(np.mean(buf)) if len(buf) > 0 else float(value)

    def _trend_sum_push(self, key: str, value: float) -> float:
        """
        Push a batch value into the rolling window and return rolling sum over last N batches.
        """
        if key not in self._trend_buffers:
            self._trend_buffers[key] = deque(maxlen=self.trend_window)
        buf = self._trend_buffers[key]
        buf.append(float(value))
        return float(np.sum(buf)) if len(buf) > 0 else float(value)

    def _on_training_start(self) -> None:
        """Initialize metrics from environment."""
        try:
            env = (
                self.training_env.envs[0]
                if hasattr(self.training_env, 'envs')
                else self.training_env
            )
            if hasattr(env, 'initial_balance'):
                self.initial_balance = env.initial_balance
            if hasattr(env, "action_space"):
                self._is_continuous = isinstance(env.action_space, gym.spaces.Box)
        except Exception:
            pass

    def _on_step(self) -> bool:
        """Track trading metrics at each step."""
        self._step_count += 1

        # Get info from environment
        infos = self.locals.get('infos', [])
        actions = self.locals.get('actions', [])

        for info in infos:
            if not info:
                continue

            # Track net worth
            if 'net_worth' in info:
                self.net_worths.append(info['net_worth'])

            # Track returns
            if 'profit_loss' in info:
                self.returns.append(info['profit_loss'])

            # Track profitable trades
            if info.get('trade_executed'):
                # Use realized_pnl_pct if available, otherwise fallback to reward
                trade_profit = info.get('realized_pnl_pct', info.get('reward', 0))
                if trade_profit != 0:
                    self.trade_profits.append(trade_profit)

            # Track smoothed per-ticker returns for curriculum/competence
            if 'recent_run_mean_return' in info:
                self.recent_run_means.append(info['recent_run_mean_return'])
            
            if 'fill_rate' in info:
                self.fill_rates.append(info['fill_rate'])
            if 'avg_entry_discount' in info:
                self.entry_discounts.append(info['avg_entry_discount'])

        # Track actions
        if len(actions) > 0:
            for i, action in enumerate(actions):
                info = infos[i] if i < len(infos) else {}
                shares_held = float(info.get("shares_held", 0.0) or 0.0)

                if self._is_continuous:
                    # Bucketize continuous action into [sell, hold, buy] indices
                    val = float(np.asarray(action).reshape(-1)[0])
                    threshold = 0.05
                    if val > threshold:
                        self.actions.append(2)  # buy
                        action_group = "buy"
                    elif val < -threshold:
                        self.actions.append(0)  # sell
                        action_group = "sell"
                    else:
                        self.actions.append(1)  # hold
                        action_group = "hold"
                else:
                    action_idx = int(action)
                    self.actions.append(action_idx)
                    # Map discrete action names to buy/sell/hold group
                    action_key = (
                        self.action_names[action_idx]
                        if 0 <= action_idx < len(self.action_names)
                        else ""
                    )
                    if "buy" in action_key:
                        action_group = "buy"
                    elif "sell" in action_key:
                        action_group = "sell"
                    else:
                        action_group = "hold"

                # Reclassify hold as waiting when flat
                if action_group == "hold" and shares_held <= 0:
                    action_group = "waiting"

                self.action_groups.append(action_group)

        # Log to TensorBoard periodically
        if self._step_count % self.log_interval == 0:
            self._log_to_tensorboard()

        return True

    def _calculate_sharpe_ratio(self, returns: List[float], annualization: float = 252.0) -> float:
        """Calculate Sharpe ratio from returns."""
        if len(returns) < 2:
            return 0.0
        
        returns_arr = np.array(returns)
        mean_return = np.mean(returns_arr)
        std_return = np.std(returns_arr)
        
        if std_return < 1e-8:
            return 0.0
        
        # Annualized Sharpe (assuming daily returns)
        sharpe = (mean_return / std_return) * np.sqrt(annualization)
        return float(np.clip(sharpe, -10.0, 10.0))  # Clip to prevent extreme values

    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trade profits."""
        if len(self.trade_profits) == 0:
            return 0.5  # Default to 50% if no trades
        
        profitable = sum(1 for p in self.trade_profits if p > 0)
        return profitable / len(self.trade_profits)

    def _calculate_win_rate_from_trades(self, trade_profits: List[float]) -> Tuple[float, int, int]:
        """
        Calculate win rate from a batch of trade profits.

        Returns:
            (win_rate, wins, trades)
        """
        trades = len(trade_profits)
        if trades == 0:
            return 0.5, 0, 0
        wins = sum(1 for p in trade_profits if p > 0)
        return wins / trades, wins, trades

    def _get_action_distribution(self) -> Dict[str, float]:
        """Get distribution of actions taken."""
        if len(self.actions) == 0:
            return {name: 0.0 for name in self.action_names}
        
        action_counts = np.bincount(self.actions, minlength=len(self.action_names))
        total = action_counts.sum()
        
        if total == 0:
            return {name: 0.0 for name in self.action_names}
        
        return {
            name: float(count / total) 
            for name, count in zip(self.action_names, action_counts)
        }

    def _log_to_tensorboard(self) -> None:
        """Log trading metrics to TensorBoard."""
        # Treat each log event as one "batch": compute deltas since last log.
        batch_returns = (
            self.returns[self._last_returns_idx:] if self.returns else []
        )
        batch_net_worths = (
            self.net_worths[self._last_net_worths_idx:] if self.net_worths else []
        )
        batch_action_groups = (
            self.action_groups[self._last_action_groups_idx:]
            if self.action_groups
            else []
        )
        batch_trade_profits = (
            self.trade_profits[self._last_trade_profits_idx:]
            if self.trade_profits
            else []
        )
        batch_recent_run_means = (
            self.recent_run_means[self._last_recent_run_means_idx:]
            if self.recent_run_means
            else []
        )
        batch_fill_rates = (
            self.fill_rates[self._last_fill_rates_idx:]
            if self.fill_rates
            else []
        )
        batch_entry_discounts = (
            self.entry_discounts[self._last_entry_discounts_idx:]
            if self.entry_discounts
            else []
        )

        self._last_returns_idx = len(self.returns)
        self._last_net_worths_idx = len(self.net_worths)
        self._last_actions_idx = len(self.actions)
        self._last_action_groups_idx = len(self.action_groups)
        self._last_trade_profits_idx = len(self.trade_profits)
        self._last_recent_run_means_idx = len(self.recent_run_means)
        self._last_fill_rates_idx = len(self.fill_rates)
        self._last_entry_discounts_idx = len(self.entry_discounts)

        # Calculate Sharpe ratio
        sharpe_raw = self._calculate_sharpe_ratio(batch_returns)
        sharpe = self._trend_push("sharpe_ratio", sharpe_raw)
        self.logger.record("trading/sharpe_ratio", sharpe)
        self.logger.record("trading_raw/sharpe_ratio", sharpe_raw)

        # Calculate win rate
        win_rate_raw, wins, trades = self._calculate_win_rate_from_trades(
            batch_trade_profits
        )
        self._trend_win_trades.append((wins, trades))
        win_trades = sum(t for _, t in self._trend_win_trades)
        win_wins = sum(w for w, _ in self._trend_win_trades)
        win_rate = (win_wins / win_trades) if win_trades > 0 else 0.5
        self.logger.record("trading/win_rate", float(win_rate))
        self.logger.record("trading_raw/win_rate", float(win_rate_raw))

        # Log net worth stats
        if batch_net_worths:
            nw_mean_raw = float(np.mean(batch_net_worths))
            nw_max_raw = float(np.max(batch_net_worths))
            nw_min_raw = float(np.min(batch_net_worths))

            self.logger.record(
                "trading/net_worth_mean",
                self._trend_push("net_worth_mean", nw_mean_raw),
            )
            self.logger.record(
                "trading/net_worth_max",
                self._trend_push("net_worth_max", nw_max_raw),
            )
            self.logger.record(
                "trading/net_worth_min",
                self._trend_push("net_worth_min", nw_min_raw),
            )

            self.logger.record("trading_raw/net_worth_mean", nw_mean_raw)
            self.logger.record("trading_raw/net_worth_max", nw_max_raw)
            self.logger.record("trading_raw/net_worth_min", nw_min_raw)

            # Portfolio return
            if self.initial_balance > 0:
                portfolio_return_raw = (
                    (batch_net_worths[-1] - self.initial_balance)
                    / self.initial_balance
                )
                self.logger.record(
                    "trading/portfolio_return_pct",
                    self._trend_push(
                        "portfolio_return_pct",
                        float(portfolio_return_raw * 100),
                    ),
                )
                self.logger.record(
                    "trading_raw/portfolio_return_pct",
                    float(portfolio_return_raw * 100),
                )

        # Log cumulative return
        if batch_returns:
            cum_return_raw = float(sum(batch_returns))
            cum_return = self._trend_sum_push("cumulative_return", cum_return_raw)
            self.logger.record("trading/cumulative_return", cum_return)
            self.logger.record("trading_raw/cumulative_return", cum_return_raw)

        # Log smoothed run mean if available
        if batch_recent_run_means:
            rrm_raw = float(batch_recent_run_means[-1])
            self.logger.record(
                "trading/recent_run_mean_return",
                self._trend_push("recent_run_mean_return", rrm_raw),
            )
            self.logger.record("trading_raw/recent_run_mean_return", rrm_raw)

        # Log action distribution (grouped)
        # Use batch action groups so proportions represent the recent batch.
        total_actions = len(batch_action_groups) if batch_action_groups else 0
        if total_actions == 0:
            buy_pct = 0.0
            sell_pct = 0.0
            hold_pct = 0.0
            waiting_pct = 0.0
        else:
            buy_pct = batch_action_groups.count("buy") / total_actions
            sell_pct = batch_action_groups.count("sell") / total_actions
            hold_pct = batch_action_groups.count("hold") / total_actions
            waiting_pct = batch_action_groups.count("waiting") / total_actions

        buy_pct_trend = self._trend_push("action_buy_pct", float(buy_pct))
        sell_pct_trend = self._trend_push("action_sell_pct", float(sell_pct))
        hold_pct_trend = self._trend_push("action_hold_pct", float(hold_pct))
        waiting_pct_trend = self._trend_push("action_waiting_pct", float(waiting_pct))

        self.logger.record("trading/action_buy_pct", buy_pct_trend)
        self.logger.record("trading/action_sell_pct", sell_pct_trend)
        self.logger.record("trading/action_hold_pct", hold_pct_trend)
        self.logger.record("trading/action_waiting_pct", waiting_pct_trend)

        self.logger.record("trading_raw/action_buy_pct", float(buy_pct))
        self.logger.record("trading_raw/action_sell_pct", float(sell_pct))
        self.logger.record("trading_raw/action_hold_pct", float(hold_pct))
        self.logger.record("trading_raw/action_waiting_pct", float(waiting_pct))

        # Log total trades executed
        trades_raw = float(len(batch_trade_profits))
        trades_trend = self._trend_sum_push("total_trades", trades_raw)
        self.logger.record("trading/total_trades", trades_trend)
        self.logger.record("trading_raw/total_trades", float(len(self.trade_profits)))
        self.logger.record("trading_raw/trades_batch", trades_raw)

        # Log Fill Rate and Entry Discount
        if batch_fill_rates:
            fr_raw = float(batch_fill_rates[-1])
            self.logger.record("trading/fill_rate", self._trend_push("fill_rate", fr_raw))
            self.logger.record("trading_raw/fill_rate", fr_raw)
        
        if batch_entry_discounts:
            ed_raw = float(batch_entry_discounts[-1])
            self.logger.record("trading/avg_entry_discount", self._trend_push("avg_entry_discount", ed_raw))
            self.logger.record("trading_raw/avg_entry_discount", ed_raw)

        if self.verbose > 0:
            logger.info(
                f"Trading metrics @ {self._step_count}: "
                f"Sharpe(trend)={sharpe:.3f}, Win(trend)={float(win_rate):.2%}, "
                f"Buy(trend)={buy_pct_trend:.1%}, "
                f"Sell(trend)={sell_pct_trend:.1%}, "
                f"Hold(trend)={hold_pct_trend:.1%}, "
                f"Waiting(trend)={waiting_pct_trend:.1%}, "
                f"Trades(last {self.trend_window})={int(trades_trend)}"
            )

    def _on_training_end(self) -> None:
        """Log final summary."""
        if self.verbose > 0:
            sharpe = self._calculate_sharpe_ratio(self.returns)
            win_rate = self._calculate_win_rate()
            final_net_worth = self.net_worths[-1] if self.net_worths else self.initial_balance
            total_return = (final_net_worth - self.initial_balance) / self.initial_balance
            
            logger.info("=" * 50)
            logger.info("TRADING METRICS SUMMARY")
            logger.info("=" * 50)
            logger.info(f"Final Net Worth: £{final_net_worth:,.2f}")
            logger.info(f"Total Return: {total_return:.2%}")
            logger.info(f"Sharpe Ratio: {sharpe:.3f}")
            logger.info(f"Win Rate: {win_rate:.2%}")
            logger.info(f"Total Trades: {len(self.trade_profits)}")


class EpisodeMetricsCallback(BaseCallback):
    """
    Custom callback to log specific metrics at the end of episodes.
    
    Logs to TensorBoard:
    - portfolio_value: Final net worth
    - win_rate: (Winning Trades / Total Trades)
    - profit_factor: (Gross Profit / Gross Loss)
    - trade_count: Number of trades executed
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_trades: List[Dict[str, Any]] = []
        self.current_episode_trades: List[Dict[str, Any]] = []
        self.episode_net_worths: List[float] = []
        self.initial_balance = 10000.0
        
    def _on_training_start(self) -> None:
        """Initialize metrics from environment."""
        try:
            env = (
                self.training_env.envs[0]
                if hasattr(self.training_env, 'envs')
                else self.training_env
            )
            # Unwrap VecNormalize if present
            while hasattr(env, 'venv'):
                env = env.venv
            if hasattr(env, 'envs') and len(env.envs) > 0:
                env = env.envs[0]
            
            if hasattr(env, 'initial_balance'):
                self.initial_balance = env.initial_balance
        except Exception:
            pass
    
    def _on_step(self) -> bool:
        """Track trades and portfolio value at each step."""
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        
        for i, info in enumerate(infos):
            if not info:
                continue
            
            # Track trade execution
            if info.get('trade_executed', False):
                trade_info = {
                    'profit_loss': info.get('profit_loss', 0.0),
                    'net_worth': info.get('net_worth', self.initial_balance),
                }
                self.current_episode_trades.append(trade_info)
            
            # Check for episode end
            done = dones[i] if i < len(dones) else False
            if done:
                # Episode ended, log metrics
                if self.current_episode_trades:
                    self.episode_trades.append(self.current_episode_trades.copy())
                    self.current_episode_trades.clear()
                
                # Get final net worth
                final_net_worth = info.get('net_worth', self.initial_balance)
                self.episode_net_worths.append(final_net_worth)
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Log metrics at the end of each rollout."""
        if not self.episode_trades and not self.episode_net_worths:
            return
        
        # Calculate metrics from recent episodes
        if self.episode_net_worths:
            # Portfolio value (final net worth)
            portfolio_value = self.episode_net_worths[-1]
            self.logger.record("episode/portfolio_value", portfolio_value)
        
        if self.episode_trades:
            # Aggregate all trades from recent episodes
            all_trades = []
            for episode_trades in self.episode_trades[-10:]:  # Last 10 episodes
                all_trades.extend(episode_trades)
            
            if all_trades:
                # Win rate: (Winning Trades / Total Trades)
                total_trades = len(all_trades)
                winning_trades = sum(1 for t in all_trades if t.get('profit_loss', 0.0) > 0)
                win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
                self.logger.record("episode/win_rate", win_rate)
                
                # Profit factor: (Gross Profit / Gross Loss)
                gross_profit = sum(t.get('profit_loss', 0.0) for t in all_trades if t.get('profit_loss', 0.0) > 0)
                gross_loss = abs(sum(t.get('profit_loss', 0.0) for t in all_trades if t.get('profit_loss', 0.0) < 0))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 1.0)
                self.logger.record("episode/profit_factor", profit_factor)
                
                # Trade count
                trade_count = total_trades
                self.logger.record("episode/trade_count", trade_count)
        
        # Clear old data to prevent memory buildup
        if len(self.episode_trades) > 100:
            self.episode_trades = self.episode_trades[-50:]
        if len(self.episode_net_worths) > 100:
            self.episode_net_worths = self.episode_net_worths[-50:]