"""
Learning rate schedules for training.

This module provides various learning rate schedule functions compatible
with Stable-Baselines3.
"""

import math
from typing import Callable, List


def linear_schedule(
    initial_value: float, final_value: float, decay_fraction: float
) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    
    Linearly decays from initial_value to final_value over decay_fraction
    of training, then maintains final_value.
    
    Args:
        initial_value: Starting learning rate
        final_value: Final learning rate
        decay_fraction: Fraction of training over which to decay
    
    Returns:
        Learning rate schedule function
    """
    def lr_schedule(progress_remaining: float) -> float:
        if progress_remaining > (1 - decay_fraction):
            fraction = (progress_remaining - (1 - decay_fraction)) / decay_fraction
            return initial_value - fraction * (initial_value - final_value)
        return final_value

    return lr_schedule


def exponential_schedule(
    initial_value: float, decay_rate: float
) -> Callable[[float], float]:
    """
    Simple exponential learning rate schedule.
    
    Args:
        initial_value: Starting learning rate
        decay_rate: Exponential decay rate
    
    Returns:
        Learning rate schedule function
    """
    def lr_schedule(progress_remaining: float) -> float:
        return initial_value * (decay_rate ** (1 - progress_remaining))
    
    return lr_schedule


def step_schedule(
    initial_value: float, decay_factor: float, step_intervals: List[float]
) -> Callable[[float], float]:
    """
    Step learning rate schedule.
    
    Reduces learning rate by decay_factor at each interval.
    
    Args:
        initial_value: Starting learning rate
        decay_factor: Factor to multiply learning rate at each step
        step_intervals: Progress thresholds at which to decay
    
    Returns:
        Learning rate schedule function
    """
    def lr_schedule(progress_remaining: float) -> float:
        current_lr = initial_value
        for interval in step_intervals:
            if progress_remaining <= interval:
                current_lr *= decay_factor
        return current_lr
    
    return lr_schedule


def exponential_decay_schedule(
    initial_value: float, final_value: float, decay_fraction: float
) -> Callable[[float], float]:
    """
    Exponential decay learning rate schedule with target final value.
    
    Exponentially decays from initial_value to final_value over decay_fraction
    of training, then maintains final_value.
    
    Args:
        initial_value: Starting learning rate
        final_value: Target final learning rate
        decay_fraction: Fraction of training over which to decay
    
    Returns:
        Learning rate schedule function
    """
    decay_rate = math.log(final_value / initial_value) / decay_fraction

    def lr_schedule(progress_remaining: float) -> float:
        if progress_remaining > (1 - decay_fraction):
            fraction = (progress_remaining - (1 - decay_fraction)) / decay_fraction
            lr = initial_value * math.exp(decay_rate * (1 - fraction))
            return max(lr, final_value)
        return final_value

    return lr_schedule


def cosine_annealing_schedule(
    initial_value: float, final_value: float, warmup_fraction: float = 0.0
) -> Callable[[float], float]:
    """
    Cosine annealing learning rate schedule.
    
    Follows a cosine curve from initial_value to final_value, with optional
    linear warmup at the beginning.
    
    Args:
        initial_value: Peak learning rate
        final_value: Minimum learning rate
        warmup_fraction: Fraction of training for linear warmup
    
    Returns:
        Learning rate schedule function
    """
    def lr_schedule(progress_remaining: float) -> float:
        progress = 1 - progress_remaining
        
        # Linear warmup
        if progress < warmup_fraction:
            return initial_value * (progress / warmup_fraction)
        
        # Cosine annealing
        adjusted_progress = (progress - warmup_fraction) / (1 - warmup_fraction)
        return final_value + (initial_value - final_value) * 0.5 * (
            1 + math.cos(math.pi * adjusted_progress)
        )
    
    return lr_schedule


def warmup_exponential_schedule(
    initial_value: float,
    peak_value: float,
    final_value: float,
    warmup_fraction: float = 0.1,
    decay_fraction: float = 0.8,
) -> Callable[[float], float]:
    """
    Learning rate schedule with warmup followed by exponential decay.
    
    Args:
        initial_value: Starting learning rate during warmup
        peak_value: Peak learning rate after warmup
        final_value: Final learning rate
        warmup_fraction: Fraction of training for warmup
        decay_fraction: Fraction of training for decay (after warmup)
    
    Returns:
        Learning rate schedule function
    """
    decay_rate = math.log(final_value / peak_value) / decay_fraction

    def lr_schedule(progress_remaining: float) -> float:
        progress = 1 - progress_remaining
        
        # Warmup phase
        if progress < warmup_fraction:
            return initial_value + (peak_value - initial_value) * (progress / warmup_fraction)
        
        # Decay phase
        decay_progress = (progress - warmup_fraction) / (1 - warmup_fraction)
        if decay_progress < decay_fraction:
            return peak_value * math.exp(decay_rate * decay_progress)
        
        return final_value

    return lr_schedule


def linear_warmup_cosine_decay(
    initial_value: float,
    peak_value: float,
    final_value: float,
    warmup_fraction: float = 0.1,
) -> Callable[[float], float]:
    """
    Linear warmup followed by cosine decay.

    Args:
        initial_value: Starting learning rate for warmup
        peak_value: Peak learning rate after warmup
        final_value: Final learning rate after decay
        warmup_fraction: Fraction of training for the warmup phase

    Returns:
        Learning rate schedule function
    """

    def lr_schedule(progress_remaining: float) -> float:
        progress = 1 - progress_remaining
        progress = min(max(progress, 0.0), 1.0)

        if progress < warmup_fraction:
            warmup_progress = progress / warmup_fraction
            return initial_value + (peak_value - initial_value) * warmup_progress

        decay_progress = (progress - warmup_fraction) / max(1e-8, (1 - warmup_fraction))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
        return final_value + (peak_value - final_value) * cosine_decay

    return lr_schedule
