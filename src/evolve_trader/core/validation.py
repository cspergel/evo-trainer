"""Walk-forward validation harness for strategy evolution.

Ensures strategies are evolved on training data and validated on
out-of-sample data. Prevents overfitting by gating promotion on
out-of-sample performance.

Reference architectures (both MIT licensed):
- futures-backtesting-engine: no-lookahead 6-phase bar loop
- july-backtester: walk-forward overfitting detection
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""

    total_days: int
    train_days: int = 30
    validate_days: int = 10
    step_days: int = 10


@dataclass
class WalkForwardWindow:
    """A single train/validate window."""

    train_start: int  # day index
    train_end: int
    validate_start: int
    validate_end: int


def generate_walk_forward_windows(
    config: WalkForwardConfig,
) -> list[WalkForwardWindow]:
    """Generate walk-forward train/validate windows.

    Each window has a training period followed immediately by a validation
    period. Windows step forward by step_days, so training periods may
    overlap but validation never sees training data.
    """
    windows: list[WalkForwardWindow] = []
    start = 0

    while start + config.train_days + config.validate_days <= config.total_days:
        train_start = start
        train_end = start + config.train_days
        validate_start = train_end
        validate_end = validate_start + config.validate_days

        windows.append(
            WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                validate_start=validate_start,
                validate_end=validate_end,
            )
        )
        start += config.step_days

    return windows


def validate_no_lookahead(window: WalkForwardWindow, data_timestamp: int) -> bool:
    """Check if a data timestamp is valid for the training period.

    Returns True if data is strictly before the training end (usable).
    Returns False if data is at or after training end (lookahead violation).
    """
    return data_timestamp < window.train_end
