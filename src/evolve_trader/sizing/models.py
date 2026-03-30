"""Position sizing skill schema and sizing methods.

Sizing skills are independent from strategy skills:
- Strategy says WHAT to trade and WHEN
- Sizing says HOW MUCH

All sizing methods produce a SizingResult that can be vetoed or
scaled by portfolio-level constraints before execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SizingMethod(Enum):
    """Available position sizing approaches."""

    KELLY = "kelly"
    VOLATILITY_TARGET = "volatility_target"
    FIXED_FRACTIONAL = "fixed_fractional"
    REGIME_ADJUSTED = "regime_adjusted"


@dataclass
class SizingSkill:
    """An evolvable position sizing skill."""

    name: str
    method: SizingMethod
    description: str
    # Method-specific parameters (evolvable)
    params: dict[str, float]


@dataclass
class SizingResult:
    """Output of a sizing calculation."""

    position_size_pct: float  # % of portfolio (0.0 to 1.0)
    position_value: float  # Dollar amount
    shares: float  # Number of shares
    method_used: str
    rationale: str


@dataclass
class SizingContext:
    """Inputs needed for any sizing calculation."""

    portfolio_value: float
    current_price: float
    win_rate: float  # Historical win rate for this strategy
    avg_win: float  # Average winning return
    avg_loss: float  # Average losing return (positive number)
    recent_volatility: float  # Annualized volatility of the asset
    regime: str  # Current regime label
    existing_exposure: float  # Current portfolio exposure (0.0 to 1.0)


def compute_kelly_size(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    kelly_fraction: float = 0.5,
) -> float:
    """Compute position size using fractional Kelly criterion.

    Full Kelly is too aggressive — default to half-Kelly (kelly_fraction=0.5).

    Returns position size as fraction of portfolio (0.0 to 1.0).
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0

    # Kelly formula: f = (p * b - q) / b
    # where p = win rate, q = 1 - p, b = avg_win / avg_loss
    b = avg_win / avg_loss
    q = 1.0 - win_rate
    kelly_full = (win_rate * b - q) / b

    if kelly_full <= 0:
        return 0.0

    return min(kelly_full * kelly_fraction, 1.0)


def compute_volatility_target_size(
    target_volatility: float,
    asset_volatility: float,
    portfolio_value: float,
    current_price: float,
) -> float:
    """Compute position size to achieve a target volatility contribution.

    Scales position inversely to asset volatility.

    Returns position size as fraction of portfolio.
    """
    if asset_volatility <= 0 or current_price <= 0:
        return 0.0

    # Position size = target_vol / asset_vol
    size_pct = target_volatility / asset_volatility
    return min(size_pct, 1.0)


def compute_regime_adjusted_size(
    base_size_pct: float,
    regime: str,
    regime_multipliers: dict[str, float] | None = None,
) -> float:
    """Adjust position size based on current market regime.

    Risk-off → reduce exposure. Risk-on → full or increased exposure.
    """
    defaults = {
        "risk-on": 1.0,
        "risk-off": 0.5,
        "transitional": 0.7,
    }
    multipliers = regime_multipliers or defaults
    multiplier = multipliers.get(regime, 0.7)
    return min(base_size_pct * multiplier, 1.0)


def compute_sizing(
    context: SizingContext,
    skill: SizingSkill,
) -> SizingResult:
    """Compute position size using the specified sizing skill.

    This is the main entry point — dispatches to the appropriate method.
    """
    if skill.method == SizingMethod.KELLY:
        kelly_frac = skill.params.get("kelly_fraction", 0.5)
        size_pct = compute_kelly_size(
            context.win_rate, context.avg_win, context.avg_loss, kelly_frac
        )
        rationale = (
            f"Half-Kelly ({kelly_frac:.0%}) based on "
            f"win_rate={context.win_rate:.0%}, "
            f"avg_win={context.avg_win:.1%}, avg_loss={context.avg_loss:.1%}"
        )

    elif skill.method == SizingMethod.VOLATILITY_TARGET:
        target_vol = skill.params.get("target_volatility", 0.15)
        size_pct = compute_volatility_target_size(
            target_vol,
            context.recent_volatility,
            context.portfolio_value,
            context.current_price,
        )
        rationale = (
            f"Volatility targeting {target_vol:.0%} vs "
            f"asset vol {context.recent_volatility:.0%}"
        )

    elif skill.method == SizingMethod.REGIME_ADJUSTED:
        base = skill.params.get("base_size", 0.03)
        size_pct = compute_regime_adjusted_size(base, context.regime)
        rationale = f"Regime-adjusted: base {base:.1%} in {context.regime} regime"

    else:
        # Fixed fractional fallback
        size_pct = skill.params.get("fixed_fraction", 0.02)
        rationale = f"Fixed fractional: {size_pct:.1%} of portfolio"

    # Convert to dollar value and shares
    position_value = context.portfolio_value * size_pct
    shares = position_value / context.current_price if context.current_price > 0 else 0.0

    return SizingResult(
        position_size_pct=round(size_pct, 6),
        position_value=round(position_value, 2),
        shares=round(shares, 2),
        method_used=skill.method.value,
        rationale=rationale,
    )
