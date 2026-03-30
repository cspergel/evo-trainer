"""ExecutionCostModel — estimates trading costs that erode alpha.

Per profitability contract section 2: we measure alpha after spread,
slippage, delay, and commissions. Strategies whose edge is less than
2x estimated round-trip cost are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostEstimate:
    """Breakdown of estimated execution costs for a trade."""

    spread_bps: float  # Half bid-ask spread, entry + exit
    slippage_bps: float  # Market impact based on order size vs ADV
    commission_bps: float  # Broker fees (Alpaca: ~0, but SEC/FINRA fees remain)
    delay_bps: float  # Price drift from signal to execution

    @property
    def total_round_trip_bps(self) -> float:
        """Total estimated round-trip cost in basis points."""
        return self.spread_bps + self.slippage_bps + self.commission_bps + self.delay_bps

    @property
    def total_round_trip_pct(self) -> float:
        """Total round-trip cost as a fraction (e.g., 0.001 = 10bps)."""
        return self.total_round_trip_bps / 10_000


# Default cost assumptions by market cap tier
_LARGE_CAP_SPREAD_BPS = 2.0  # ~0.02% half-spread for S&P 500
_MID_CAP_SPREAD_BPS = 10.0  # ~0.10% for mid-cap
_SMALL_CAP_SPREAD_BPS = 25.0  # ~0.25% for small-cap

# SEC fee + FINRA TAF (approximate, Alpaca is $0 commission)
_COMMISSION_BPS = 0.3

# Default signal-to-execution delay cost
_DELAY_BPS_PER_DAY = 2.0  # ~2bps per day of delay


def estimate_costs(
    order_value: float,
    average_daily_volume: float = 0.0,
    market_cap_tier: str = "large",
    signal_delay_days: float = 0.0,
) -> CostEstimate:
    """Estimate execution costs for a proposed trade.

    Args:
        order_value: Dollar value of the order.
        average_daily_volume: Average daily dollar volume of the instrument.
        market_cap_tier: "large", "mid", or "small" cap.
        signal_delay_days: Days between signal generation and intended execution.

    Returns:
        CostEstimate with per-component breakdown.
    """
    # Spread by cap tier
    if market_cap_tier == "large":
        spread_bps = _LARGE_CAP_SPREAD_BPS
    elif market_cap_tier == "mid":
        spread_bps = _MID_CAP_SPREAD_BPS
    else:
        spread_bps = _SMALL_CAP_SPREAD_BPS

    # Slippage: square-root market impact model
    # impact_bps ~ coefficient * sqrt(participation_pct)
    # Calibrated: ~1bps at 0.01% ADV, ~10bps at 1% ADV
    slippage_bps = 0.0
    if average_daily_volume > 0:
        participation_pct = (order_value / average_daily_volume) * 100
        slippage_bps = 3.0 * (participation_pct**0.5)
        slippage_bps = min(slippage_bps, 50.0)
    else:
        slippage_bps = 5.0

    # Delay cost
    delay_bps = signal_delay_days * _DELAY_BPS_PER_DAY

    return CostEstimate(
        spread_bps=spread_bps,
        slippage_bps=round(slippage_bps, 2),
        commission_bps=_COMMISSION_BPS,
        delay_bps=round(delay_bps, 2),
    )


def check_edge_vs_cost(
    expected_edge_bps: float,
    cost_estimate: CostEstimate,
    min_ratio: float = 2.0,
) -> bool:
    """Check if expected edge is at least min_ratio times the cost.

    Per profitability contract: reject if edge < 2x round-trip cost.
    """
    if cost_estimate.total_round_trip_bps <= 0:
        return True
    return expected_edge_bps / cost_estimate.total_round_trip_bps >= min_ratio
