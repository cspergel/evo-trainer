"""Real market replay — evaluates strategies against historical prices.

Replaces synthetic trade generation. Uses yfinance price data to
simulate strategy execution with no-lookahead enforcement.

Per profitability contract: pessimistic execution assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass

from evolve_trader.core.analyzer import TradeResult
from evolve_trader.core.execution_costs import estimate_costs
from evolve_trader.core.market_data import PriceSeries


@dataclass
class ReplayConfig:
    """Configuration for a market replay run."""

    holding_period_days: int = 5  # Days to weeks per contract scope
    entry_signal_threshold: float = 0.02  # Min daily return to trigger entry
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.08  # 8% take profit
    signal_delay_days: int = 1  # Execute next day (no lookahead)


def replay_strategy_on_prices(
    prices: PriceSeries,
    config: ReplayConfig | None = None,
) -> list[TradeResult]:
    """Replay a simple momentum strategy against real price data.

    No-lookahead: entry signal at close[t], execute at open[t+1].
    Exit on stop loss, take profit, or holding period expiry.

    Returns actual trade results with real prices.
    """
    if config is None:
        config = ReplayConfig()

    bars = prices.bars
    if len(bars) < config.holding_period_days + config.signal_delay_days + 2:
        return []

    trades: list[TradeResult] = []
    i = config.signal_delay_days  # Start after delay (no lookahead)

    while i < len(bars) - config.holding_period_days:
        # Entry signal: previous day's return exceeds threshold
        prev_return = (bars[i].close - bars[i - 1].close) / bars[i - 1].close

        if abs(prev_return) >= config.entry_signal_threshold:
            direction = "long" if prev_return > 0 else "short"
            entry_price = bars[i + 1].open  # Execute next day open (no lookahead)
            entry_date = bars[i + 1].date

            # Simulate holding period
            exit_price = entry_price
            exit_date = entry_date

            for j in range(1, config.holding_period_days + 1):
                if i + 1 + j >= len(bars):
                    break

                current_bar = bars[i + 1 + j]

                if direction == "long":
                    # Check stop loss
                    low_change = (current_bar.low - entry_price) / entry_price
                    if low_change <= -config.stop_loss_pct:
                        exit_price = entry_price * (1 - config.stop_loss_pct)
                        exit_date = current_bar.date
                        break
                    # Check take profit
                    high_change = (current_bar.high - entry_price) / entry_price
                    if high_change >= config.take_profit_pct:
                        exit_price = entry_price * (1 + config.take_profit_pct)
                        exit_date = current_bar.date
                        break
                else:
                    high_change = (current_bar.high - entry_price) / entry_price
                    if high_change >= config.stop_loss_pct:
                        exit_price = entry_price * (1 + config.stop_loss_pct)
                        exit_date = current_bar.date
                        break
                    low_change = (entry_price - current_bar.low) / entry_price
                    if low_change >= config.take_profit_pct:
                        exit_price = entry_price * (1 - config.take_profit_pct)
                        exit_date = current_bar.date
                        break

                exit_price = current_bar.close
                exit_date = current_bar.date

            # Apply execution costs
            cost = estimate_costs(
                order_value=entry_price * 10,  # 10 shares
                average_daily_volume=bars[i].volume * bars[i].close,
                market_cap_tier="large",
                signal_delay_days=config.signal_delay_days,
            )
            cost_pct = cost.total_round_trip_pct

            # Adjust exit price for costs
            if direction == "long":
                exit_price_after_costs = exit_price * (1 - cost_pct)
            else:
                exit_price_after_costs = exit_price * (1 + cost_pct)

            trades.append(
                TradeResult(
                    ticker=prices.ticker,
                    entry_price=entry_price,
                    exit_price=exit_price_after_costs,
                    shares=10,
                    entry_date=str(entry_date),
                    exit_date=str(exit_date),
                    reasoning=f"{direction} on momentum signal",
                )
            )

            # Skip ahead past holding period
            i += config.holding_period_days + 1
        else:
            i += 1

    return trades
