"""TradeIntent — the canonical trade object per the master spec.

Contains strategy, sizing, regime, signals, confidence, rationale,
and projected portfolio impact. This is what flows through the
3-gate execution pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TradeIntent:
    """A fully specified trade intent ready for gate processing.

    Per profitability contract: includes rationale_summary and
    rationale_evidence as structured data, not raw LLM output.
    """

    ticker: str
    direction: str  # "BUY" or "SELL"
    quantity: float
    order_type: str = "MARKET"  # MARKET, LIMIT
    strategy_skill: str = ""
    strategy_lineage: str = ""  # version DAG path
    sizing_skill: str = ""
    sizing_rationale: str = ""
    regime_label: str = ""
    regime_confidence: float = 0.0
    signal_sources: list[str] = field(default_factory=list)
    confidence: float = 0.0
    rationale_summary: str = ""
    rationale_evidence: dict[str, object] = field(default_factory=dict)
    position_impact: dict[str, float] = field(default_factory=dict)
    paper_track_record: dict[str, float] = field(default_factory=dict)
    estimated_cost_bps: float = 0.0
    created_at: datetime | None = None
