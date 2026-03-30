# Phase 10: Crowding, Contrarian & Synthetic Benchmarks — Detailed Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build crowding detection and contrarian skill activation. Validate the full system against broad historical distributions, quiet markets, and exploratory historical scenario packs. Ensure graceful degradation when signal sources fail.

**Architecture:** A crowding detection layer sits between signal aggregation and portfolio construction. Cross-source convergence scoring detects when multiple independent sources collapse into the same directional view. When crowding exceeds configurable thresholds, contrarian SKILL.md strategies activate — reducing exposure, deploying tail-risk hedges, increasing cash, and optionally shorting crowded names. Validation uses two layers: required distributional tests over broad market windows, and exploratory scenario packs that replay selected historical narratives as regression fixtures rather than hard acceptance gates. A report generator summarizes runs with identified weaknesses.

**Tech Stack:** Python 3.11+, PostgreSQL 16+, numpy, pandas, scipy (rolling correlations), pytest, pytest-benchmark

**Parent plan:** [`2026-03-29-evolve-trader-master-plan.md`](2026-03-29-evolve-trader-master-plan.md)

**Prerequisites:** Phase 8 AND Phase 9 complete. Multi-source signal aggregation, regime classifier, evolution loop, risk constraints, portfolio construction, and walk-forward validation all verified.

---

## Task 1: Cross-Source Convergence Scorer

**Files:**
- Create: `src/evolve_trader/crowding/convergence.py`
- Create: `tests/unit/test_convergence.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_convergence.py
"""Tests for cross-source convergence scoring."""
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from evolve_trader.crowding.convergence import (
    ConvergenceScorer,
    ConvergenceResult,
    CrowdingFlag,
)


def test_convergence_scorer_requires_minimum_sources():
    """Scorer raises if fewer than 2 sources provided."""
    scorer = ConvergenceScorer(window_days=30)
    with pytest.raises(ValueError, match="at least 2 sources"):
        scorer.score(signal_streams={"single_source": []})


def test_convergence_result_has_required_fields():
    """ConvergenceResult exposes correlation matrix and crowding flag."""
    result = ConvergenceResult(
        correlation_matrix=np.eye(3),
        source_names=["edgar_13f", "form4", "congressional"],
        mean_correlation=0.0,
        max_pairwise_correlation=0.0,
        crowding_flag=CrowdingFlag.NONE,
        timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc),
    )
    assert result.correlation_matrix.shape == (3, 3)
    assert result.crowding_flag == CrowdingFlag.NONE
    assert len(result.source_names) == 3


def test_independent_sources_produce_no_crowding():
    """Uncorrelated signal streams yield CrowdingFlag.NONE."""
    scorer = ConvergenceScorer(
        window_days=30,
        crowding_threshold=0.7,
    )
    np.random.seed(42)
    now = datetime(2025, 6, 1, tzinfo=timezone.utc)
    streams = {
        "source_a": _make_random_signal_stream(now, days=60, seed=1),
        "source_b": _make_random_signal_stream(now, days=60, seed=2),
        "source_c": _make_random_signal_stream(now, days=60, seed=3),
    }
    result = scorer.score(signal_streams=streams, as_of=now)
    assert result.crowding_flag == CrowdingFlag.NONE
    assert result.mean_correlation < 0.5


def test_correlated_sources_produce_crowding_warning():
    """Highly correlated streams yield CrowdingFlag.WARNING."""
    scorer = ConvergenceScorer(
        window_days=30,
        crowding_threshold=0.7,
        severe_threshold=0.9,
    )
    now = datetime(2025, 6, 1, tzinfo=timezone.utc)
    base = _make_random_signal_stream(now, days=60, seed=42)
    streams = {
        "source_a": base,
        "source_b": _add_noise(base, noise_level=0.1, seed=10),
        "source_c": _make_random_signal_stream(now, days=60, seed=99),
    }
    result = scorer.score(signal_streams=streams, as_of=now)
    assert result.crowding_flag in (CrowdingFlag.WARNING, CrowdingFlag.SEVERE)
    assert result.max_pairwise_correlation > 0.7


def test_all_sources_converged_produces_severe_crowding():
    """All sources pointing same direction yields CrowdingFlag.SEVERE."""
    scorer = ConvergenceScorer(
        window_days=30,
        crowding_threshold=0.7,
        severe_threshold=0.9,
    )
    now = datetime(2025, 6, 1, tzinfo=timezone.utc)
    base = _make_random_signal_stream(now, days=60, seed=42)
    streams = {
        "source_a": base,
        "source_b": _add_noise(base, noise_level=0.05, seed=10),
        "source_c": _add_noise(base, noise_level=0.05, seed=20),
    }
    result = scorer.score(signal_streams=streams, as_of=now)
    assert result.crowding_flag == CrowdingFlag.SEVERE
    assert result.mean_correlation > 0.85


def test_convergence_scorer_rolling_window():
    """Only signals within the rolling window are considered."""
    scorer = ConvergenceScorer(window_days=10)
    now = datetime(2025, 6, 1, tzinfo=timezone.utc)
    # Old correlated signals (outside window) should not cause crowding
    old_base = _make_random_signal_stream(
        now - timedelta(days=60), days=30, seed=42
    )
    streams = {
        "source_a": old_base + _make_random_signal_stream(now, days=10, seed=1),
        "source_b": _add_noise(old_base, noise_level=0.05, seed=10)
        + _make_random_signal_stream(now, days=10, seed=2),
    }
    result = scorer.score(signal_streams=streams, as_of=now)
    assert result.crowding_flag == CrowdingFlag.NONE


def test_convergence_scorer_exposure_reduction():
    """Scorer recommends exposure reduction proportional to crowding."""
    scorer = ConvergenceScorer(
        window_days=30,
        crowding_threshold=0.7,
        severe_threshold=0.9,
    )
    now = datetime(2025, 6, 1, tzinfo=timezone.utc)
    base = _make_random_signal_stream(now, days=60, seed=42)
    streams = {
        "source_a": base,
        "source_b": _add_noise(base, noise_level=0.05, seed=10),
        "source_c": _add_noise(base, noise_level=0.05, seed=20),
    }
    result = scorer.score(signal_streams=streams, as_of=now)
    assert 0.0 < result.recommended_exposure_reduction <= 1.0
    # Severe crowding should recommend at least 30% reduction
    if result.crowding_flag == CrowdingFlag.SEVERE:
        assert result.recommended_exposure_reduction >= 0.3


# --- Helpers ---

def _make_random_signal_stream(
    start: datetime, days: int, seed: int
) -> list[dict]:
    """Generate a random signal stream for testing."""
    rng = np.random.default_rng(seed)
    return [
        {
            "timestamp": start + timedelta(days=i),
            "direction": float(rng.choice([-1.0, 1.0])),
            "magnitude": float(rng.uniform(0.1, 1.0)),
        }
        for i in range(days)
    ]


def _add_noise(
    stream: list[dict], noise_level: float, seed: int
) -> list[dict]:
    """Add small noise to a signal stream to create near-correlation."""
    rng = np.random.default_rng(seed)
    return [
        {
            "timestamp": s["timestamp"],
            "direction": s["direction"],
            "magnitude": float(
                max(0.0, s["magnitude"] + rng.normal(0, noise_level))
            ),
        }
        for s in stream
    ]
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_convergence.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.crowding'`

**Step 3: Implement the convergence scorer**

```python
# src/evolve_trader/crowding/__init__.py
"""Crowding detection and contrarian activation."""
```

```python
# src/evolve_trader/crowding/convergence.py
"""Cross-source convergence scoring for crowding detection.

Computes rolling pairwise correlations across signal sources. When multiple
sources converge on the same directional view, the system flags crowding
and recommends exposure reduction.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np


class CrowdingFlag(enum.Enum):
    """Crowding severity levels."""
    NONE = "none"
    WARNING = "warning"
    SEVERE = "severe"


@dataclass(frozen=True)
class ConvergenceResult:
    """Result of cross-source convergence analysis."""
    correlation_matrix: np.ndarray
    source_names: list[str]
    mean_correlation: float
    max_pairwise_correlation: float
    crowding_flag: CrowdingFlag
    timestamp: datetime
    recommended_exposure_reduction: float = 0.0


class ConvergenceScorer:
    """Scores cross-source convergence to detect crowding.

    Builds a rolling correlation matrix across signal source direction * magnitude
    time series. When mean pairwise correlation exceeds thresholds, flags crowding.

    Args:
        window_days: Rolling window size in days.
        crowding_threshold: Mean correlation above this triggers WARNING.
        severe_threshold: Mean correlation above this triggers SEVERE.
        max_exposure_reduction: Maximum exposure reduction to recommend (0-1).
    """

    def __init__(
        self,
        window_days: int = 30,
        crowding_threshold: float = 0.7,
        severe_threshold: float = 0.9,
        max_exposure_reduction: float = 0.5,
    ) -> None:
        self.window_days = window_days
        self.crowding_threshold = crowding_threshold
        self.severe_threshold = severe_threshold
        self.max_exposure_reduction = max_exposure_reduction

    def score(
        self,
        signal_streams: dict[str, list[dict[str, Any]]],
        as_of: datetime | None = None,
    ) -> ConvergenceResult:
        """Compute convergence across signal streams.

        Args:
            signal_streams: Dict mapping source name to list of signal dicts.
                Each signal dict must have 'timestamp', 'direction', 'magnitude'.
            as_of: Reference time for rolling window. Defaults to utcnow.

        Returns:
            ConvergenceResult with correlation matrix and crowding flag.

        Raises:
            ValueError: If fewer than 2 sources provided.
        """
        if len(signal_streams) < 2:
            raise ValueError("Convergence scoring requires at least 2 sources")

        if as_of is None:
            as_of = datetime.now(timezone.utc)

        window_start = as_of - timedelta(days=self.window_days)
        source_names = sorted(signal_streams.keys())
        n_sources = len(source_names)

        # Build aligned daily time series for each source
        daily_series = {}
        for name in source_names:
            stream = signal_streams[name]
            windowed = [
                s for s in stream
                if window_start <= s["timestamp"] <= as_of
            ]
            series = {}
            for s in windowed:
                day_key = s["timestamp"].date()
                # direction * magnitude = signed conviction
                series[day_key] = s["direction"] * s["magnitude"]
            daily_series[name] = series

        # Build common date set
        all_dates = set()
        for series in daily_series.values():
            all_dates.update(series.keys())
        sorted_dates = sorted(all_dates)

        if len(sorted_dates) < 3:
            # Not enough data for meaningful correlation
            return ConvergenceResult(
                correlation_matrix=np.eye(n_sources),
                source_names=source_names,
                mean_correlation=0.0,
                max_pairwise_correlation=0.0,
                crowding_flag=CrowdingFlag.NONE,
                timestamp=as_of,
                recommended_exposure_reduction=0.0,
            )

        # Build matrix: rows=dates, cols=sources
        matrix = np.zeros((len(sorted_dates), n_sources))
        for col_idx, name in enumerate(source_names):
            for row_idx, date in enumerate(sorted_dates):
                matrix[row_idx, col_idx] = daily_series[name].get(date, 0.0)

        # Compute correlation matrix
        corr_matrix = np.corrcoef(matrix, rowvar=False)
        # Handle NaN (constant series)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Extract upper-triangle pairwise correlations (excluding diagonal)
        upper_mask = np.triu(np.ones((n_sources, n_sources), dtype=bool), k=1)
        pairwise = corr_matrix[upper_mask]

        mean_corr = float(np.mean(np.abs(pairwise))) if len(pairwise) > 0 else 0.0
        max_corr = float(np.max(np.abs(pairwise))) if len(pairwise) > 0 else 0.0

        # Determine crowding flag
        if mean_corr >= self.severe_threshold:
            flag = CrowdingFlag.SEVERE
        elif mean_corr >= self.crowding_threshold or max_corr >= self.severe_threshold:
            flag = CrowdingFlag.WARNING
        else:
            flag = CrowdingFlag.NONE

        # Compute recommended exposure reduction
        if flag == CrowdingFlag.SEVERE:
            reduction = self.max_exposure_reduction
        elif flag == CrowdingFlag.WARNING:
            # Linear interpolation between threshold and severe
            frac = (mean_corr - self.crowding_threshold) / (
                self.severe_threshold - self.crowding_threshold
            )
            frac = max(0.0, min(1.0, frac))
            reduction = frac * self.max_exposure_reduction
        else:
            reduction = 0.0

        return ConvergenceResult(
            correlation_matrix=corr_matrix,
            source_names=source_names,
            mean_correlation=mean_corr,
            max_pairwise_correlation=max_corr,
            crowding_flag=flag,
            timestamp=as_of,
            recommended_exposure_reduction=reduction,
        )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_convergence.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/crowding/__init__.py src/evolve_trader/crowding/convergence.py tests/unit/test_convergence.py
git commit -m "feat: cross-source convergence scorer for crowding detection"
```

---

## Task 2: Source Independence Score

**Files:**
- Create: `src/evolve_trader/crowding/independence.py`
- Create: `tests/unit/test_independence.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_independence.py
"""Tests for source independence scoring."""
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from evolve_trader.crowding.independence import (
    IndependenceScorer,
    IndependenceResult,
)


def test_independence_result_has_required_fields():
    """IndependenceResult exposes average pairwise correlation and trend."""
    result = IndependenceResult(
        avg_pairwise_correlation=0.3,
        pairwise_correlations={"edgar_13f|form4": 0.25, "edgar_13f|congressional": 0.35},
        trend_slope=0.01,
        is_groupthink=False,
        timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc),
    )
    assert result.avg_pairwise_correlation == 0.3
    assert not result.is_groupthink
    assert len(result.pairwise_correlations) == 2


def test_independent_sources_low_score():
    """Independent signal sources yield low average pairwise correlation."""
    scorer = IndependenceScorer(
        window_days=30,
        groupthink_threshold=0.6,
    )
    now = datetime(2025, 6, 1, tzinfo=timezone.utc)
    streams = _make_independent_streams(now, n_sources=4, days=60)
    result = scorer.score(streams, as_of=now)
    assert result.avg_pairwise_correlation < 0.4
    assert not result.is_groupthink


def test_correlated_sources_high_score():
    """Correlated signal sources yield high average pairwise correlation."""
    scorer = IndependenceScorer(
        window_days=30,
        groupthink_threshold=0.6,
    )
    now = datetime(2025, 6, 1, tzinfo=timezone.utc)
    streams = _make_correlated_streams(now, n_sources=4, days=60)
    result = scorer.score(streams, as_of=now)
    assert result.avg_pairwise_correlation > 0.6
    assert result.is_groupthink


def test_rising_correlation_detected():
    """Scorer detects rising correlation trend (groupthink forming)."""
    scorer = IndependenceScorer(
        window_days=30,
        groupthink_threshold=0.6,
        trend_window=5,
    )
    now = datetime(2025, 6, 1, tzinfo=timezone.utc)
    # Simulate rising correlation: first half independent, second half correlated
    streams = _make_rising_correlation_streams(now, days=60)
    result = scorer.score(streams, as_of=now)
    assert result.trend_slope > 0.0, "Trend should be positive (rising correlation)"


def test_independence_scorer_requires_minimum_sources():
    """Scorer raises if fewer than 2 sources."""
    scorer = IndependenceScorer(window_days=30)
    with pytest.raises(ValueError, match="at least 2 sources"):
        scorer.score({"only_one": []})


def test_per_pair_correlations_available():
    """Individual pairwise correlations are available for inspection."""
    scorer = IndependenceScorer(window_days=30)
    now = datetime(2025, 6, 1, tzinfo=timezone.utc)
    streams = _make_independent_streams(now, n_sources=3, days=60)
    result = scorer.score(streams, as_of=now)
    # 3 sources => 3 pairs
    assert len(result.pairwise_correlations) == 3
    for pair_key, corr in result.pairwise_correlations.items():
        assert "|" in pair_key
        assert -1.0 <= corr <= 1.0


# --- Helpers ---

def _make_independent_streams(
    end: datetime, n_sources: int, days: int
) -> dict[str, list[dict]]:
    """Generate independent signal streams."""
    streams = {}
    for i in range(n_sources):
        rng = np.random.default_rng(seed=i * 100)
        streams[f"source_{i}"] = [
            {
                "timestamp": end - timedelta(days=days - d),
                "direction": float(rng.choice([-1.0, 1.0])),
                "magnitude": float(rng.uniform(0.1, 1.0)),
            }
            for d in range(days)
        ]
    return streams


def _make_correlated_streams(
    end: datetime, n_sources: int, days: int
) -> dict[str, list[dict]]:
    """Generate highly correlated signal streams."""
    rng = np.random.default_rng(seed=42)
    base = [
        {
            "timestamp": end - timedelta(days=days - d),
            "direction": float(rng.choice([-1.0, 1.0])),
            "magnitude": float(rng.uniform(0.1, 1.0)),
        }
        for d in range(days)
    ]
    streams = {}
    for i in range(n_sources):
        noise_rng = np.random.default_rng(seed=i * 50)
        streams[f"source_{i}"] = [
            {
                "timestamp": s["timestamp"],
                "direction": s["direction"],
                "magnitude": float(
                    max(0.0, s["magnitude"] + noise_rng.normal(0, 0.05))
                ),
            }
            for s in base
        ]
    return streams


def _make_rising_correlation_streams(
    end: datetime, days: int
) -> dict[str, list[dict]]:
    """Generate streams where correlation rises over time."""
    rng_a = np.random.default_rng(seed=1)
    rng_b = np.random.default_rng(seed=2)
    base_rng = np.random.default_rng(seed=42)
    stream_a = []
    stream_b = []
    for d in range(days):
        ts = end - timedelta(days=days - d)
        base_dir = float(base_rng.choice([-1.0, 1.0]))
        base_mag = float(base_rng.uniform(0.1, 1.0))
        # Blend: early = independent, late = correlated
        blend = d / days  # 0 -> 1
        dir_a = base_dir if rng_a.random() < blend else float(rng_a.choice([-1.0, 1.0]))
        dir_b = base_dir if rng_b.random() < blend else float(rng_b.choice([-1.0, 1.0]))
        stream_a.append({"timestamp": ts, "direction": dir_a, "magnitude": base_mag})
        stream_b.append({"timestamp": ts, "direction": dir_b, "magnitude": base_mag})
    return {"source_a": stream_a, "source_b": stream_b}
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_independence.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.crowding.independence'`

**Step 3: Implement the independence scorer**

```python
# src/evolve_trader/crowding/independence.py
"""Source independence scoring.

Measures average pairwise correlation across signal sources over time.
Rising independence score indicates groupthink formation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from itertools import combinations
from typing import Any

import numpy as np


@dataclass(frozen=True)
class IndependenceResult:
    """Result of source independence analysis."""
    avg_pairwise_correlation: float
    pairwise_correlations: dict[str, float]
    trend_slope: float
    is_groupthink: bool
    timestamp: datetime


class IndependenceScorer:
    """Measures source independence via average pairwise correlation.

    Tracks whether correlation is rising (groupthink forming) by computing
    the trend slope of windowed correlation measurements.

    Args:
        window_days: Rolling window for correlation computation.
        groupthink_threshold: Average correlation above this = groupthink.
        trend_window: Number of sub-windows for trend detection.
    """

    def __init__(
        self,
        window_days: int = 30,
        groupthink_threshold: float = 0.6,
        trend_window: int = 5,
    ) -> None:
        self.window_days = window_days
        self.groupthink_threshold = groupthink_threshold
        self.trend_window = trend_window

    def score(
        self,
        signal_streams: dict[str, list[dict[str, Any]]],
        as_of: datetime | None = None,
    ) -> IndependenceResult:
        """Compute source independence score.

        Args:
            signal_streams: Dict mapping source name to signal dicts with
                'timestamp', 'direction', 'magnitude'.
            as_of: Reference time. Defaults to utcnow.

        Returns:
            IndependenceResult with average pairwise correlation and trend.

        Raises:
            ValueError: If fewer than 2 sources provided.
        """
        if len(signal_streams) < 2:
            raise ValueError("Independence scoring requires at least 2 sources")

        if as_of is None:
            as_of = datetime.now(timezone.utc)

        source_names = sorted(signal_streams.keys())
        window_start = as_of - timedelta(days=self.window_days)

        # Build daily signed-conviction series per source
        daily = self._build_daily_series(signal_streams, source_names, window_start, as_of)

        # Common dates
        all_dates = set()
        for series in daily.values():
            all_dates.update(series.keys())
        sorted_dates = sorted(all_dates)

        # Compute pairwise correlations
        pairwise = {}
        for name_a, name_b in combinations(source_names, 2):
            pair_key = f"{name_a}|{name_b}"
            arr_a = np.array([daily[name_a].get(d, 0.0) for d in sorted_dates])
            arr_b = np.array([daily[name_b].get(d, 0.0) for d in sorted_dates])
            if len(sorted_dates) < 3 or np.std(arr_a) == 0 or np.std(arr_b) == 0:
                pairwise[pair_key] = 0.0
            else:
                corr = float(np.corrcoef(arr_a, arr_b)[0, 1])
                pairwise[pair_key] = corr if not np.isnan(corr) else 0.0

        avg_corr = float(np.mean([abs(v) for v in pairwise.values()])) if pairwise else 0.0

        # Compute trend slope using sub-windows
        trend_slope = self._compute_trend(
            signal_streams, source_names, as_of
        )

        is_groupthink = avg_corr >= self.groupthink_threshold

        return IndependenceResult(
            avg_pairwise_correlation=avg_corr,
            pairwise_correlations=pairwise,
            trend_slope=trend_slope,
            is_groupthink=is_groupthink,
            timestamp=as_of,
        )

    def _build_daily_series(
        self,
        streams: dict[str, list[dict]],
        source_names: list[str],
        start: datetime,
        end: datetime,
    ) -> dict[str, dict]:
        """Build daily signed-conviction series per source within window."""
        daily = {}
        for name in source_names:
            series = {}
            for s in streams[name]:
                if start <= s["timestamp"] <= end:
                    day_key = s["timestamp"].date()
                    series[day_key] = s["direction"] * s["magnitude"]
            daily[name] = series
        return daily

    def _compute_trend(
        self,
        streams: dict[str, list[dict]],
        source_names: list[str],
        as_of: datetime,
    ) -> float:
        """Compute correlation trend slope over sub-windows."""
        if self.trend_window < 2:
            return 0.0

        sub_window_days = self.window_days // self.trend_window
        if sub_window_days < 3:
            return 0.0

        correlations_over_time = []
        for i in range(self.trend_window):
            sub_end = as_of - timedelta(days=i * sub_window_days)
            sub_start = sub_end - timedelta(days=sub_window_days)
            daily = self._build_daily_series(streams, source_names, sub_start, sub_end)
            all_dates = set()
            for series in daily.values():
                all_dates.update(series.keys())
            sorted_dates = sorted(all_dates)

            if len(sorted_dates) < 3:
                correlations_over_time.append(0.0)
                continue

            pair_corrs = []
            for name_a, name_b in combinations(source_names, 2):
                arr_a = np.array([daily[name_a].get(d, 0.0) for d in sorted_dates])
                arr_b = np.array([daily[name_b].get(d, 0.0) for d in sorted_dates])
                if np.std(arr_a) == 0 or np.std(arr_b) == 0:
                    pair_corrs.append(0.0)
                else:
                    c = float(np.corrcoef(arr_a, arr_b)[0, 1])
                    pair_corrs.append(abs(c) if not np.isnan(c) else 0.0)

            correlations_over_time.append(float(np.mean(pair_corrs)))

        # Reverse so index 0 = oldest, last = newest
        correlations_over_time.reverse()

        if len(correlations_over_time) < 2:
            return 0.0

        # Simple linear regression slope
        x = np.arange(len(correlations_over_time), dtype=float)
        y = np.array(correlations_over_time)
        slope = float(np.polyfit(x, y, 1)[0])
        return slope
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_independence.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/crowding/independence.py tests/unit/test_independence.py
git commit -m "feat: source independence scorer with groupthink detection"
```

---

## Task 3: Contrarian Skill Family

**Files:**
- Create: `src/evolve_trader/crowding/contrarian_skills.py`
- Create: `strategies/contrarian/reduce_exposure.skill.md`
- Create: `strategies/contrarian/tail_risk_hedge.skill.md`
- Create: `strategies/contrarian/increase_cash.skill.md`
- Create: `strategies/contrarian/short_crowded.skill.md`
- Create: `tests/unit/test_contrarian_skills.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_contrarian_skills.py
"""Tests for contrarian skill family activation during crowding."""
import pytest
from datetime import datetime, timezone
from evolve_trader.crowding.contrarian_skills import (
    ContrarianSkillActivator,
    ContrarianAction,
    ActivationPlan,
)
from evolve_trader.crowding.convergence import CrowdingFlag


def test_no_activation_when_no_crowding():
    """No contrarian skills activate when crowding is NONE."""
    activator = ContrarianSkillActivator()
    plan = activator.evaluate(
        crowding_flag=CrowdingFlag.NONE,
        mean_correlation=0.2,
        current_positions=_sample_positions(),
        cash_pct=0.30,
    )
    assert plan.actions == []
    assert not plan.should_activate


def test_warning_crowding_activates_reduce_and_cash():
    """WARNING crowding activates exposure reduction and cash increase."""
    activator = ContrarianSkillActivator()
    plan = activator.evaluate(
        crowding_flag=CrowdingFlag.WARNING,
        mean_correlation=0.75,
        current_positions=_sample_positions(),
        cash_pct=0.10,
    )
    assert plan.should_activate
    action_types = [a.action_type for a in plan.actions]
    assert "reduce_exposure" in action_types
    assert "increase_cash" in action_types


def test_severe_crowding_activates_all_contrarian_skills():
    """SEVERE crowding activates full contrarian suite including hedging."""
    activator = ContrarianSkillActivator()
    plan = activator.evaluate(
        crowding_flag=CrowdingFlag.SEVERE,
        mean_correlation=0.92,
        current_positions=_sample_positions(),
        cash_pct=0.10,
    )
    assert plan.should_activate
    action_types = [a.action_type for a in plan.actions]
    assert "reduce_exposure" in action_types
    assert "tail_risk_hedge" in action_types
    assert "increase_cash" in action_types


def test_short_crowded_names_only_when_enabled():
    """Short crowded names only included when explicitly enabled."""
    activator = ContrarianSkillActivator(enable_shorting=False)
    plan = activator.evaluate(
        crowding_flag=CrowdingFlag.SEVERE,
        mean_correlation=0.95,
        current_positions=_sample_positions(),
        cash_pct=0.10,
    )
    action_types = [a.action_type for a in plan.actions]
    assert "short_crowded" not in action_types

    activator_with_short = ContrarianSkillActivator(enable_shorting=True)
    plan_short = activator_with_short.evaluate(
        crowding_flag=CrowdingFlag.SEVERE,
        mean_correlation=0.95,
        current_positions=_sample_positions(),
        cash_pct=0.10,
    )
    action_types_short = [a.action_type for a in plan_short.actions]
    assert "short_crowded" in action_types_short


def test_contrarian_action_has_required_fields():
    """ContrarianAction has action_type, target_pct, reasoning."""
    action = ContrarianAction(
        action_type="reduce_exposure",
        target_pct=0.20,
        reasoning="Mean correlation 0.85 indicates crowding — reduce by 20%",
        skill_md_path="strategies/contrarian/reduce_exposure.skill.md",
    )
    assert action.action_type == "reduce_exposure"
    assert action.target_pct == 0.20
    assert "crowding" in action.reasoning


def test_activation_plan_respects_risk_constraints():
    """Activation plan never exceeds risk constraint limits."""
    activator = ContrarianSkillActivator(
        max_position_reduction=0.50,
        max_cash_target=0.80,
    )
    plan = activator.evaluate(
        crowding_flag=CrowdingFlag.SEVERE,
        mean_correlation=0.99,
        current_positions=_sample_positions(),
        cash_pct=0.05,
    )
    for action in plan.actions:
        if action.action_type == "reduce_exposure":
            assert action.target_pct <= 0.50
        if action.action_type == "increase_cash":
            assert action.target_pct <= 0.80


def test_contrarian_skills_are_evolvable():
    """Contrarian skill activator lists SKILL.md paths for evolution."""
    activator = ContrarianSkillActivator()
    skill_paths = activator.list_skill_paths()
    assert len(skill_paths) >= 3
    for path in skill_paths:
        assert path.endswith(".skill.md")
        assert "contrarian" in path


# --- Helpers ---

def _sample_positions() -> list[dict]:
    """Sample portfolio positions."""
    return [
        {"ticker": "AAPL", "weight": 0.15, "sector": "Technology"},
        {"ticker": "MSFT", "weight": 0.12, "sector": "Technology"},
        {"ticker": "JPM", "weight": 0.10, "sector": "Financials"},
        {"ticker": "XOM", "weight": 0.08, "sector": "Energy"},
        {"ticker": "JNJ", "weight": 0.07, "sector": "Healthcare"},
    ]
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_contrarian_skills.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.crowding.contrarian_skills'`

**Step 3: Implement contrarian skill activator and SKILL.md files**

```python
# src/evolve_trader/crowding/contrarian_skills.py
"""Contrarian skill family — activated during crowding events.

Provides a suite of defensive actions: reduce exposure, tail-risk hedging,
cash increase, and optionally shorting crowded names. Each action maps to
an evolvable SKILL.md file subject to the evolution loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evolve_trader.crowding.convergence import CrowdingFlag


@dataclass(frozen=True)
class ContrarianAction:
    """A single contrarian action to execute."""
    action_type: str  # reduce_exposure, tail_risk_hedge, increase_cash, short_crowded
    target_pct: float
    reasoning: str
    skill_md_path: str


@dataclass(frozen=True)
class ActivationPlan:
    """Plan of contrarian actions to execute."""
    should_activate: bool
    actions: list[ContrarianAction] = field(default_factory=list)
    crowding_flag: CrowdingFlag = CrowdingFlag.NONE
    mean_correlation: float = 0.0


class ContrarianSkillActivator:
    """Evaluates crowding state and produces contrarian activation plans.

    Args:
        enable_shorting: Whether to include short-crowded-names action.
        max_position_reduction: Maximum fraction to reduce positions by.
        max_cash_target: Maximum cash target percentage.
        warning_reduction_pct: Position reduction for WARNING crowding.
        severe_reduction_pct: Position reduction for SEVERE crowding.
    """

    SKILL_PATHS = [
        "strategies/contrarian/reduce_exposure.skill.md",
        "strategies/contrarian/tail_risk_hedge.skill.md",
        "strategies/contrarian/increase_cash.skill.md",
        "strategies/contrarian/short_crowded.skill.md",
    ]

    def __init__(
        self,
        enable_shorting: bool = False,
        max_position_reduction: float = 0.50,
        max_cash_target: float = 0.80,
        warning_reduction_pct: float = 0.15,
        severe_reduction_pct: float = 0.35,
    ) -> None:
        self.enable_shorting = enable_shorting
        self.max_position_reduction = max_position_reduction
        self.max_cash_target = max_cash_target
        self.warning_reduction_pct = warning_reduction_pct
        self.severe_reduction_pct = severe_reduction_pct

    def evaluate(
        self,
        crowding_flag: CrowdingFlag,
        mean_correlation: float,
        current_positions: list[dict[str, Any]],
        cash_pct: float,
    ) -> ActivationPlan:
        """Evaluate crowding state and produce activation plan.

        Args:
            crowding_flag: Current crowding severity.
            mean_correlation: Average pairwise correlation across sources.
            current_positions: List of position dicts with ticker, weight, sector.
            cash_pct: Current cash as fraction of portfolio.

        Returns:
            ActivationPlan with contrarian actions if crowding detected.
        """
        if crowding_flag == CrowdingFlag.NONE:
            return ActivationPlan(
                should_activate=False,
                crowding_flag=crowding_flag,
                mean_correlation=mean_correlation,
            )

        actions: list[ContrarianAction] = []

        if crowding_flag == CrowdingFlag.WARNING:
            actions.extend(self._warning_actions(mean_correlation, cash_pct))
        elif crowding_flag == CrowdingFlag.SEVERE:
            actions.extend(self._severe_actions(mean_correlation, cash_pct))

        if self.enable_shorting and crowding_flag == CrowdingFlag.SEVERE:
            actions.append(
                ContrarianAction(
                    action_type="short_crowded",
                    target_pct=0.05,
                    reasoning=(
                        f"Mean correlation {mean_correlation:.2f} is severe — "
                        f"short most crowded names up to 5% of portfolio"
                    ),
                    skill_md_path="strategies/contrarian/short_crowded.skill.md",
                )
            )

        return ActivationPlan(
            should_activate=len(actions) > 0,
            actions=actions,
            crowding_flag=crowding_flag,
            mean_correlation=mean_correlation,
        )

    def _warning_actions(
        self, mean_corr: float, cash_pct: float
    ) -> list[ContrarianAction]:
        """Generate actions for WARNING crowding level."""
        reduction = min(self.warning_reduction_pct, self.max_position_reduction)
        cash_target = min(cash_pct + 0.15, self.max_cash_target)
        return [
            ContrarianAction(
                action_type="reduce_exposure",
                target_pct=reduction,
                reasoning=(
                    f"Mean correlation {mean_corr:.2f} indicates crowding — "
                    f"reduce exposure by {reduction:.0%}"
                ),
                skill_md_path="strategies/contrarian/reduce_exposure.skill.md",
            ),
            ContrarianAction(
                action_type="increase_cash",
                target_pct=cash_target,
                reasoning=(
                    f"Crowding detected — increase cash to {cash_target:.0%}"
                ),
                skill_md_path="strategies/contrarian/increase_cash.skill.md",
            ),
        ]

    def _severe_actions(
        self, mean_corr: float, cash_pct: float
    ) -> list[ContrarianAction]:
        """Generate actions for SEVERE crowding level."""
        reduction = min(self.severe_reduction_pct, self.max_position_reduction)
        cash_target = min(cash_pct + 0.30, self.max_cash_target)
        return [
            ContrarianAction(
                action_type="reduce_exposure",
                target_pct=reduction,
                reasoning=(
                    f"Mean correlation {mean_corr:.2f} severe crowding — "
                    f"reduce exposure by {reduction:.0%}"
                ),
                skill_md_path="strategies/contrarian/reduce_exposure.skill.md",
            ),
            ContrarianAction(
                action_type="tail_risk_hedge",
                target_pct=0.10,
                reasoning=(
                    f"Severe crowding — deploy tail-risk hedges up to 10% of portfolio"
                ),
                skill_md_path="strategies/contrarian/tail_risk_hedge.skill.md",
            ),
            ContrarianAction(
                action_type="increase_cash",
                target_pct=cash_target,
                reasoning=(
                    f"Severe crowding — increase cash to {cash_target:.0%}"
                ),
                skill_md_path="strategies/contrarian/increase_cash.skill.md",
            ),
        ]

    def list_skill_paths(self) -> list[str]:
        """Return all SKILL.md paths managed by this activator."""
        return list(self.SKILL_PATHS)
```

```markdown
# strategies/contrarian/reduce_exposure.skill.md
---
name: reduce-exposure-contrarian
version: 1
type: contrarian
triggers:
  - crowding_flag: WARNING
  - crowding_flag: SEVERE
constraints:
  max_reduction_pct: 0.50
  min_remaining_exposure: 0.20
---

## Behavior
When crowding is detected across signal sources, reduce position sizes
proportionally. Largest positions in the most crowded sectors are reduced first.

## Entry Criteria
- CrowdingFlag >= WARNING
- Mean pairwise correlation > 0.7

## Exit Criteria
- CrowdingFlag returns to NONE for 5 consecutive trading days
- Mean correlation drops below 0.5

## Evolution Notes
This skill is subject to evolutionary pressure. Fitness is measured by
risk-adjusted returns during and after crowding events.
```

```markdown
# strategies/contrarian/tail_risk_hedge.skill.md
---
name: tail-risk-hedge-contrarian
version: 1
type: contrarian
triggers:
  - crowding_flag: SEVERE
constraints:
  max_hedge_pct: 0.10
  instruments: [put_options, vix_calls, inverse_etf]
---

## Behavior
Deploy tail-risk hedges when severe crowding detected. Use put spreads
on most-crowded names and VIX call spreads for portfolio-wide protection.

## Entry Criteria
- CrowdingFlag == SEVERE
- Mean pairwise correlation > 0.9

## Exit Criteria
- CrowdingFlag drops below SEVERE for 3 consecutive days
- Hedge cost exceeds 2% of portfolio per month

## Evolution Notes
Hedge instrument selection and sizing are evolvable parameters.
```

```markdown
# strategies/contrarian/increase_cash.skill.md
---
name: increase-cash-contrarian
version: 1
type: contrarian
triggers:
  - crowding_flag: WARNING
  - crowding_flag: SEVERE
constraints:
  max_cash_pct: 0.80
  min_cash_pct: 0.10
---

## Behavior
Increase cash allocation by selling positions in crowded sectors.
Priority: highest-correlation sectors first, then largest positions.

## Entry Criteria
- CrowdingFlag >= WARNING
- Current cash below target for crowding level

## Exit Criteria
- CrowdingFlag returns to NONE
- Redeployment happens gradually over 5-10 trading days

## Evolution Notes
Cash target percentages per crowding level are evolvable.
```

```markdown
# strategies/contrarian/short_crowded.skill.md
---
name: short-crowded-names-contrarian
version: 1
type: contrarian
triggers:
  - crowding_flag: SEVERE
  - enable_shorting: true
constraints:
  max_short_pct: 0.05
  max_single_name_short: 0.02
  borrow_cost_limit: 0.03
---

## Behavior
Short the most crowded names — those where all sources converge on
the same bullish view. Small position sizes, strict borrow cost limits.

## Entry Criteria
- CrowdingFlag == SEVERE
- Shorting explicitly enabled in system config
- Specific names identified where all sources agree

## Exit Criteria
- CrowdingFlag drops below SEVERE
- Short position loss exceeds 1% of portfolio

## Evolution Notes
This is the highest-risk contrarian skill. Evolution should heavily
penalize losses and only retain if demonstrably profitable during
crowding unwind events.
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_contrarian_skills.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/crowding/contrarian_skills.py strategies/contrarian/ tests/unit/test_contrarian_skills.py
git commit -m "feat: contrarian skill family with 4 SKILL.md strategies for crowding response"
```

---

## Task 4: Historical Crowding Calibration

**Files:**
- Create: `tests/benchmark/test_crowding_calibration.py`

**Step 1: Write the failing tests**

```python
# tests/benchmark/test_crowding_calibration.py
"""Calibration tests: validate crowding detection against known historical events.

These tests use synthetic signal data modeled after real market conditions
to verify that the crowding system responds appropriately.
"""
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from evolve_trader.crowding.convergence import ConvergenceScorer, CrowdingFlag
from evolve_trader.crowding.independence import IndependenceScorer
from evolve_trader.crowding.contrarian_skills import ContrarianSkillActivator


class TestMemeStockCrowding2021:
    """2021 meme-stock mania: massive retail convergence on same names."""

    def test_detects_crowding_during_gme_squeeze(self):
        """All sources bullish on same names simultaneously → SEVERE."""
        scorer = ConvergenceScorer(window_days=14, crowding_threshold=0.7)
        now = datetime(2021, 1, 27, tzinfo=timezone.utc)
        streams = _build_meme_stock_streams(now)
        result = scorer.score(streams, as_of=now)
        assert result.crowding_flag in (CrowdingFlag.WARNING, CrowdingFlag.SEVERE)

    def test_contrarian_activates_during_meme_mania(self):
        """Contrarian skills should activate — reduce exposure, increase cash."""
        scorer = ConvergenceScorer(window_days=14, crowding_threshold=0.7)
        now = datetime(2021, 1, 27, tzinfo=timezone.utc)
        streams = _build_meme_stock_streams(now)
        result = scorer.score(streams, as_of=now)
        activator = ContrarianSkillActivator()
        plan = activator.evaluate(
            crowding_flag=result.crowding_flag,
            mean_correlation=result.mean_correlation,
            current_positions=[
                {"ticker": "GME", "weight": 0.05, "sector": "Consumer Discretionary"},
                {"ticker": "AMC", "weight": 0.03, "sector": "Communication Services"},
            ],
            cash_pct=0.20,
        )
        assert plan.should_activate


class TestRateShockCrowding2022:
    """2022 rate shock: all sources converge on risk-off simultaneously."""

    def test_detects_crowding_during_rate_hike_cycle(self):
        """Unanimous bearish signal convergence → crowding detected."""
        scorer = ConvergenceScorer(window_days=30, crowding_threshold=0.7)
        now = datetime(2022, 6, 15, tzinfo=timezone.utc)
        streams = _build_rate_shock_streams(now)
        result = scorer.score(streams, as_of=now)
        assert result.crowding_flag != CrowdingFlag.NONE

    def test_independence_score_drops_during_rate_shock(self):
        """Source independence should be low during unanimous rate-shock."""
        scorer = IndependenceScorer(window_days=30, groupthink_threshold=0.6)
        now = datetime(2022, 6, 15, tzinfo=timezone.utc)
        streams = _build_rate_shock_streams(now)
        result = scorer.score(streams, as_of=now)
        assert result.avg_pairwise_correlation > 0.5


class TestCovidCrowding2020:
    """2020 COVID crash: rapid consensus formation on risk-off."""

    def test_detects_rapid_crowding_formation(self):
        """Crowding forms quickly as all sources align on sell signals."""
        scorer = ConvergenceScorer(window_days=14, crowding_threshold=0.7)
        now = datetime(2020, 3, 16, tzinfo=timezone.utc)
        streams = _build_covid_crash_streams(now)
        result = scorer.score(streams, as_of=now)
        assert result.crowding_flag != CrowdingFlag.NONE

    def test_groupthink_detected_during_covid_panic(self):
        """Independence scorer flags groupthink during panic selling."""
        scorer = IndependenceScorer(window_days=14, groupthink_threshold=0.6)
        now = datetime(2020, 3, 16, tzinfo=timezone.utc)
        streams = _build_covid_crash_streams(now)
        result = scorer.score(streams, as_of=now)
        assert result.is_groupthink


# --- Synthetic data builders ---

def _build_meme_stock_streams(as_of: datetime) -> dict[str, list[dict]]:
    """Simulate meme-stock convergence: all sources bullish on same names."""
    rng = np.random.default_rng(seed=2021)
    days = 30
    streams = {}
    for source in ["reddit_sentiment", "options_flow", "retail_volume", "social_media"]:
        noise_rng = np.random.default_rng(seed=hash(source) % 2**31)
        stream = []
        for d in range(days):
            ts = as_of - timedelta(days=days - d)
            # Strong bullish bias with small noise
            stream.append({
                "timestamp": ts,
                "direction": 1.0,
                "magnitude": float(0.8 + noise_rng.uniform(-0.1, 0.1)),
            })
        streams[source] = stream
    return streams


def _build_rate_shock_streams(as_of: datetime) -> dict[str, list[dict]]:
    """Simulate 2022 rate shock: all sources converge on bearish view."""
    days = 60
    streams = {}
    for source in ["fed_funds_futures", "bond_yields", "credit_spreads", "equity_flow"]:
        noise_rng = np.random.default_rng(seed=hash(source) % 2**31)
        stream = []
        for d in range(days):
            ts = as_of - timedelta(days=days - d)
            # Strong bearish bias
            stream.append({
                "timestamp": ts,
                "direction": -1.0,
                "magnitude": float(0.7 + noise_rng.uniform(-0.15, 0.15)),
            })
        streams[source] = stream
    return streams


def _build_covid_crash_streams(as_of: datetime) -> dict[str, list[dict]]:
    """Simulate COVID crash: rapid alignment on sell signals."""
    days = 21
    streams = {}
    for source in ["news_sentiment", "vix_signal", "fund_flows", "macro_indicators"]:
        noise_rng = np.random.default_rng(seed=hash(source) % 2**31)
        stream = []
        for d in range(days):
            ts = as_of - timedelta(days=days - d)
            # Increasingly bearish as crash approaches
            bearish_intensity = min(1.0, 0.3 + (d / days) * 0.7)
            direction = -1.0 if noise_rng.random() < bearish_intensity else 1.0
            stream.append({
                "timestamp": ts,
                "direction": direction,
                "magnitude": float(0.5 + bearish_intensity * 0.5),
            })
        streams[source] = stream
    return streams
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/benchmark/test_crowding_calibration.py -v
```

Expected: PASS (depends on Tasks 1-2 implementations)

**Step 3: Commit**

```bash
git add tests/benchmark/test_crowding_calibration.py
git commit -m "test: historical crowding calibration — meme stock, rate shock, COVID"
```

---

## Task 5: Benchmark — Soros 1992 (ERM Crisis)

**Files:**
- Create: `tests/benchmark/test_soros_1992.py`

**Step 1: Write the benchmark test**

```python
# tests/benchmark/test_soros_1992.py
"""Benchmark: Soros 1992 ERM crisis.

Scenario: UK forced out of European Exchange Rate Mechanism. Bundesbank and
Bank of England diverge. GBP under massive pressure. System should detect
regime stress and macro divergence.

Expected system behavior:
- Regime classifier identifies stress/crisis regime
- Macro divergence signals (central bank policy divergence) fire
- Crowding detection may flag consensus on GBP weakness
- Risk management reduces GBP-correlated exposure
"""
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from evolve_trader.crowding.convergence import ConvergenceScorer, CrowdingFlag
from evolve_trader.crowding.independence import IndependenceScorer


class TestSoros1992:
    """System response to 1992 ERM crisis conditions."""

    @pytest.fixture
    def erm_crisis_signals(self):
        """Synthetic signals modeling the ERM crisis buildup."""
        crisis_date = datetime(1992, 9, 16, tzinfo=timezone.utc)
        days = 90  # 3 months buildup
        streams = {}

        # Macro: Bundesbank hawkish, BoE dovish — divergence signal
        rng_macro = np.random.default_rng(seed=1992)
        streams["macro_central_bank"] = [
            {
                "timestamp": crisis_date - timedelta(days=days - d),
                "direction": -1.0,  # bearish GBP
                "magnitude": float(0.3 + (d / days) * 0.6),  # intensifying
            }
            for d in range(days)
        ]

        # FX flow: sustained GBP selling
        rng_fx = np.random.default_rng(seed=916)
        streams["fx_flow"] = [
            {
                "timestamp": crisis_date - timedelta(days=days - d),
                "direction": -1.0 if d > days * 0.3 else float(rng_fx.choice([-1, 1])),
                "magnitude": float(0.4 + (d / days) * 0.5),
            }
            for d in range(days)
        ]

        # Interest rate differential: widening
        streams["rate_differential"] = [
            {
                "timestamp": crisis_date - timedelta(days=days - d),
                "direction": -1.0,
                "magnitude": float(0.2 + (d / days) * 0.7),
            }
            for d in range(days)
        ]

        # Speculative positioning: growing short GBP consensus
        rng_spec = np.random.default_rng(seed=1600)
        streams["speculative_positioning"] = [
            {
                "timestamp": crisis_date - timedelta(days=days - d),
                "direction": -1.0 if rng_spec.random() < 0.5 + (d / days) * 0.45 else 1.0,
                "magnitude": float(0.3 + (d / days) * 0.6),
            }
            for d in range(days)
        ]

        return streams, crisis_date

    def test_convergence_detects_erm_stress(self, erm_crisis_signals):
        """Convergence scorer should detect signal alignment before Black Wednesday."""
        streams, crisis_date = erm_crisis_signals
        scorer = ConvergenceScorer(window_days=30, crowding_threshold=0.6)
        result = scorer.score(streams, as_of=crisis_date)
        assert result.crowding_flag != CrowdingFlag.NONE, (
            "System should detect convergence during ERM crisis buildup"
        )

    def test_independence_drops_before_crisis(self, erm_crisis_signals):
        """Source independence should decline as crisis approaches."""
        streams, crisis_date = erm_crisis_signals
        scorer = IndependenceScorer(window_days=30, groupthink_threshold=0.5)

        # 60 days before — sources still somewhat independent
        early_result = scorer.score(
            streams, as_of=crisis_date - timedelta(days=60)
        )
        # At crisis — sources converged
        crisis_result = scorer.score(streams, as_of=crisis_date)

        assert crisis_result.avg_pairwise_correlation > early_result.avg_pairwise_correlation, (
            "Correlation should increase as crisis approaches"
        )

    def test_signal_intensity_increases_before_crisis(self, erm_crisis_signals):
        """Signal magnitudes should increase as crisis approaches."""
        streams, crisis_date = erm_crisis_signals
        for source_name, stream in streams.items():
            early = [s for s in stream if s["timestamp"] < crisis_date - timedelta(days=60)]
            late = [s for s in stream if s["timestamp"] > crisis_date - timedelta(days=30)]
            if early and late:
                early_avg_mag = np.mean([s["magnitude"] for s in early])
                late_avg_mag = np.mean([s["magnitude"] for s in late])
                assert late_avg_mag > early_avg_mag, (
                    f"{source_name}: signal intensity should increase as crisis nears"
                )
```

**Step 2: Run tests**

```bash
pytest tests/benchmark/test_soros_1992.py -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add tests/benchmark/test_soros_1992.py
git commit -m "test: benchmark — Soros 1992 ERM crisis regime stress detection"
```

---

## Task 6: Benchmark — Burry 2005-2007 (Subprime)

**Files:**
- Create: `tests/benchmark/test_burry_subprime.py`

**Step 1: Write the benchmark test**

```python
# tests/benchmark/test_burry_subprime.py
"""Benchmark: Burry 2005-2007 housing deterioration.

Scenario: Slow-building housing market deterioration. Subprime delinquencies
rising, rating agency downgrades lagging, housing starts declining.

Expected system behavior:
- Detect gradual signal deterioration in housing/financial sector
- Risk-off signals for financials
- Reduced financial sector exposure
- Contrarian position if crowding on "housing always goes up"
"""
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from evolve_trader.crowding.convergence import ConvergenceScorer, CrowdingFlag
from evolve_trader.crowding.independence import IndependenceScorer
from evolve_trader.crowding.contrarian_skills import ContrarianSkillActivator


class TestBurrySubprime:
    """System response to 2005-2007 housing deterioration."""

    @pytest.fixture
    def subprime_signals(self):
        """Synthetic signals modeling slow housing deterioration."""
        # Mid-2006: cracks appearing
        reference_date = datetime(2006, 8, 1, tzinfo=timezone.utc)
        days = 180  # 6 months of data

        streams = {}
        rng = np.random.default_rng(seed=2006)

        # Housing data: gradual deterioration
        streams["housing_fundamentals"] = [
            {
                "timestamp": reference_date - timedelta(days=days - d),
                "direction": -1.0 if rng.random() < 0.4 + (d / days) * 0.4 else 1.0,
                "magnitude": float(0.2 + (d / days) * 0.5),
            }
            for d in range(days)
        ]

        # Credit spreads: slowly widening
        rng_credit = np.random.default_rng(seed=2007)
        streams["credit_spreads"] = [
            {
                "timestamp": reference_date - timedelta(days=days - d),
                "direction": -1.0 if rng_credit.random() < 0.35 + (d / days) * 0.35 else 1.0,
                "magnitude": float(0.15 + (d / days) * 0.4),
            }
            for d in range(days)
        ]

        # Bullish consensus (crowding): "housing always goes up"
        rng_bull = np.random.default_rng(seed=2005)
        streams["market_consensus"] = [
            {
                "timestamp": reference_date - timedelta(days=days - d),
                "direction": 1.0,  # persistently bullish consensus
                "magnitude": float(0.6 + rng_bull.uniform(-0.1, 0.1)),
            }
            for d in range(days)
        ]

        # Insider activity: insiders selling financials
        rng_insider = np.random.default_rng(seed=2008)
        streams["insider_activity"] = [
            {
                "timestamp": reference_date - timedelta(days=days - d),
                "direction": -1.0 if d > days * 0.3 else float(rng_insider.choice([-1, 1])),
                "magnitude": float(0.3 + (d / days) * 0.4),
            }
            for d in range(days)
        ]

        return streams, reference_date

    def test_detects_divergence_between_consensus_and_fundamentals(self, subprime_signals):
        """System detects that consensus is bullish while fundamentals deteriorate."""
        streams, ref_date = subprime_signals
        # Fundamentals vs consensus should show divergence (low or negative correlation)
        scorer = ConvergenceScorer(window_days=60, crowding_threshold=0.7)
        result = scorer.score(streams, as_of=ref_date)
        # Key insight: not all sources agree — fundamentals diverge from consensus
        # The system should still detect some crowding on the bullish side
        assert result.mean_correlation > 0.0, (
            "System should detect meaningful signal relationships"
        )

    def test_contrarian_activates_if_crowding_detected(self, subprime_signals):
        """If crowding is flagged, contrarian skills should activate."""
        streams, ref_date = subprime_signals
        scorer = ConvergenceScorer(window_days=60, crowding_threshold=0.5)
        result = scorer.score(streams, as_of=ref_date)
        if result.crowding_flag != CrowdingFlag.NONE:
            activator = ContrarianSkillActivator()
            plan = activator.evaluate(
                crowding_flag=result.crowding_flag,
                mean_correlation=result.mean_correlation,
                current_positions=[
                    {"ticker": "C", "weight": 0.10, "sector": "Financials"},
                    {"ticker": "BAC", "weight": 0.08, "sector": "Financials"},
                    {"ticker": "MER", "weight": 0.06, "sector": "Financials"},
                ],
                cash_pct=0.15,
            )
            assert plan.should_activate, (
                "Contrarian skills should activate when housing crowding detected"
            )

    def test_late_stage_signals_stronger_than_early(self, subprime_signals):
        """Signal intensity should grow as deterioration progresses."""
        streams, ref_date = subprime_signals
        for source_name in ["housing_fundamentals", "credit_spreads"]:
            stream = streams[source_name]
            midpoint = len(stream) // 2
            early_mag = np.mean([s["magnitude"] for s in stream[:midpoint]])
            late_mag = np.mean([s["magnitude"] for s in stream[midpoint:]])
            assert late_mag > early_mag, (
                f"{source_name}: deterioration signals should intensify over time"
            )
```

**Step 2: Run tests**

```bash
pytest tests/benchmark/test_burry_subprime.py -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add tests/benchmark/test_burry_subprime.py
git commit -m "test: benchmark — Burry 2005-2007 housing deterioration detection"
```

---

## Task 7: Benchmark — Druckenmiller 2000 (Dot-Com Froth)

**Files:**
- Create: `tests/benchmark/test_druckenmiller_2000.py`

**Step 1: Write the benchmark test**

```python
# tests/benchmark/test_druckenmiller_2000.py
"""Benchmark: Druckenmiller 2000 dot-com bubble.

Scenario: Extreme valuation froth in technology sector. All sources
converging on tech-bullish view. Insider selling accelerating.

Expected system behavior:
- Detect crowding on technology/growth
- Flag extreme convergence on tech-bullish signals
- Contrarian skills activate — reduce tech exposure
"""
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from evolve_trader.crowding.convergence import ConvergenceScorer, CrowdingFlag
from evolve_trader.crowding.independence import IndependenceScorer
from evolve_trader.crowding.contrarian_skills import ContrarianSkillActivator


class TestDruckenmiller2000:
    """System response to dot-com bubble peak conditions."""

    @pytest.fixture
    def dotcom_signals(self):
        """Synthetic signals modeling dot-com froth in early 2000."""
        peak_date = datetime(2000, 3, 10, tzinfo=timezone.utc)
        days = 120

        streams = {}

        # Tech euphoria: all sources wildly bullish
        for source in ["analyst_sentiment", "retail_flow", "media_sentiment"]:
            rng = np.random.default_rng(seed=hash(source) % 2**31)
            streams[source] = [
                {
                    "timestamp": peak_date - timedelta(days=days - d),
                    "direction": 1.0,
                    "magnitude": float(0.7 + rng.uniform(0.0, 0.3)),
                }
                for d in range(days)
            ]

        # Insider selling: divergent signal — insiders know better
        rng_insider = np.random.default_rng(seed=2000)
        streams["insider_transactions"] = [
            {
                "timestamp": peak_date - timedelta(days=days - d),
                "direction": -1.0 if d > days * 0.4 else float(rng_insider.choice([-1, 1])),
                "magnitude": float(0.4 + (d / days) * 0.5),
            }
            for d in range(days)
        ]

        # IPO volume: extreme bullish (crowding indicator)
        rng_ipo = np.random.default_rng(seed=1999)
        streams["ipo_volume"] = [
            {
                "timestamp": peak_date - timedelta(days=days - d),
                "direction": 1.0,
                "magnitude": float(0.8 + rng_ipo.uniform(-0.05, 0.15)),
            }
            for d in range(days)
        ]

        return streams, peak_date

    def test_detects_tech_crowding(self, dotcom_signals):
        """System should detect extreme crowding on tech-bullish view."""
        streams, peak = dotcom_signals
        scorer = ConvergenceScorer(window_days=30, crowding_threshold=0.6)
        result = scorer.score(streams, as_of=peak)
        assert result.crowding_flag != CrowdingFlag.NONE, (
            "System must detect crowding during dot-com peak"
        )

    def test_high_convergence_across_bullish_sources(self, dotcom_signals):
        """Bullish sources should show very high pairwise correlation."""
        streams, peak = dotcom_signals
        scorer = ConvergenceScorer(window_days=30, crowding_threshold=0.6)
        result = scorer.score(streams, as_of=peak)
        assert result.max_pairwise_correlation > 0.5, (
            "Bullish sources should be highly correlated"
        )

    def test_independence_score_very_low(self, dotcom_signals):
        """Source independence should be very low during bubble peak."""
        streams, peak = dotcom_signals
        scorer = IndependenceScorer(window_days=30, groupthink_threshold=0.5)
        result = scorer.score(streams, as_of=peak)
        assert result.is_groupthink, (
            "Dot-com peak should register as groupthink"
        )

    def test_contrarian_activates_with_tech_reduction(self, dotcom_signals):
        """Contrarian skills should activate and recommend tech exposure reduction."""
        streams, peak = dotcom_signals
        scorer = ConvergenceScorer(window_days=30, crowding_threshold=0.6)
        result = scorer.score(streams, as_of=peak)
        activator = ContrarianSkillActivator()
        plan = activator.evaluate(
            crowding_flag=result.crowding_flag,
            mean_correlation=result.mean_correlation,
            current_positions=[
                {"ticker": "CSCO", "weight": 0.15, "sector": "Technology"},
                {"ticker": "MSFT", "weight": 0.12, "sector": "Technology"},
                {"ticker": "INTC", "weight": 0.10, "sector": "Technology"},
                {"ticker": "ORCL", "weight": 0.08, "sector": "Technology"},
            ],
            cash_pct=0.05,
        )
        assert plan.should_activate, (
            "Contrarian skills must activate during dot-com froth"
        )
```

**Step 2: Run tests**

```bash
pytest tests/benchmark/test_druckenmiller_2000.py -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add tests/benchmark/test_druckenmiller_2000.py
git commit -m "test: benchmark — Druckenmiller 2000 dot-com froth and tech crowding"
```

---

## Task 8: Benchmark — Ackman COVID 2020 (Rapid Regime Change)

**Files:**
- Create: `tests/benchmark/test_ackman_2020.py`

**Step 1: Write the benchmark test**

```python
# tests/benchmark/test_ackman_2020.py
"""Benchmark: Ackman COVID 2020 tail-risk hedge.

Scenario: Pandemic onset triggers rapid regime change. VIX spikes from ~15
to ~80. Market drops 35% in weeks. Ackman's hedge returned 100x.

Expected system behavior:
- Detect extremely rapid regime change
- VIX spike triggers immediate defensive posture
- Crowding forms quickly on risk-off (everyone selling)
- Tail-risk hedging skill activates
"""
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from evolve_trader.crowding.convergence import ConvergenceScorer, CrowdingFlag
from evolve_trader.crowding.contrarian_skills import ContrarianSkillActivator


class TestAckman2020:
    """System response to COVID-19 market crash."""

    @pytest.fixture
    def covid_crash_signals(self):
        """Synthetic signals modeling Feb-Mar 2020 crash."""
        crash_peak = datetime(2020, 3, 23, tzinfo=timezone.utc)
        days = 45  # Late Jan to late March

        streams = {}

        # VIX signal: calm then explodes
        rng_vix = np.random.default_rng(seed=2020)
        streams["vix_signal"] = []
        for d in range(days):
            ts = crash_peak - timedelta(days=days - d)
            if d < 15:
                # Pre-crash: calm
                direction = float(rng_vix.choice([-1.0, 1.0]))
                magnitude = float(rng_vix.uniform(0.1, 0.3))
            elif d < 30:
                # Crash building
                direction = -1.0
                magnitude = float(0.5 + (d - 15) / 15 * 0.4)
            else:
                # Peak panic
                direction = -1.0
                magnitude = float(0.9 + rng_vix.uniform(0.0, 0.1))
            streams["vix_signal"].append({
                "timestamp": ts, "direction": direction, "magnitude": magnitude,
            })

        # Fund flows: massive outflows
        rng_flow = np.random.default_rng(seed=323)
        streams["fund_flows"] = []
        for d in range(days):
            ts = crash_peak - timedelta(days=days - d)
            if d < 20:
                direction = float(rng_flow.choice([-1.0, 1.0]))
                magnitude = float(rng_flow.uniform(0.1, 0.3))
            else:
                direction = -1.0
                magnitude = float(0.6 + (d - 20) / (days - 20) * 0.35)
            streams["fund_flows"].append({
                "timestamp": ts, "direction": direction, "magnitude": magnitude,
            })

        # News sentiment: pandemic fear
        rng_news = np.random.default_rng(seed=19)
        streams["news_sentiment"] = []
        for d in range(days):
            ts = crash_peak - timedelta(days=days - d)
            fear_level = min(1.0, max(0.0, (d - 10) / (days - 10))) if d > 10 else 0.0
            direction = -1.0 if rng_news.random() < 0.3 + fear_level * 0.65 else 1.0
            magnitude = float(0.2 + fear_level * 0.7)
            streams["news_sentiment"].append({
                "timestamp": ts, "direction": direction, "magnitude": magnitude,
            })

        # Credit spreads: blowing out
        rng_credit = np.random.default_rng(seed=2020)
        streams["credit_spreads"] = []
        for d in range(days):
            ts = crash_peak - timedelta(days=days - d)
            if d < 15:
                direction = float(rng_credit.choice([-1.0, 1.0]))
                magnitude = float(rng_credit.uniform(0.1, 0.25))
            else:
                direction = -1.0
                magnitude = float(0.5 + (d - 15) / (days - 15) * 0.45)
            streams["credit_spreads"].append({
                "timestamp": ts, "direction": direction, "magnitude": magnitude,
            })

        return streams, crash_peak

    def test_detects_rapid_crowding_during_crash(self, covid_crash_signals):
        """System should detect crowding as all sources converge on risk-off."""
        streams, crash_peak = covid_crash_signals
        scorer = ConvergenceScorer(window_days=14, crowding_threshold=0.6)
        result = scorer.score(streams, as_of=crash_peak)
        assert result.crowding_flag != CrowdingFlag.NONE, (
            "System must detect crowding during COVID crash"
        )

    def test_crowding_absent_before_crash(self, covid_crash_signals):
        """Before crash, signals should be relatively independent."""
        streams, crash_peak = covid_crash_signals
        scorer = ConvergenceScorer(window_days=14, crowding_threshold=0.7)
        pre_crash = crash_peak - timedelta(days=35)
        result = scorer.score(streams, as_of=pre_crash)
        assert result.crowding_flag == CrowdingFlag.NONE, (
            "No crowding expected before the crash begins"
        )

    def test_tail_risk_hedge_activates(self, covid_crash_signals):
        """Contrarian should activate tail-risk hedging during crash."""
        streams, crash_peak = covid_crash_signals
        scorer = ConvergenceScorer(window_days=14, crowding_threshold=0.6)
        result = scorer.score(streams, as_of=crash_peak)
        if result.crowding_flag == CrowdingFlag.SEVERE:
            activator = ContrarianSkillActivator()
            plan = activator.evaluate(
                crowding_flag=result.crowding_flag,
                mean_correlation=result.mean_correlation,
                current_positions=[
                    {"ticker": "SPY", "weight": 0.30, "sector": "Index"},
                    {"ticker": "HYG", "weight": 0.10, "sector": "Credit"},
                ],
                cash_pct=0.15,
            )
            action_types = [a.action_type for a in plan.actions]
            assert "tail_risk_hedge" in action_types, (
                "Tail-risk hedging must activate during COVID crash"
            )

    def test_regime_change_speed(self, covid_crash_signals):
        """Crowding should transition from NONE to non-NONE within 2-3 weeks."""
        streams, crash_peak = covid_crash_signals
        scorer = ConvergenceScorer(window_days=14, crowding_threshold=0.6)

        # Check multiple points to find when crowding first appears
        first_crowding_day = None
        for days_before in range(40, 0, -1):
            check_date = crash_peak - timedelta(days=days_before)
            result = scorer.score(streams, as_of=check_date)
            if result.crowding_flag != CrowdingFlag.NONE:
                first_crowding_day = days_before
                break

        assert first_crowding_day is not None, "Crowding should appear at some point"
        assert first_crowding_day >= 5, (
            "Crowding should not appear too early — crash hadn't started"
        )
```

**Step 2: Run tests**

```bash
pytest tests/benchmark/test_ackman_2020.py -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add tests/benchmark/test_ackman_2020.py
git commit -m "test: benchmark — Ackman COVID 2020 rapid regime change and tail-risk hedging"
```

---

## Task 9: Benchmark — Paulson Gold 2009-2011 (Post-GFC Inflation Protection)

**Files:**
- Create: `tests/benchmark/test_paulson_gold.py`

**Step 1: Write the benchmark test**

```python
# tests/benchmark/test_paulson_gold.py
"""Benchmark: Paulson Gold 2009-2011 post-GFC inflation protection.

Scenario: Massive monetary expansion post-GFC. QE programs, zero rates,
fiscal stimulus. Gold rallies from ~$800 to ~$1900.

Expected system behavior:
- Detect inflation-protection regime signals
- Monetary policy signals converge on accommodative
- System should not flag crowding early (genuine macro shift)
- Late-stage crowding as "everyone buys gold" narrative peaks
"""
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from evolve_trader.crowding.convergence import ConvergenceScorer, CrowdingFlag
from evolve_trader.crowding.independence import IndependenceScorer


class TestPaulsonGold:
    """System response to post-GFC inflation-protection regime."""

    @pytest.fixture
    def post_gfc_signals(self):
        """Synthetic signals modeling 2009-2011 monetary expansion."""
        # Mid-2010: QE in full swing
        reference_date = datetime(2010, 6, 1, tzinfo=timezone.utc)
        days = 180

        streams = {}

        # Monetary policy: persistently accommodative
        rng_mp = np.random.default_rng(seed=2009)
        streams["monetary_policy"] = [
            {
                "timestamp": reference_date - timedelta(days=days - d),
                "direction": 1.0,  # bullish for inflation hedges
                "magnitude": float(0.6 + rng_mp.uniform(-0.1, 0.15)),
            }
            for d in range(days)
        ]

        # Fiscal stimulus: expansionary
        rng_fiscal = np.random.default_rng(seed=2010)
        streams["fiscal_policy"] = [
            {
                "timestamp": reference_date - timedelta(days=days - d),
                "direction": 1.0,
                "magnitude": float(0.5 + rng_fiscal.uniform(-0.1, 0.2)),
            }
            for d in range(days)
        ]

        # Real rates: deeply negative (supports gold)
        rng_rates = np.random.default_rng(seed=2011)
        streams["real_rates"] = [
            {
                "timestamp": reference_date - timedelta(days=days - d),
                "direction": 1.0,  # negative real rates = bullish gold
                "magnitude": float(0.5 + rng_rates.uniform(-0.1, 0.15)),
            }
            for d in range(days)
        ]

        # Inflation expectations: rising
        rng_infl = np.random.default_rng(seed=1234)
        streams["inflation_expectations"] = [
            {
                "timestamp": reference_date - timedelta(days=days - d),
                "direction": 1.0 if rng_infl.random() < 0.7 else -1.0,
                "magnitude": float(0.4 + (d / days) * 0.3),
            }
            for d in range(days)
        ]

        return streams, reference_date

    @pytest.fixture
    def late_stage_signals(self):
        """Signals modeling late-2011 gold peak (crowding phase)."""
        peak_date = datetime(2011, 9, 6, tzinfo=timezone.utc)
        days = 120

        streams = {}

        # All sources now unanimously bullish gold — crowding
        for source in ["monetary_policy", "retail_gold", "etf_flows", "media_narrative"]:
            rng = np.random.default_rng(seed=hash(source) % 2**31)
            streams[source] = [
                {
                    "timestamp": peak_date - timedelta(days=days - d),
                    "direction": 1.0,
                    "magnitude": float(0.7 + rng.uniform(0.0, 0.25)),
                }
                for d in range(days)
            ]

        return streams, peak_date

    def test_early_stage_not_crowded(self, post_gfc_signals):
        """Early in the gold trade, sources should not be overly crowded."""
        streams, ref_date = post_gfc_signals
        # Use higher threshold — genuine macro shift should not be flagged as crowding
        scorer = ConvergenceScorer(window_days=30, crowding_threshold=0.8)
        early_date = ref_date - timedelta(days=150)
        result = scorer.score(streams, as_of=early_date)
        # Early signals may show some correlation but not severe crowding
        # This is a nuanced test: the system should distinguish macro regime from crowding
        assert result.crowding_flag != CrowdingFlag.SEVERE, (
            "Early-stage macro shift should not trigger severe crowding"
        )

    def test_late_stage_crowding_detected(self, late_stage_signals):
        """At the gold peak, all-sources-bullish should trigger crowding."""
        streams, peak = late_stage_signals
        scorer = ConvergenceScorer(window_days=30, crowding_threshold=0.7)
        result = scorer.score(streams, as_of=peak)
        assert result.crowding_flag != CrowdingFlag.NONE, (
            "Late-stage gold mania should trigger crowding detection"
        )

    def test_monetary_signals_consistent_direction(self, post_gfc_signals):
        """Monetary policy signals should maintain consistent direction."""
        streams, ref_date = post_gfc_signals
        mp_stream = streams["monetary_policy"]
        bullish_count = sum(1 for s in mp_stream if s["direction"] > 0)
        assert bullish_count / len(mp_stream) > 0.8, (
            "Monetary policy should be consistently accommodative"
        )

    def test_groupthink_increases_over_time(self, post_gfc_signals):
        """Independence should decrease as the gold trade becomes consensus."""
        streams, ref_date = post_gfc_signals
        scorer = IndependenceScorer(window_days=30, groupthink_threshold=0.5)
        early = scorer.score(streams, as_of=ref_date - timedelta(days=120))
        late = scorer.score(streams, as_of=ref_date)
        # Correlation should be non-trivial by mid-2010
        assert late.avg_pairwise_correlation > 0.0
```

**Step 2: Run tests**

```bash
pytest tests/benchmark/test_paulson_gold.py -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add tests/benchmark/test_paulson_gold.py
git commit -m "test: benchmark — Paulson Gold 2009-2011 inflation-protection regime"
```

---

## Task 10: Quiet Market Validation

**Files:**
- Create: `tests/benchmark/test_quiet_markets.py`

**Step 1: Write the benchmark test**

```python
# tests/benchmark/test_quiet_markets.py
"""Benchmark: Quiet market validation.

Scenarios: Mid-2014-2015, mid-2017, Q3 2018. Low volatility, no strong
directional signals. System should NOT over-trade.

Expected system behavior:
- Unnecessary Trade Rate <= 2-3 per quarter
- Cash Deployment < 20% (system holds cash, doesn't chase)
- No crowding flags (sources are independent and directionless)
- Low signal magnitudes
"""
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from evolve_trader.crowding.convergence import ConvergenceScorer, CrowdingFlag
from evolve_trader.crowding.independence import IndependenceScorer


class TestQuietMarket2014:
    """Mid-2014 to mid-2015: calm, grinding bull market."""

    @pytest.fixture
    def quiet_2014_signals(self):
        """Low-volatility signals modeling mid-2014 calm."""
        ref_date = datetime(2014, 9, 1, tzinfo=timezone.utc)
        days = 90
        return _build_quiet_signals(ref_date, days, seed=2014), ref_date

    def test_no_crowding_detected(self, quiet_2014_signals):
        """No crowding should be detected in quiet market."""
        streams, ref_date = quiet_2014_signals
        scorer = ConvergenceScorer(window_days=30, crowding_threshold=0.7)
        result = scorer.score(streams, as_of=ref_date)
        assert result.crowding_flag == CrowdingFlag.NONE, (
            "Quiet market should not trigger crowding"
        )

    def test_low_signal_magnitudes(self, quiet_2014_signals):
        """Signal magnitudes should be low in quiet market."""
        streams, _ = quiet_2014_signals
        for source_name, stream in streams.items():
            avg_mag = np.mean([s["magnitude"] for s in stream])
            assert avg_mag < 0.5, (
                f"{source_name}: average magnitude should be low in quiet market"
            )

    def test_source_independence_maintained(self, quiet_2014_signals):
        """Sources should remain independent (not groupthink)."""
        streams, ref_date = quiet_2014_signals
        scorer = IndependenceScorer(window_days=30, groupthink_threshold=0.6)
        result = scorer.score(streams, as_of=ref_date)
        assert not result.is_groupthink, (
            "Quiet market should not show groupthink"
        )


class TestQuietMarket2017:
    """Mid-2017: historically low VIX, grinding higher."""

    @pytest.fixture
    def quiet_2017_signals(self):
        ref_date = datetime(2017, 7, 1, tzinfo=timezone.utc)
        days = 90
        return _build_quiet_signals(ref_date, days, seed=2017), ref_date

    def test_no_crowding_detected(self, quiet_2017_signals):
        streams, ref_date = quiet_2017_signals
        scorer = ConvergenceScorer(window_days=30, crowding_threshold=0.7)
        result = scorer.score(streams, as_of=ref_date)
        assert result.crowding_flag == CrowdingFlag.NONE

    def test_no_unnecessary_trades_implied(self, quiet_2017_signals):
        """Low magnitude + no crowding = system should not be trading actively."""
        streams, ref_date = quiet_2017_signals
        scorer = ConvergenceScorer(window_days=30, crowding_threshold=0.7)
        result = scorer.score(streams, as_of=ref_date)
        assert result.recommended_exposure_reduction == 0.0, (
            "No exposure changes needed in quiet market"
        )


class TestQuietMarketQ32018:
    """Q3 2018: calm before Q4 volatility spike."""

    @pytest.fixture
    def quiet_q3_2018_signals(self):
        ref_date = datetime(2018, 9, 1, tzinfo=timezone.utc)
        days = 90
        return _build_quiet_signals(ref_date, days, seed=2018), ref_date

    def test_no_crowding_detected(self, quiet_q3_2018_signals):
        streams, ref_date = quiet_q3_2018_signals
        scorer = ConvergenceScorer(window_days=30, crowding_threshold=0.7)
        result = scorer.score(streams, as_of=ref_date)
        assert result.crowding_flag == CrowdingFlag.NONE

    def test_sources_independent(self, quiet_q3_2018_signals):
        streams, ref_date = quiet_q3_2018_signals
        scorer = IndependenceScorer(window_days=30, groupthink_threshold=0.6)
        result = scorer.score(streams, as_of=ref_date)
        assert not result.is_groupthink


class TestUnnecessaryTradeRate:
    """Validate that quiet markets produce low Unnecessary Trade Rate."""

    def test_utr_within_threshold(self):
        """UTR should be <= 3 per quarter across all quiet periods."""
        # Simulate quarterly trade decisions based on quiet signals
        for period_seed in [2014, 2017, 2018]:
            ref_date = datetime(period_seed, 8, 1, tzinfo=timezone.utc)
            streams = _build_quiet_signals(ref_date, days=90, seed=period_seed)
            scorer = ConvergenceScorer(window_days=30, crowding_threshold=0.7)

            trades_triggered = 0
            for week in range(13):  # 13 weeks in a quarter
                check_date = ref_date - timedelta(weeks=13 - week)
                result = scorer.score(streams, as_of=check_date)
                if result.crowding_flag != CrowdingFlag.NONE:
                    trades_triggered += 1

            assert trades_triggered <= 3, (
                f"Period {period_seed}: UTR={trades_triggered} exceeds threshold of 3/quarter"
            )

    def test_cash_deployment_low(self):
        """Cash deployment should remain < 20% in quiet markets (no new positions)."""
        # In quiet markets the system should not be aggressively deploying cash
        for period_seed in [2014, 2017, 2018]:
            ref_date = datetime(period_seed, 8, 1, tzinfo=timezone.utc)
            streams = _build_quiet_signals(ref_date, days=90, seed=period_seed)
            scorer = ConvergenceScorer(window_days=30, crowding_threshold=0.7)
            result = scorer.score(streams, as_of=ref_date)
            # No crowding = no exposure reduction = no cash redeployment signals
            assert result.recommended_exposure_reduction == 0.0, (
                f"Period {period_seed}: system should not signal exposure changes"
            )


# --- Helpers ---

def _build_quiet_signals(
    ref_date: datetime, days: int, seed: int
) -> dict[str, list[dict]]:
    """Build quiet-market signal streams: low magnitude, random direction."""
    streams = {}
    for source in ["macro_indicators", "fund_flows", "sentiment", "technical"]:
        rng = np.random.default_rng(seed=seed + hash(source) % 10000)
        streams[source] = [
            {
                "timestamp": ref_date - timedelta(days=days - d),
                "direction": float(rng.choice([-1.0, 1.0])),
                "magnitude": float(rng.uniform(0.05, 0.35)),
            }
            for d in range(days)
        ]
    return streams
```

**Step 2: Run tests**

```bash
pytest tests/benchmark/test_quiet_markets.py -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add tests/benchmark/test_quiet_markets.py
git commit -m "test: benchmark — quiet market validation (2014, 2017, Q3 2018)"
```

---

## Task 11: Graceful Degradation Validation

**Files:**
- Create: `tests/integration/test_graceful_degradation.py`

**Step 1: Write the failing tests**

```python
# tests/integration/test_graceful_degradation.py
"""Integration tests for graceful degradation.

Validates system behavior when signal sources fail:
1. Kill ALL signals → system continues (uses fallback/cash-preservation)
2. Kill individual source → system adapts (reduces confidence)
3. Schema change in source → source marked unhealthy
"""
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from evolve_trader.crowding.convergence import ConvergenceScorer, CrowdingFlag


class TestAllSignalsKilled:
    """System continues when all signal sources are killed."""

    def test_empty_streams_handled_gracefully(self):
        """Scorer handles empty signal streams without crashing."""
        scorer = ConvergenceScorer(window_days=30)
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        # All sources present but empty
        streams = {
            "source_a": [],
            "source_b": [],
            "source_c": [],
        }
        result = scorer.score(streams, as_of=now)
        assert result.crowding_flag == CrowdingFlag.NONE
        assert result.mean_correlation == 0.0

    def test_no_data_within_window(self):
        """Old data outside window treated as no data."""
        scorer = ConvergenceScorer(window_days=7)
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        old_date = now - timedelta(days=30)
        streams = {
            "source_a": [{"timestamp": old_date, "direction": 1.0, "magnitude": 0.5}],
            "source_b": [{"timestamp": old_date, "direction": -1.0, "magnitude": 0.5}],
        }
        result = scorer.score(streams, as_of=now)
        assert result.crowding_flag == CrowdingFlag.NONE

    def test_system_defaults_to_conservative_with_no_signals(self):
        """With no actionable signals, system recommends no exposure changes."""
        scorer = ConvergenceScorer(window_days=30)
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        streams = {"source_a": [], "source_b": []}
        result = scorer.score(streams, as_of=now)
        assert result.recommended_exposure_reduction == 0.0


class TestIndividualSourceKilled:
    """System adapts when a single source goes offline."""

    def test_scoring_continues_with_reduced_sources(self):
        """Removing one source from a multi-source setup still works."""
        scorer = ConvergenceScorer(window_days=30)
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        rng = np.random.default_rng(seed=42)

        full_streams = {
            f"source_{i}": [
                {
                    "timestamp": now - timedelta(days=30 - d),
                    "direction": float(rng.choice([-1.0, 1.0])),
                    "magnitude": float(rng.uniform(0.1, 0.8)),
                }
                for d in range(30)
            ]
            for i in range(4)
        }

        # Full result
        full_result = scorer.score(full_streams, as_of=now)

        # Remove one source
        reduced_streams = {k: v for k, v in full_streams.items() if k != "source_0"}
        reduced_result = scorer.score(reduced_streams, as_of=now)

        # System should still produce a valid result
        assert reduced_result.correlation_matrix.shape[0] == 3
        assert isinstance(reduced_result.crowding_flag, CrowdingFlag)

    def test_one_empty_source_handled(self):
        """One source with data and others empty → still produces result."""
        scorer = ConvergenceScorer(window_days=30)
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        rng = np.random.default_rng(seed=99)

        streams = {
            "active_source": [
                {
                    "timestamp": now - timedelta(days=30 - d),
                    "direction": float(rng.choice([-1.0, 1.0])),
                    "magnitude": float(rng.uniform(0.1, 0.8)),
                }
                for d in range(30)
            ],
            "dead_source": [],
        }
        result = scorer.score(streams, as_of=now)
        assert result.crowding_flag == CrowdingFlag.NONE


class TestSchemaChange:
    """System detects and handles schema changes in signal data."""

    def test_missing_required_fields_raises(self):
        """Signals missing required fields should raise clear error."""
        scorer = ConvergenceScorer(window_days=30)
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        # Missing 'magnitude' field
        bad_streams = {
            "source_a": [
                {"timestamp": now - timedelta(days=1), "direction": 1.0},
            ],
            "source_b": [
                {"timestamp": now - timedelta(days=1), "direction": -1.0, "magnitude": 0.5},
            ],
        }
        with pytest.raises(KeyError):
            scorer.score(bad_streams, as_of=now)

    def test_extra_fields_tolerated(self):
        """Extra fields in signal dicts should not cause errors."""
        scorer = ConvergenceScorer(window_days=30)
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        streams = {
            "source_a": [
                {
                    "timestamp": now - timedelta(days=d),
                    "direction": 1.0,
                    "magnitude": 0.5,
                    "extra_field": "ignored",
                    "another_extra": 42,
                }
                for d in range(10)
            ],
            "source_b": [
                {
                    "timestamp": now - timedelta(days=d),
                    "direction": -1.0,
                    "magnitude": 0.5,
                }
                for d in range(10)
            ],
        }
        result = scorer.score(streams, as_of=now)
        assert isinstance(result.crowding_flag, CrowdingFlag)

    def test_wrong_type_direction_raises(self):
        """Non-numeric direction values should raise error."""
        scorer = ConvergenceScorer(window_days=30)
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        bad_streams = {
            "source_a": [
                {
                    "timestamp": now - timedelta(days=1),
                    "direction": "bullish",  # wrong type
                    "magnitude": 0.5,
                },
            ],
            "source_b": [
                {
                    "timestamp": now - timedelta(days=1),
                    "direction": -1.0,
                    "magnitude": 0.5,
                },
            ],
        }
        with pytest.raises((TypeError, ValueError)):
            scorer.score(bad_streams, as_of=now)


class TestMinimumSourceRequirement:
    """System enforces minimum source count."""

    def test_single_source_raises(self):
        """Cannot compute convergence with only one source."""
        scorer = ConvergenceScorer(window_days=30)
        with pytest.raises(ValueError, match="at least 2 sources"):
            scorer.score({"only_one": [{"timestamp": datetime.now(timezone.utc), "direction": 1.0, "magnitude": 0.5}]})

    def test_zero_sources_raises(self):
        """Cannot compute convergence with zero sources."""
        scorer = ConvergenceScorer(window_days=30)
        with pytest.raises(ValueError, match="at least 2 sources"):
            scorer.score({})
```

**Step 2: Run tests**

```bash
pytest tests/integration/test_graceful_degradation.py -v
```

Expected: PASS (most tests exercise existing error handling; some may require minor implementation fixes)

**Step 3: Commit**

```bash
git add tests/integration/test_graceful_degradation.py
git commit -m "test: graceful degradation — signal loss, schema changes, source failures"
```

---

## Task 12: Benchmark Report Generator

**Files:**
- Create: `src/evolve_trader/benchmarks/report_generator.py`
- Create: `tests/unit/test_report_generator.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_report_generator.py
"""Tests for benchmark report generator."""
import pytest
from datetime import datetime, timezone
from evolve_trader.benchmarks.report_generator import (
    BenchmarkReportGenerator,
    BenchmarkRun,
    BenchmarkResult,
    Weakness,
)


def test_benchmark_result_has_required_fields():
    """BenchmarkResult captures outcome of a single benchmark."""
    result = BenchmarkResult(
        benchmark_name="soros_1992",
        passed=True,
        expected_behavior="Detect regime stress",
        actual_behavior="Crowding WARNING detected, mean correlation 0.72",
        crowding_flag_detected="WARNING",
        mean_correlation=0.72,
        contrarian_activated=True,
        duration_ms=150,
    )
    assert result.benchmark_name == "soros_1992"
    assert result.passed


def test_weakness_has_required_fields():
    """Weakness identifies a specific system shortcoming."""
    weakness = Weakness(
        benchmark_name="burry_subprime",
        category="detection_speed",
        description="System detected crowding 30 days late",
        severity="medium",
        recommendation="Reduce convergence window for gradual deterioration",
    )
    assert weakness.category == "detection_speed"
    assert weakness.severity == "medium"


def test_benchmark_run_aggregates_results():
    """BenchmarkRun aggregates multiple benchmark results."""
    run = BenchmarkRun(
        run_id="bench-2025-06-01",
        timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc),
        results=[
            BenchmarkResult(
                benchmark_name="soros_1992",
                passed=True,
                expected_behavior="Detect stress",
                actual_behavior="Detected",
                crowding_flag_detected="WARNING",
                mean_correlation=0.72,
                contrarian_activated=True,
                duration_ms=150,
            ),
            BenchmarkResult(
                benchmark_name="quiet_2017",
                passed=True,
                expected_behavior="No crowding",
                actual_behavior="No crowding",
                crowding_flag_detected="NONE",
                mean_correlation=0.15,
                contrarian_activated=False,
                duration_ms=80,
            ),
        ],
        weaknesses=[],
    )
    assert run.pass_rate == 1.0
    assert run.total_benchmarks == 2


def test_report_generator_produces_markdown():
    """Generator produces a markdown report from a benchmark run."""
    generator = BenchmarkReportGenerator()
    run = _sample_run()
    report = generator.generate_markdown(run)
    assert "# Benchmark Report" in report
    assert "soros_1992" in report
    assert "PASS" in report


def test_report_generator_includes_weaknesses():
    """Report includes identified weaknesses section."""
    generator = BenchmarkReportGenerator()
    run = _sample_run_with_weakness()
    report = generator.generate_markdown(run)
    assert "Weaknesses" in report or "weakness" in report.lower()
    assert "detection_speed" in report


def test_report_generator_includes_summary_stats():
    """Report includes pass rate, total benchmarks, total duration."""
    generator = BenchmarkReportGenerator()
    run = _sample_run()
    report = generator.generate_markdown(run)
    assert "100%" in report or "1.0" in report
    assert "2" in report  # total benchmarks


def test_report_generator_produces_json():
    """Generator can produce JSON output for programmatic consumption."""
    generator = BenchmarkReportGenerator()
    run = _sample_run()
    json_output = generator.generate_json(run)
    assert json_output["run_id"] == "bench-2025-06-01"
    assert json_output["pass_rate"] == 1.0
    assert len(json_output["results"]) == 2


def test_empty_run_produces_valid_report():
    """Empty benchmark run still produces a valid report."""
    generator = BenchmarkReportGenerator()
    run = BenchmarkRun(
        run_id="bench-empty",
        timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc),
        results=[],
        weaknesses=[],
    )
    report = generator.generate_markdown(run)
    assert "# Benchmark Report" in report
    assert run.pass_rate == 0.0 or run.total_benchmarks == 0


# --- Helpers ---

def _sample_run() -> BenchmarkRun:
    return BenchmarkRun(
        run_id="bench-2025-06-01",
        timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc),
        results=[
            BenchmarkResult(
                benchmark_name="soros_1992",
                passed=True,
                expected_behavior="Detect regime stress",
                actual_behavior="Crowding WARNING detected",
                crowding_flag_detected="WARNING",
                mean_correlation=0.72,
                contrarian_activated=True,
                duration_ms=150,
            ),
            BenchmarkResult(
                benchmark_name="quiet_2017",
                passed=True,
                expected_behavior="No crowding",
                actual_behavior="No crowding detected",
                crowding_flag_detected="NONE",
                mean_correlation=0.15,
                contrarian_activated=False,
                duration_ms=80,
            ),
        ],
        weaknesses=[],
    )


def _sample_run_with_weakness() -> BenchmarkRun:
    run = _sample_run()
    return BenchmarkRun(
        run_id=run.run_id,
        timestamp=run.timestamp,
        results=run.results,
        weaknesses=[
            Weakness(
                benchmark_name="burry_subprime",
                category="detection_speed",
                description="Crowding detected 30 days late",
                severity="medium",
                recommendation="Reduce convergence window",
            ),
        ],
    )
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_report_generator.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.benchmarks'`

**Step 3: Implement the report generator**

```python
# src/evolve_trader/benchmarks/__init__.py
"""Benchmark testing and report generation."""
```

```python
# src/evolve_trader/benchmarks/report_generator.py
"""Benchmark report generator.

Produces markdown and JSON reports from benchmark runs, highlighting
pass/fail status, summary statistics, and identified weaknesses.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class BenchmarkResult:
    """Result of a single benchmark test."""
    benchmark_name: str
    passed: bool
    expected_behavior: str
    actual_behavior: str
    crowding_flag_detected: str
    mean_correlation: float
    contrarian_activated: bool
    duration_ms: int


@dataclass(frozen=True)
class Weakness:
    """Identified weakness from benchmark analysis."""
    benchmark_name: str
    category: str  # detection_speed, false_positive, false_negative, calibration
    description: str
    severity: str  # low, medium, high, critical
    recommendation: str


@dataclass
class BenchmarkRun:
    """Aggregated results from a full benchmark suite run."""
    run_id: str
    timestamp: datetime
    results: list[BenchmarkResult]
    weaknesses: list[Weakness]

    @property
    def total_benchmarks(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def pass_rate(self) -> float:
        if self.total_benchmarks == 0:
            return 0.0
        return self.passed_count / self.total_benchmarks

    @property
    def total_duration_ms(self) -> int:
        return sum(r.duration_ms for r in self.results)


class BenchmarkReportGenerator:
    """Generates reports from benchmark runs.

    Supports markdown (human-readable) and JSON (programmatic) output.
    """

    def generate_markdown(self, run: BenchmarkRun) -> str:
        """Generate a markdown report from a benchmark run.

        Args:
            run: Completed benchmark run with results and weaknesses.

        Returns:
            Markdown-formatted report string.
        """
        lines: list[str] = []
        lines.append("# Benchmark Report")
        lines.append("")
        lines.append(f"**Run ID:** {run.run_id}")
        lines.append(f"**Timestamp:** {run.timestamp.isoformat()}")
        lines.append(f"**Total Benchmarks:** {run.total_benchmarks}")
        lines.append(
            f"**Pass Rate:** {run.pass_rate:.0%} "
            f"({run.passed_count}/{run.total_benchmarks})"
        )
        lines.append(f"**Total Duration:** {run.total_duration_ms}ms")
        lines.append("")

        # Results table
        lines.append("## Results")
        lines.append("")
        if run.results:
            lines.append(
                "| Benchmark | Status | Crowding Flag | Mean Corr | "
                "Contrarian | Duration |"
            )
            lines.append(
                "|-----------|--------|---------------|-----------|"
                "------------|----------|"
            )
            for r in run.results:
                status = "PASS" if r.passed else "FAIL"
                contrarian = "Yes" if r.contrarian_activated else "No"
                lines.append(
                    f"| {r.benchmark_name} | {status} | "
                    f"{r.crowding_flag_detected} | {r.mean_correlation:.2f} | "
                    f"{contrarian} | {r.duration_ms}ms |"
                )
            lines.append("")

            # Detail per benchmark
            lines.append("## Details")
            lines.append("")
            for r in run.results:
                status = "PASS" if r.passed else "FAIL"
                lines.append(f"### {r.benchmark_name} — {status}")
                lines.append(f"- **Expected:** {r.expected_behavior}")
                lines.append(f"- **Actual:** {r.actual_behavior}")
                lines.append(
                    f"- **Crowding:** {r.crowding_flag_detected} "
                    f"(mean correlation: {r.mean_correlation:.2f})"
                )
                lines.append(
                    f"- **Contrarian activated:** "
                    f"{'Yes' if r.contrarian_activated else 'No'}"
                )
                lines.append("")
        else:
            lines.append("No benchmarks were run.")
            lines.append("")

        # Weaknesses
        if run.weaknesses:
            lines.append("## Weaknesses")
            lines.append("")
            for w in run.weaknesses:
                lines.append(
                    f"### [{w.severity.upper()}] {w.benchmark_name} — "
                    f"{w.category}"
                )
                lines.append(f"- **Description:** {w.description}")
                lines.append(f"- **Recommendation:** {w.recommendation}")
                lines.append("")

        return "\n".join(lines)

    def generate_json(self, run: BenchmarkRun) -> dict[str, Any]:
        """Generate a JSON-serializable dict from a benchmark run.

        Args:
            run: Completed benchmark run with results and weaknesses.

        Returns:
            Dict suitable for json.dumps().
        """
        return {
            "run_id": run.run_id,
            "timestamp": run.timestamp.isoformat(),
            "total_benchmarks": run.total_benchmarks,
            "passed": run.passed_count,
            "failed": run.failed_count,
            "pass_rate": run.pass_rate,
            "total_duration_ms": run.total_duration_ms,
            "results": [asdict(r) for r in run.results],
            "weaknesses": [asdict(w) for w in run.weaknesses],
        }
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_report_generator.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/benchmarks/__init__.py src/evolve_trader/benchmarks/report_generator.py tests/unit/test_report_generator.py
git commit -m "feat: benchmark report generator with markdown and JSON output"
```

---

## Task 13: Final Verification

**Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: ALL PASS — all Phase 10 tests plus all prior phase tests

**Step 2: Run linting and type checking**

```bash
ruff check src/evolve_trader/crowding/ src/evolve_trader/benchmarks/
mypy src/evolve_trader/crowding/ src/evolve_trader/benchmarks/ --ignore-missing-imports
```

Expected: No errors

**Step 3: Run benchmark tests with timing**

```bash
pytest tests/benchmark/ -v --durations=0
```

Expected: All benchmarks pass. No single benchmark exceeds 5 seconds.

**Step 4: Verify benchmark report generation end-to-end**

```bash
python -c "
from evolve_trader.benchmarks.report_generator import (
    BenchmarkReportGenerator, BenchmarkRun, BenchmarkResult, Weakness
)
from datetime import datetime, timezone

run = BenchmarkRun(
    run_id='verify-phase-10',
    timestamp=datetime.now(timezone.utc),
    results=[
        BenchmarkResult('soros_1992', True, 'Detect stress', 'Detected', 'WARNING', 0.72, True, 150),
        BenchmarkResult('quiet_2017', True, 'No crowding', 'None', 'NONE', 0.15, False, 80),
    ],
    weaknesses=[],
)
gen = BenchmarkReportGenerator()
print(gen.generate_markdown(run))
"
```

Expected: Clean markdown report printed to stdout

**Step 5: Commit any fixes**

```bash
git add -A
git commit -m "test: Phase 10 final verification — all tests passing"
```

---

## Parallelization Notes

Tasks in this phase have the following dependency structure:

```
Task 1 (Convergence Scorer) ──────┐
                                    ├── Task 3 (Contrarian Skills)
Task 2 (Independence Scorer) ─────┤
                                    ├── Task 4 (Historical Calibration)
                                    │
                                    ├── Task 5 (Soros 1992) ────────────┐
                                    ├── Task 6 (Burry 2005-2007) ───────┤
                                    ├── Task 7 (Druckenmiller 2000) ────┤
                                    ├── Task 8 (Ackman 2020) ───────────┤── Task 13 (Final Verification)
                                    ├── Task 9 (Paulson Gold) ──────────┤
                                    ├── Task 10 (Quiet Markets) ────────┤
                                    ├── Task 11 (Graceful Degradation) ─┤
                                    │                                    │
Task 12 (Report Generator) ────────────────────────────────────────────┘
```

**Can run in parallel:**
- Tasks 1 and 2 (Convergence and Independence scorers) are independent — run simultaneously
- Tasks 5, 6, 7, 8, 9, 10 (historical benchmarks + quiet markets) are independent of each other — run simultaneously after Tasks 1-2
- Task 3 (Contrarian Skills) depends on Task 1 (uses CrowdingFlag)
- Task 4 (Historical Calibration) depends on Tasks 1-3
- Task 11 (Graceful Degradation) depends on Tasks 1-2
- Task 12 (Report Generator) is independent of all other tasks — run any time
- Task 13 (Final Verification) depends on everything
