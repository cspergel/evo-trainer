"""Tests for the walk-forward validation harness."""

from evolve_trader.core.validation import (
    WalkForwardConfig,
    WalkForwardWindow,
    generate_walk_forward_windows,
    validate_no_lookahead,
)


def test_generate_windows_correct_count():
    """Generate correct number of train/validate windows."""
    config = WalkForwardConfig(
        total_days=100,
        train_days=30,
        validate_days=10,
        step_days=10,
    )
    windows = generate_walk_forward_windows(config)
    assert len(windows) > 0
    for w in windows:
        assert w.train_end == w.validate_start
        assert w.validate_end - w.validate_start == 10
        assert w.train_end - w.train_start == 30


def test_windows_no_overlap():
    """Validation data never overlaps with training data."""
    config = WalkForwardConfig(
        total_days=100,
        train_days=30,
        validate_days=10,
        step_days=10,
    )
    windows = generate_walk_forward_windows(config)
    for w in windows:
        assert w.validate_start >= w.train_end


def test_validate_no_lookahead():
    """Lookahead check catches data leakage."""
    window = WalkForwardWindow(
        train_start=0,
        train_end=30,
        validate_start=30,
        validate_end=40,
    )
    assert validate_no_lookahead(window, data_timestamp=15) is True
    assert validate_no_lookahead(window, data_timestamp=35) is False
    assert validate_no_lookahead(window, data_timestamp=50) is False


def test_windows_cover_full_range():
    """Windows span the full data range without gaps in validation coverage."""
    config = WalkForwardConfig(
        total_days=100,
        train_days=30,
        validate_days=10,
        step_days=10,
    )
    windows = generate_walk_forward_windows(config)
    last_validate_end = max(w.validate_end for w in windows)
    assert last_validate_end <= 100


def test_edge_case_insufficient_data():
    """If total days < train + validate, no windows are generated."""
    config = WalkForwardConfig(
        total_days=20,
        train_days=30,
        validate_days=10,
        step_days=10,
    )
    windows = generate_walk_forward_windows(config)
    assert len(windows) == 0


def test_boundary_at_train_end():
    """Data at exactly train_end is lookahead (validation period starts there)."""
    window = WalkForwardWindow(
        train_start=0,
        train_end=30,
        validate_start=30,
        validate_end=40,
    )
    assert validate_no_lookahead(window, data_timestamp=30) is False
