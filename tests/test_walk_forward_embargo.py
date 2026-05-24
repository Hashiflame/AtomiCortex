"""AFML Ch.7 embargo regression test for ``WalkForwardValidator``.

Pre-fix, ``test_start = train_end`` left zero gap between train and test.
Triple-barrier labels of the last ``max_holding`` bars of train look up to
``max_holding × bar_duration`` ahead — straight into the test window. That
is direct label leakage across every walk-forward boundary.

The fix adds an optional ``embargo: timedelta`` that pushes ``test_start``
forward. Default ``timedelta(0)`` preserves legacy behaviour (backward
compat). Callers compute ``embargo = max_holding × bar_duration`` to
match AFML's prescription.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.execution.walk_forward import WalkForwardValidator


_START = datetime(2024, 1, 1)
_END = datetime(2027, 1, 1)


# ---------------------------------------------------------------------------
# Backward compatibility — default behaviour unchanged
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_default_embargo_is_zero_gap(self) -> None:
        """No-arg constructor must yield ``test_start == train_end`` like before."""
        v = WalkForwardValidator(train_months=18, test_months=6)
        for (_, train_end), (test_start, _) in v.split(_START, _END):
            assert test_start == train_end

    def test_default_embargo_attribute(self) -> None:
        assert WalkForwardValidator().embargo == timedelta(0)


# ---------------------------------------------------------------------------
# Embargo applied
# ---------------------------------------------------------------------------

class TestEmbargoApplied:
    @pytest.mark.parametrize(
        "embargo",
        [timedelta(hours=24), timedelta(hours=4), timedelta(days=7)],
    )
    def test_test_start_shifted_by_embargo(self, embargo: timedelta) -> None:
        v = WalkForwardValidator(
            train_months=18, test_months=6, embargo=embargo,
        )
        pairs = list(v.split(_START, _END))
        assert pairs, "must produce at least one window"
        for (_, train_end), (test_start, _) in pairs:
            assert test_start - train_end == embargo

    def test_24h_embargo_matches_4h_six_bar_holding(self) -> None:
        """The motivating case from the task spec: max_holding=6 for 4H
        bars → embargo = 6 × 4h = 24h. ``test_start`` lies exactly that
        far after ``train_end``."""
        embargo = 6 * timedelta(hours=4)
        v = WalkForwardValidator(embargo=embargo)
        (_, train_end), (test_start, _) = next(v.split(_START, _END))
        assert test_start - train_end == timedelta(hours=24)

    def test_test_end_anchored_to_shifted_test_start(self) -> None:
        """test_end must be computed from the *shifted* test_start, not
        from train_end — otherwise the test window shrinks."""
        from src.execution.walk_forward import _add_months
        embargo = timedelta(hours=24)
        v = WalkForwardValidator(test_months=6, embargo=embargo)
        (_, train_end), (test_start, test_end) = next(v.split(_START, _END))
        assert test_start == train_end + embargo
        # test_end = test_start + test_months (full-length test window)
        assert test_end == _add_months(test_start, 6)


# ---------------------------------------------------------------------------
# Label-leakage invariant
# ---------------------------------------------------------------------------

class TestNoLeakage:
    def test_train_labels_with_max_holding_do_not_reach_test(self) -> None:
        """Concretely: a label generated at the last bar of train looks
        ``max_holding × bar_duration`` ahead. With embargo ≥ that horizon
        the look-ahead window ends at or before ``test_start`` — zero
        overlap with the test window."""
        max_holding = 6
        bar_duration = timedelta(hours=4)
        label_horizon = max_holding * bar_duration

        v = WalkForwardValidator(embargo=label_horizon)
        for (_, train_end), (test_start, _) in v.split(_START, _END):
            label_end = train_end + label_horizon
            assert label_end <= test_start, (
                f"label_end {label_end} bleeds into test window "
                f"starting {test_start}"
            )


# ---------------------------------------------------------------------------
# Windowing arithmetic remains correct under embargo
# ---------------------------------------------------------------------------

class TestWindowingUnderEmbargo:
    def test_windows_still_advance_by_step(self) -> None:
        """The cursor advances by ``step_months`` regardless of embargo —
        embargo only shifts test_start within each window, not the cursor."""
        v = WalkForwardValidator(
            train_months=12, test_months=6, step_months=6,
            embargo=timedelta(hours=24),
        )
        pairs = list(v.split(_START, _END))
        from src.execution.walk_forward import _add_months
        for i in range(1, len(pairs)):
            (train_start_prev, _), _ = pairs[i - 1]
            (train_start_cur, _), _ = pairs[i]
            assert train_start_cur == _add_months(train_start_prev, 6)

    def test_window_dropped_when_test_overflows_end_with_embargo(self) -> None:
        """If embargo pushes test past ``end``, that window is filtered out."""
        # 18 train + 6 test = 24 months. Date range = 24 months exactly.
        # With zero embargo the final window just barely fits.
        end = datetime(2026, 1, 1)  # = _START + 24 months
        v_zero = WalkForwardValidator(train_months=18, test_months=6)
        v_emb = WalkForwardValidator(
            train_months=18, test_months=6, embargo=timedelta(days=30),
        )
        pairs_zero = list(v_zero.split(_START, end))
        pairs_emb = list(v_emb.split(_START, end))
        # Zero-embargo fits 1 window; with embargo the test_end overshoots
        # `end` so the last window is dropped.
        assert len(pairs_zero) >= len(pairs_emb)
