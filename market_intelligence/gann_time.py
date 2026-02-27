"""
Phase 6 — Gann Time Light Model
Candle-count cycle analysis using standard Gann time cycles.
No geometric Square-of-9 calculations.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from market_intelligence.config import GANN_CYCLES, GANN_CYCLE_TOLERANCE

logger = logging.getLogger(__name__)


class GannTimeAnalyzer:
    """
    Detects Gann time cycles by counting bars from significant
    swing points and checking for cycle convergence.
    """

    def __init__(
        self,
        cycles: Optional[List[int]] = None,
        tolerance: int = GANN_CYCLE_TOLERANCE,
    ):
        self.cycles = cycles or GANN_CYCLES
        self.tolerance = tolerance

    def analyze(self, df: pd.DataFrame) -> "GannResult":
        """
        Analyze Gann time cycles.

        Args:
            df: OHLCV DataFrame

        Returns:
            GannResult with active cycles and confluence
        """
        from market_intelligence.models import GannResult

        if df.empty or len(df) < max(self.cycles):
            return GannResult()

        highs = df["High"].values
        lows = df["Low"].values
        bar_count = len(df)

        # Find significant pivot points
        pivot_indices = self._find_pivots(highs, lows)

        if not pivot_indices:
            return GannResult()

        # Check which cycles are active (current bar near a cycle turn)
        active_cycles: List[int] = []
        next_cycle_bars: List[int] = []

        for cycle in self.cycles:
            is_active, bars_to_next = self._check_cycle(pivot_indices, bar_count, cycle)
            if is_active:
                active_cycles.append(cycle)
            next_cycle_bars.append(bars_to_next)

        # Check for confluence (multiple cycles converging)
        cycle_confluence = len(active_cycles) >= 2

        # Score
        score = self._calculate_score(active_cycles, cycle_confluence, next_cycle_bars)

        return GannResult(
            active_cycles=active_cycles,
            next_cycle_bars=next_cycle_bars,
            cycle_confluence=cycle_confluence,
            score=score,
        )

    def _find_pivots(self, highs: np.ndarray, lows: np.ndarray, lookback: int = 10) -> List[int]:
        """Find significant pivot points (both highs and lows)."""
        pivots = []

        for i in range(lookback, len(highs) - lookback):
            # Swing high
            if highs[i] == max(highs[i - lookback:i + lookback + 1]):
                pivots.append(i)
            # Swing low
            elif lows[i] == min(lows[i - lookback:i + lookback + 1]):
                pivots.append(i)

        return sorted(set(pivots))

    def _check_cycle(
        self, pivot_indices: List[int], current_bar: int, cycle_length: int
    ) -> Tuple[bool, int]:
        """
        Check if a cycle is currently active.
        A cycle is active if the distance from any significant pivot
        to the current bar is a multiple of the cycle length (within tolerance).
        """
        best_bars_to_next = cycle_length  # Default: full cycle away

        for pivot_idx in pivot_indices:
            bars_since = current_bar - pivot_idx

            if bars_since <= 0:
                continue

            # How many complete cycles since this pivot
            cycles_elapsed = bars_since / cycle_length
            remainder = bars_since % cycle_length

            # Distance to nearest cycle multiple
            dist_to_cycle = min(remainder, cycle_length - remainder)

            if dist_to_cycle <= self.tolerance:
                # We're at or very near a cycle turn
                return True, 0

            # Track bars to next cycle point
            bars_to_next = cycle_length - remainder
            best_bars_to_next = min(best_bars_to_next, bars_to_next)

        return False, best_bars_to_next

    def _calculate_score(
        self,
        active_cycles: List[int],
        cycle_confluence: bool,
        next_cycle_bars: List[int],
    ) -> float:
        """Calculate Gann time score 0-100."""
        score = 0.0

        # Active cycles
        score += min(len(active_cycles) * 15, 45)

        # Confluence bonus
        if cycle_confluence:
            score += 25.0

        # Proximity to next cycle turn
        if next_cycle_bars:
            min_bars = min(next_cycle_bars)
            if min_bars <= 3:
                score += 20.0
            elif min_bars <= 7:
                score += 10.0

        return min(score, 100.0)
