"""
Phase 5 — Fibonacci Confluence
Calculate retracements, extensions, and detect confluence zones.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from market_intelligence.config import (
    FIBONACCI_CONFLUENCE_TOLERANCE,
    FIBONACCI_EXTENSION_LEVELS,
    FIBONACCI_RETRACEMENT_LEVELS,
)
from market_intelligence.models import (
    FibLevel,
    FibonacciResult,
    SwingPoint,
    SwingType,
)

logger = logging.getLogger(__name__)


class FibonacciAnalyzer:
    """
    Calculates Fibonacci retracement and extension levels,
    then identifies confluence zones where multiple levels cluster.
    """

    def __init__(
        self,
        retracement_levels: Optional[List[float]] = None,
        extension_levels: Optional[List[float]] = None,
        confluence_tolerance: float = FIBONACCI_CONFLUENCE_TOLERANCE,
    ):
        self.retracement_levels = retracement_levels or FIBONACCI_RETRACEMENT_LEVELS
        self.extension_levels = extension_levels or FIBONACCI_EXTENSION_LEVELS
        self.tolerance = confluence_tolerance

    def analyze(
        self,
        df: pd.DataFrame,
        swing_points: Optional[List[SwingPoint]] = None,
    ) -> FibonacciResult:
        """
        Compute Fibonacci levels from the most recent significant swing.

        Args:
            df: OHLCV DataFrame
            swing_points: Pre-computed swing points

        Returns:
            FibonacciResult with retracements, extensions, confluences
        """
        if df.empty or len(df) < 10:
            return FibonacciResult()

        # Find the most recent significant high and low
        swing_high, swing_low = self._find_anchor_swings(df, swing_points)

        if swing_high is None or swing_low is None:
            return FibonacciResult()

        # Determine direction: if high is more recent, downward retracement
        is_upswing = swing_low < swing_high and self._is_upswing(df, swing_points)

        # Calculate retracement levels
        retracement_levels = self._calculate_retracements(swing_high, swing_low, is_upswing)

        # Calculate extension levels
        extension_levels = self._calculate_extensions(swing_high, swing_low, is_upswing)

        # Find confluence zones
        all_levels = retracement_levels + extension_levels
        confluence_zones = self._find_confluences(all_levels, df["Close"].iloc[-1])

        # Score
        score = self._calculate_score(retracement_levels, extension_levels, confluence_zones, df["Close"].iloc[-1])

        return FibonacciResult(
            retracement_levels=retracement_levels,
            extension_levels=extension_levels,
            confluence_zones=confluence_zones,
            score=score,
        )

    def _find_anchor_swings(
        self,
        df: pd.DataFrame,
        swing_points: Optional[List[SwingPoint]],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Find the significant high and low for Fibonacci calculation."""
        if swing_points and len(swing_points) >= 2:
            highs = [sp for sp in swing_points if sp.swing_type in (SwingType.HH, SwingType.LH)]
            lows = [sp for sp in swing_points if sp.swing_type in (SwingType.HL, SwingType.LL)]

            swing_high = max((sp.price for sp in highs), default=None) if highs else None
            swing_low = min((sp.price for sp in lows), default=None) if lows else None

            if swing_high and swing_low:
                return swing_high, swing_low

        # Fallback: use rolling window
        lookback = min(50, len(df))
        recent = df.iloc[-lookback:]
        return float(recent["High"].max()), float(recent["Low"].min())

    def _is_upswing(self, df: pd.DataFrame, swing_points: Optional[List[SwingPoint]]) -> bool:
        """Determine if the recent move was up or down."""
        if swing_points and len(swing_points) >= 2:
            last = swing_points[-1]
            prev = swing_points[-2]
            return last.price > prev.price

        # Fallback: compare close to midpoint
        lookback = min(20, len(df))
        recent = df.iloc[-lookback:]
        mid = (recent["High"].max() + recent["Low"].min()) / 2
        return df["Close"].iloc[-1] > mid

    def _calculate_retracements(
        self, high: float, low: float, is_upswing: bool
    ) -> List[FibLevel]:
        """Calculate Fibonacci retracement levels."""
        levels = []
        diff = high - low

        for ratio in self.retracement_levels:
            if is_upswing:
                # Retracing from high downward
                price = high - (diff * ratio)
            else:
                # Retracing from low upward
                price = low + (diff * ratio)

            levels.append(FibLevel(
                ratio=ratio,
                price=round(price, 2),
                level_type="retracement",
            ))

        return levels

    def _calculate_extensions(
        self, high: float, low: float, is_upswing: bool
    ) -> List[FibLevel]:
        """Calculate Fibonacci extension levels."""
        levels = []
        diff = high - low

        for ratio in self.extension_levels:
            if is_upswing:
                # Extension upward beyond high
                price = low + (diff * ratio)
            else:
                # Extension downward beyond low
                price = high - (diff * ratio)

            levels.append(FibLevel(
                ratio=ratio,
                price=round(price, 2),
                level_type="extension",
            ))

        return levels

    def _find_confluences(
        self, all_levels: List[FibLevel], current_price: float
    ) -> List[Tuple[float, float]]:
        """
        Find zones where multiple Fibonacci levels cluster together.
        Returns list of (zone_low, zone_high) tuples.
        """
        if not all_levels:
            return []

        prices = sorted(set(lv.price for lv in all_levels))
        if len(prices) < 2:
            return []

        confluences: List[Tuple[float, float]] = []
        used = set()

        for i, p1 in enumerate(prices):
            if i in used:
                continue
            cluster = [p1]
            for j in range(i + 1, len(prices)):
                if j in used:
                    continue
                if abs(prices[j] - p1) / max(p1, 1e-10) <= self.tolerance:
                    cluster.append(prices[j])
                    used.add(j)

            if len(cluster) >= 2:
                zone_low = min(cluster)
                zone_high = max(cluster)
                confluences.append((round(zone_low, 2), round(zone_high, 2)))
                used.add(i)

        return confluences

    def _calculate_score(
        self,
        retracements: List[FibLevel],
        extensions: List[FibLevel],
        confluences: List[Tuple[float, float]],
        current_price: float,
    ) -> float:
        """Calculate Fibonacci score 0-100."""
        score = 0.0

        # Base score for having levels
        if retracements:
            score += 15.0
        if extensions:
            score += 15.0

        # Confluence zones are very valuable
        score += min(len(confluences) * 15, 30)

        # Price near a Fibonacci level
        all_prices = [lv.price for lv in retracements + extensions]
        if current_price > 0 and all_prices:
            closest = min(abs(p - current_price) / current_price for p in all_prices)
            if closest < 0.005:
                score += 25.0  # Very close to a Fib level
            elif closest < 0.02:
                score += 15.0

        # Price within a confluence zone
        for zone_low, zone_high in confluences:
            if zone_low <= current_price <= zone_high:
                score += 20.0
                break

        return min(score, 100.0)
