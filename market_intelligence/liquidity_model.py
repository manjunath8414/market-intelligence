"""
Phase 3 — Liquidity Model
Detects equal highs/lows, recent swing liquidity,
support/resistance clusters, and liquidity sweep candles.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from market_intelligence.config import LIQUIDITY_CFG, LiquidityConfig
from market_intelligence.models import (
    Direction,
    LiquidityResult,
    LiquidityZone,
)

logger = logging.getLogger(__name__)


class LiquidityAnalyzer:
    """
    Identifies liquidity pools, equal levels, S/R clusters,
    and sweep candles.
    """

    def __init__(self, config: Optional[LiquidityConfig] = None):
        self.cfg = config or LIQUIDITY_CFG

    def analyze(self, df: pd.DataFrame) -> LiquidityResult:
        """
        Full liquidity analysis.

        Args:
            df: OHLCV DataFrame

        Returns:
            LiquidityResult with equal levels, S/R, sweeps, and score
        """
        if df.empty or len(df) < 20:
            return LiquidityResult()

        highs = df["High"].values
        lows = df["Low"].values
        opens = df["Open"].values
        closes = df["Close"].values

        lookback = min(self.cfg.lookback_bars, len(df))
        h = highs[-lookback:]
        l = lows[-lookback:]
        o = opens[-lookback:]
        c = closes[-lookback:]

        # Step 1: Equal highs and lows
        equal_highs = self._find_equal_levels(h, is_high=True)
        equal_lows = self._find_equal_levels(l, is_high=False)

        # Step 2: Support/Resistance clusters
        support_levels = self._find_sr_clusters(l, is_support=True)
        resistance_levels = self._find_sr_clusters(h, is_support=False)

        # Step 3: Liquidity sweep detection
        sweep_detected, sweep_direction = self._detect_sweep(h, l, o, c, equal_highs, equal_lows)

        # Build zones
        zones = self._build_zones(equal_highs, equal_lows, support_levels, resistance_levels, sweep_detected)

        # Score
        score = self._calculate_score(
            equal_highs, equal_lows, support_levels, resistance_levels,
            sweep_detected, closes[-1] if len(closes) > 0 else 0
        )

        return LiquidityResult(
            equal_highs=equal_highs,
            equal_lows=equal_lows,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            sweep_detected=sweep_detected,
            sweep_direction=sweep_direction,
            zones=zones,
            score=score,
        )

    def _find_equal_levels(self, prices: np.ndarray, is_high: bool) -> List[float]:
        """Find prices that cluster at the same level (equal highs/lows)."""
        tolerance = self.cfg.equal_level_tolerance
        levels: List[float] = []

        # Find local extremes
        extremes = self._local_extremes(prices, is_high)

        if len(extremes) < 2:
            return levels

        # Compare each pair
        for i in range(len(extremes)):
            for j in range(i + 1, len(extremes)):
                diff = abs(extremes[i] - extremes[j]) / max(extremes[i], 1e-10)
                if diff <= tolerance:
                    avg = (extremes[i] + extremes[j]) / 2
                    # Avoid duplicates
                    if not any(abs(avg - lv) / max(avg, 1e-10) < tolerance for lv in levels):
                        levels.append(round(avg, 2))

        return levels

    def _local_extremes(self, prices: np.ndarray, is_high: bool) -> List[float]:
        """Find local maxima or minima."""
        extremes = []
        lookback = 3
        for i in range(lookback, len(prices) - lookback):
            window_left = prices[i - lookback:i]
            window_right = prices[i + 1:i + lookback + 1]
            if is_high:
                if prices[i] >= max(window_left) and prices[i] >= max(window_right):
                    extremes.append(prices[i])
            else:
                if prices[i] <= min(window_left) and prices[i] <= min(window_right):
                    extremes.append(prices[i])
        return extremes

    def _find_sr_clusters(self, prices: np.ndarray, is_support: bool) -> List[float]:
        """Find support/resistance clusters by grouping nearby price levels."""
        tolerance = self.cfg.cluster_tolerance
        extremes = self._local_extremes(prices, is_high=not is_support)

        if not extremes:
            return []

        # Sort and cluster
        sorted_levels = sorted(extremes)
        clusters: List[List[float]] = []
        current_cluster = [sorted_levels[0]]

        for i in range(1, len(sorted_levels)):
            if (sorted_levels[i] - current_cluster[-1]) / max(current_cluster[-1], 1e-10) <= tolerance:
                current_cluster.append(sorted_levels[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [sorted_levels[i]]
        clusters.append(current_cluster)

        # Return cluster centers with 2+ touches
        result = []
        for cluster in clusters:
            if len(cluster) >= 2:
                result.append(round(np.mean(cluster), 2))

        return result

    def _detect_sweep(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        opens: np.ndarray,
        closes: np.ndarray,
        equal_highs: List[float],
        equal_lows: List[float],
    ) -> Tuple[bool, Optional[Direction]]:
        """
        Detect liquidity sweep: price spikes beyond a level then reverses.
        A sweep candle has a long wick past the level but closes back inside.
        """
        if len(highs) < 3:
            return False, None

        last_idx = len(highs) - 1
        tolerance = self.cfg.equal_level_tolerance

        # Check last few candles for sweep
        for i in range(max(0, last_idx - 2), last_idx + 1):
            body_top = max(opens[i], closes[i])
            body_bottom = min(opens[i], closes[i])
            upper_wick = highs[i] - body_top
            lower_wick = body_bottom - lows[i]
            body_size = body_top - body_bottom

            # Sweep of highs (bearish sweep)
            for level in equal_highs:
                if highs[i] > level and closes[i] < level:
                    if body_size > 0 and upper_wick / body_size >= self.cfg.sweep_wick_ratio:
                        return True, Direction.SHORT

            # Sweep of lows (bullish sweep)
            for level in equal_lows:
                if lows[i] < level and closes[i] > level:
                    if body_size > 0 and lower_wick / body_size >= self.cfg.sweep_wick_ratio:
                        return True, Direction.LONG

            # Also check S/R sweeps if no equal levels swept
            for level in self._find_sr_clusters(highs, is_support=False):
                if highs[i] > level * (1 + tolerance) and closes[i] < level:
                    return True, Direction.SHORT

            for level in self._find_sr_clusters(lows, is_support=True):
                if lows[i] < level * (1 - tolerance) and closes[i] > level:
                    return True, Direction.LONG

        return False, None

    def _build_zones(
        self,
        equal_highs: List[float],
        equal_lows: List[float],
        supports: List[float],
        resistances: List[float],
        sweep_detected: bool,
    ) -> List[LiquidityZone]:
        """Compile all liquidity zones."""
        zones: List[LiquidityZone] = []

        for level in equal_highs:
            zones.append(LiquidityZone(price_level=level, zone_type="equal_high", strength=2))
        for level in equal_lows:
            zones.append(LiquidityZone(price_level=level, zone_type="equal_low", strength=2))
        for level in supports:
            zones.append(LiquidityZone(price_level=level, zone_type="sr_cluster", strength=3))
        for level in resistances:
            zones.append(LiquidityZone(price_level=level, zone_type="sr_cluster", strength=3))

        return zones

    def _calculate_score(
        self,
        equal_highs: List[float],
        equal_lows: List[float],
        supports: List[float],
        resistances: List[float],
        sweep_detected: bool,
        current_price: float,
    ) -> float:
        """Calculate liquidity score 0-100."""
        score = 0.0

        # Equal levels provide liquidity targets
        score += min(len(equal_highs) * 8, 20)
        score += min(len(equal_lows) * 8, 20)

        # S/R clusters
        score += min(len(supports) * 5, 15)
        score += min(len(resistances) * 5, 15)

        # Proximity to S/R (within 2%)
        all_levels = supports + resistances + equal_highs + equal_lows
        if current_price > 0 and all_levels:
            closest_dist = min(abs(lv - current_price) / current_price for lv in all_levels)
            if closest_dist < 0.02:
                score += 15

        # Sweep is a strong signal
        if sweep_detected:
            score += 25

        return min(score, 100.0)
