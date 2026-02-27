"""
Phase 8 — Timing Engine (Light)
Detects expansion probability using ATR contraction,
range compression, and breakout velocity.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from market_intelligence.config import TIMING_CFG, TimingConfig
from market_intelligence.models import TimingResult

logger = logging.getLogger(__name__)


class TimingEngine:
    """
    Evaluates whether the market is likely to expand soon
    based on volatility contraction patterns.
    """

    def __init__(self, config: Optional[TimingConfig] = None):
        self.cfg = config or TIMING_CFG

    def analyze(self, df: pd.DataFrame) -> TimingResult:
        """
        Timing analysis.

        Args:
            df: OHLCV DataFrame

        Returns:
            TimingResult with contraction/compression/velocity flags
        """
        if df.empty or len(df) < self.cfg.atr_period + 10:
            return TimingResult()

        highs = df["High"].values
        lows = df["Low"].values
        closes = df["Close"].values

        # Signal 1: ATR contraction
        atr_contracted = self._check_atr_contraction(highs, lows, closes)

        # Signal 2: Range compression
        range_compressed = self._check_range_compression(highs, lows)

        # Signal 3: Breakout velocity
        breakout_velocity_met = self._check_breakout_velocity(highs, lows, closes)

        # Expansion probability
        expansion_probability = self._calculate_expansion_probability(
            atr_contracted, range_compressed, breakout_velocity_met
        )

        # Score
        score = expansion_probability  # Direct mapping

        return TimingResult(
            atr_contracted=atr_contracted,
            range_compressed=range_compressed,
            breakout_velocity_met=breakout_velocity_met,
            expansion_probability=expansion_probability,
            score=score,
        )

    def _calculate_atr(
        self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int
    ) -> np.ndarray:
        """Calculate ATR series."""
        tr = np.zeros(len(highs))
        tr[0] = highs[0] - lows[0]

        for i in range(1, len(highs)):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )

        # Simple moving average of TR
        atr = np.zeros(len(tr))
        for i in range(period - 1, len(tr)):
            atr[i] = np.mean(tr[i - period + 1:i + 1])

        return atr

    def _check_atr_contraction(
        self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
    ) -> bool:
        """
        ATR contraction: current ATR significantly below recent average.
        Low volatility often precedes big moves.
        """
        atr = self._calculate_atr(highs, lows, closes, self.cfg.atr_period)

        if len(atr) < 20:
            return False

        current_atr = atr[-1]
        avg_atr = np.mean(atr[-20:])

        if avg_atr == 0:
            return False

        return bool(current_atr < avg_atr * self.cfg.atr_contraction_threshold)

    def _check_range_compression(self, highs: np.ndarray, lows: np.ndarray) -> bool:
        """
        Range compression: recent N bars have decreasing range.
        Narrowing bars indicate a coiling pattern.
        """
        n = self.cfg.range_compression_bars
        if len(highs) < n:
            return False

        recent_highs = highs[-n:]
        recent_lows = lows[-n:]
        ranges = recent_highs - recent_lows

        # Check if ranges are generally decreasing
        decreasing_count = sum(
            1 for i in range(1, len(ranges)) if ranges[i] < ranges[i - 1]
        )

        # At least 60% of bars should have decreasing range
        return bool(decreasing_count >= n * 0.6)

    def _check_breakout_velocity(
        self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
    ) -> bool:
        """
        Breakout velocity: last candle's move exceeds ATR threshold.
        Strong moves signal conviction.
        """
        if len(closes) < 2:
            return False

        atr = self._calculate_atr(highs, lows, closes, self.cfg.atr_period)
        if atr[-1] == 0:
            return False

        last_move = abs(closes[-1] - closes[-2])
        return bool(last_move > atr[-2] * self.cfg.breakout_velocity_threshold)

    def _calculate_expansion_probability(
        self,
        atr_contracted: bool,
        range_compressed: bool,
        breakout_velocity: bool,
    ) -> float:
        """
        Calculate probability of price expansion.
        More signals = higher probability.
        """
        probability = 20.0  # Base

        if atr_contracted:
            probability += 30.0
        if range_compressed:
            probability += 25.0
        if breakout_velocity:
            probability += 25.0

        return min(probability, 100.0)
