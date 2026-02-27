"""
Phase 7 — Market Intent Logic
Infers institutional intent using liquidity sweep, structure shift,
rejection candles, and trend continuation behavior.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from market_intelligence.models import (
    Direction,
    IntentResult,
    LiquidityResult,
    StructureResult,
    TrendDirection,
)

logger = logging.getLogger(__name__)


class MarketIntentAnalyzer:
    """
    Combines multiple signals to infer market intent (direction).
    """

    def analyze(
        self,
        df: pd.DataFrame,
        structure: Optional[StructureResult] = None,
        liquidity: Optional[LiquidityResult] = None,
    ) -> IntentResult:
        """
        Infer market intent from price action, structure, and liquidity.

        Args:
            df: OHLCV DataFrame
            structure: Market structure analysis result
            liquidity: Liquidity analysis result

        Returns:
            IntentResult with inferred direction and signals
        """
        if df.empty or len(df) < 10:
            return IntentResult()

        opens = df["Open"].values
        highs = df["High"].values
        lows = df["Low"].values
        closes = df["Close"].values

        # Signal 1: Liquidity sweep
        sweep_detected = False
        sweep_direction = Direction.WAIT
        if liquidity and liquidity.sweep_detected:
            sweep_detected = True
            sweep_direction = liquidity.sweep_direction or Direction.WAIT

        # Signal 2: Structure shift
        structure_shift = False
        shift_direction = Direction.WAIT
        if structure:
            if structure.choch_detected:
                structure_shift = True
                # CHoCH in uptrend = potential bearish shift
                if structure.trend == TrendDirection.BULLISH:
                    shift_direction = Direction.SHORT
                elif structure.trend == TrendDirection.BEARISH:
                    shift_direction = Direction.LONG
            elif structure.bos_detected:
                structure_shift = True
                if structure.trend == TrendDirection.BULLISH:
                    shift_direction = Direction.LONG
                elif structure.trend == TrendDirection.BEARISH:
                    shift_direction = Direction.SHORT

        # Signal 3: Rejection candles
        rejection_candle, rejection_direction = self._detect_rejection(
            opens, highs, lows, closes
        )

        # Signal 4: Trend continuation
        continuation, continuation_direction = self._detect_continuation(
            df, structure
        )

        # Combine signals to infer direction
        direction = self._synthesize_intent(
            sweep_detected, sweep_direction,
            structure_shift, shift_direction,
            rejection_candle, rejection_direction,
            continuation, continuation_direction,
        )

        # Score
        score = self._calculate_score(
            sweep_detected, structure_shift, rejection_candle, continuation
        )

        return IntentResult(
            sweep_detected=sweep_detected,
            structure_shift=structure_shift,
            rejection_candle=rejection_candle,
            continuation=continuation,
            inferred_direction=direction,
            score=score,
        )

    def _detect_rejection(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> tuple:
        """
        Detect rejection candles (pin bars / hammer / shooting star).
        Long upper wick = bearish rejection.
        Long lower wick = bullish rejection.
        """
        # Check last 3 candles
        for i in range(max(0, len(opens) - 3), len(opens)):
            body_top = max(opens[i], closes[i])
            body_bottom = min(opens[i], closes[i])
            body_size = body_top - body_bottom
            upper_wick = highs[i] - body_top
            lower_wick = body_bottom - lows[i]
            total_range = highs[i] - lows[i]

            if total_range == 0:
                continue

            # Bearish rejection: upper wick > 60% of range, small body
            if upper_wick / total_range > 0.6 and body_size / total_range < 0.3:
                return True, Direction.SHORT

            # Bullish rejection: lower wick > 60% of range, small body
            if lower_wick / total_range > 0.6 and body_size / total_range < 0.3:
                return True, Direction.LONG

        return False, Direction.WAIT

    def _detect_continuation(
        self,
        df: pd.DataFrame,
        structure: Optional[StructureResult],
    ) -> tuple:
        """
        Detect trend continuation behavior:
        - Series of candles closing in trend direction
        - Pullback followed by trend resumption
        """
        if len(df) < 5:
            return False, Direction.WAIT

        closes = df["Close"].values
        recent = closes[-5:]

        # Count consecutive closes in same direction
        bullish_count = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i - 1])
        bearish_count = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i - 1])

        trend = TrendDirection.SIDEWAYS
        if structure:
            trend = structure.trend

        # Strong continuation: 4 of 5 candles in trend direction
        if trend == TrendDirection.BULLISH and bullish_count >= 3:
            return True, Direction.LONG
        if trend == TrendDirection.BEARISH and bearish_count >= 3:
            return True, Direction.SHORT

        # Pullback continuation: 2 against, then 2 with trend
        if len(recent) >= 4:
            if trend == TrendDirection.BULLISH:
                pullback = recent[-4] > recent[-3] > recent[-2]  # Down
                resumption = recent[-1] > recent[-2]  # Up
                if pullback and resumption:
                    return True, Direction.LONG
            elif trend == TrendDirection.BEARISH:
                pullback = recent[-4] < recent[-3] < recent[-2]  # Up
                resumption = recent[-1] < recent[-2]  # Down
                if pullback and resumption:
                    return True, Direction.SHORT

        return False, Direction.WAIT

    def _synthesize_intent(
        self,
        sweep_detected: bool, sweep_dir: Direction,
        structure_shift: bool, shift_dir: Direction,
        rejection: bool, rejection_dir: Direction,
        continuation: bool, continuation_dir: Direction,
    ) -> Direction:
        """Combine all signals into a single direction."""
        votes = {Direction.LONG: 0, Direction.SHORT: 0, Direction.WAIT: 0}

        # Sweep + structure shift is the strongest combo
        if sweep_detected and sweep_dir != Direction.WAIT:
            votes[sweep_dir] += 3

        if structure_shift and shift_dir != Direction.WAIT:
            votes[shift_dir] += 2

        if rejection and rejection_dir != Direction.WAIT:
            votes[rejection_dir] += 2

        if continuation and continuation_dir != Direction.WAIT:
            votes[continuation_dir] += 1

        # Need meaningful agreement
        long_score = votes[Direction.LONG]
        short_score = votes[Direction.SHORT]

        if long_score > short_score and long_score >= 2:
            return Direction.LONG
        elif short_score > long_score and short_score >= 2:
            return Direction.SHORT

        return Direction.WAIT

    def _calculate_score(
        self,
        sweep: bool,
        structure_shift: bool,
        rejection: bool,
        continuation: bool,
    ) -> float:
        """Calculate intent score 0-100."""
        score = 0.0

        if sweep:
            score += 30.0
        if structure_shift:
            score += 30.0
        if rejection:
            score += 20.0
        if continuation:
            score += 20.0

        return min(score, 100.0)
