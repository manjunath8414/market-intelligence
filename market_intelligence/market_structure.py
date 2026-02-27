"""
Phase 2 — Market Structure Detection
Detects HH, HL, LH, LL, trend direction, range vs trend,
Break of Structure (BOS), and Change of Character (CHoCH).
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from market_intelligence.config import MARKET_STRUCTURE_CFG, MarketStructureConfig
from market_intelligence.models import (
    MarketPhase,
    StructureResult,
    SwingPoint,
    SwingType,
    TrendDirection,
)

logger = logging.getLogger(__name__)


class MarketStructureAnalyzer:
    """
    Identifies market structure from price action.
    """

    def __init__(self, config: Optional[MarketStructureConfig] = None):
        self.cfg = config or MARKET_STRUCTURE_CFG

    def analyze(self, df: pd.DataFrame) -> StructureResult:
        """
        Full market structure analysis.

        Args:
            df: OHLCV DataFrame with columns Open, High, Low, Close

        Returns:
            StructureResult with trend, phase, swing points, BOS, CHoCH
        """
        if df.empty or len(df) < self.cfg.swing_lookback * 3:
            return StructureResult(
                trend=TrendDirection.SIDEWAYS,
                phase=MarketPhase.RANGING,
            )

        highs = df["High"].values
        lows = df["Low"].values
        closes = df["Close"].values

        # Step 1: Identify swing points
        swing_highs = self._find_swing_highs(highs)
        swing_lows = self._find_swing_lows(lows)

        # Step 2: Label swing points (HH, HL, LH, LL)
        swing_points = self._label_swings(swing_highs, swing_lows, highs, lows, df)

        # Step 3: Determine trend direction
        trend = self._determine_trend(swing_points)

        # Step 4: Detect range vs trend
        phase = self._detect_phase(df, swing_points)

        # Step 5: Detect Break of Structure
        bos_detected, bos_level = self._detect_bos(swing_points, closes)

        # Step 6: Detect Change of Character
        choch_detected, choch_level = self._detect_choch(swing_points, closes)

        # Score the structure
        score = self._calculate_score(trend, phase, bos_detected, choch_detected, swing_points)

        return StructureResult(
            trend=trend,
            phase=phase,
            swing_points=swing_points,
            bos_detected=bos_detected,
            bos_level=bos_level,
            choch_detected=choch_detected,
            choch_level=choch_level,
            score=score,
        )

    def _find_swing_highs(self, highs: np.ndarray) -> List[int]:
        """Find indices of swing highs using lookback period."""
        lb = self.cfg.swing_lookback
        swing_indices = []
        for i in range(lb, len(highs) - lb):
            left = highs[i - lb:i]
            right = highs[i + 1:i + lb + 1]
            if highs[i] >= max(left) and highs[i] >= max(right):
                swing_indices.append(i)
        return swing_indices

    def _find_swing_lows(self, lows: np.ndarray) -> List[int]:
        """Find indices of swing lows using lookback period."""
        lb = self.cfg.swing_lookback
        swing_indices = []
        for i in range(lb, len(lows) - lb):
            left = lows[i - lb:i]
            right = lows[i + 1:i + lb + 1]
            if lows[i] <= min(left) and lows[i] <= min(right):
                swing_indices.append(i)
        return swing_indices

    def _label_swings(
        self,
        swing_high_indices: List[int],
        swing_low_indices: List[int],
        highs: np.ndarray,
        lows: np.ndarray,
        df: pd.DataFrame,
    ) -> List[SwingPoint]:
        """Label each swing point as HH, HL, LH, or LL."""
        points: List[SwingPoint] = []

        # Merge and sort all swing points
        all_swings: List[Tuple[int, str]] = []
        for idx in swing_high_indices:
            all_swings.append((idx, "high"))
        for idx in swing_low_indices:
            all_swings.append((idx, "low"))
        all_swings.sort(key=lambda x: x[0])

        prev_high: Optional[float] = None
        prev_low: Optional[float] = None

        for idx, swing_kind in all_swings:
            bar_date = str(df.index[idx]) if idx < len(df) else ""

            if swing_kind == "high":
                price = highs[idx]
                if prev_high is None:
                    swing_type = SwingType.HH  # First high, assume HH
                elif price > prev_high:
                    swing_type = SwingType.HH
                else:
                    swing_type = SwingType.LH
                prev_high = price
            else:
                price = lows[idx]
                if prev_low is None:
                    swing_type = SwingType.HL  # First low, assume HL
                elif price > prev_low:
                    swing_type = SwingType.HL
                else:
                    swing_type = SwingType.LL
                prev_low = price

            points.append(SwingPoint(
                index=idx,
                price=price,
                swing_type=swing_type,
                bar_date=bar_date,
            ))

        return points

    def _determine_trend(self, swing_points: List[SwingPoint]) -> TrendDirection:
        """Determine trend from recent swing point sequence."""
        if len(swing_points) < 4:
            return TrendDirection.SIDEWAYS

        recent = swing_points[-6:]  # Look at last 6 swing points

        hh_count = sum(1 for sp in recent if sp.swing_type == SwingType.HH)
        hl_count = sum(1 for sp in recent if sp.swing_type == SwingType.HL)
        lh_count = sum(1 for sp in recent if sp.swing_type == SwingType.LH)
        ll_count = sum(1 for sp in recent if sp.swing_type == SwingType.LL)

        bullish = hh_count + hl_count
        bearish = lh_count + ll_count

        if bullish >= bearish + 2:
            return TrendDirection.BULLISH
        elif bearish >= bullish + 2:
            return TrendDirection.BEARISH
        return TrendDirection.SIDEWAYS

    def _detect_phase(self, df: pd.DataFrame, swing_points: List[SwingPoint]) -> MarketPhase:
        """Detect if market is trending or ranging using ATR-based analysis."""
        if len(df) < 20:
            return MarketPhase.RANGING

        # Calculate ATR
        high = df["High"].values
        low = df["Low"].values
        close = df["Close"].values

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)

        # Range: difference between highest high and lowest low of recent swings
        if len(swing_points) >= 4:
            recent_prices = [sp.price for sp in swing_points[-8:]]
            price_range = max(recent_prices) - min(recent_prices)
            avg_price = np.mean(recent_prices)

            # If range is small relative to ATR, it's ranging
            if price_range < atr * self.cfg.range_threshold_atr:
                return MarketPhase.RANGING

        return MarketPhase.TRENDING

    def _detect_bos(
        self, swing_points: List[SwingPoint], closes: np.ndarray
    ) -> Tuple[bool, Optional[float]]:
        """
        Break of Structure: Price breaks a significant swing level
        in the direction of the trend.
        """
        if len(swing_points) < 3:
            return False, None

        current_close = closes[-1]

        # Find the last significant swing high and low
        last_swing_high = None
        last_swing_low = None

        for sp in reversed(swing_points):
            if sp.swing_type in (SwingType.HH, SwingType.LH) and last_swing_high is None:
                last_swing_high = sp
            elif sp.swing_type in (SwingType.HL, SwingType.LL) and last_swing_low is None:
                last_swing_low = sp
            if last_swing_high and last_swing_low:
                break

        # Bullish BOS: close above last swing high
        if last_swing_high and current_close > last_swing_high.price:
            return True, last_swing_high.price

        # Bearish BOS: close below last swing low
        if last_swing_low and current_close < last_swing_low.price:
            return True, last_swing_low.price

        return False, None

    def _detect_choch(
        self, swing_points: List[SwingPoint], closes: np.ndarray
    ) -> Tuple[bool, Optional[float]]:
        """
        Change of Character: Trend reversal signal.
        In an uptrend, price breaks below a higher low.
        In a downtrend, price breaks above a lower high.
        """
        if len(swing_points) < 4:
            return False, None

        current_close = closes[-1]
        recent = swing_points[-6:]

        # Check for bullish-to-bearish CHoCH
        higher_lows = [sp for sp in recent if sp.swing_type == SwingType.HL]
        if higher_lows:
            last_hl = higher_lows[-1]
            if current_close < last_hl.price:
                return True, last_hl.price

        # Check for bearish-to-bullish CHoCH
        lower_highs = [sp for sp in recent if sp.swing_type == SwingType.LH]
        if lower_highs:
            last_lh = lower_highs[-1]
            if current_close > last_lh.price:
                return True, last_lh.price

        return False, None

    def _calculate_score(
        self,
        trend: TrendDirection,
        phase: MarketPhase,
        bos_detected: bool,
        choch_detected: bool,
        swing_points: List[SwingPoint],
    ) -> float:
        """Calculate structure score 0-100."""
        score = 0.0

        # Clear trend = higher score
        if trend in (TrendDirection.BULLISH, TrendDirection.BEARISH):
            score += 30.0
        else:
            score += 10.0

        # Trending phase bonus
        if phase == MarketPhase.TRENDING:
            score += 20.0
        else:
            score += 5.0

        # BOS is a strong signal
        if bos_detected:
            score += 25.0

        # CHoCH is notable (could be reversal)
        if choch_detected:
            score += 15.0

        # Consistent swing points
        if len(swing_points) >= 4:
            recent = swing_points[-4:]
            types = [sp.swing_type for sp in recent]

            # All bullish (HH + HL) or all bearish (LH + LL)
            bullish_types = {SwingType.HH, SwingType.HL}
            bearish_types = {SwingType.LH, SwingType.LL}

            if all(t in bullish_types for t in types) or all(t in bearish_types for t in types):
                score += 10.0

        return min(score, 100.0)
