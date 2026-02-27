"""
Phase 4 — Simplified Elliott Wave Engine
Rule-based impulse/corrective wave detection with
extension recognition and invalidation rules.
Degrees: Primary, Intermediate, Minor.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from market_intelligence.config import ELLIOTT_CFG, ElliottConfig
from market_intelligence.models import (
    ElliottResult,
    SwingPoint,
    SwingType,
    WaveDegree,
    WaveLabel,
    WaveType,
)

logger = logging.getLogger(__name__)


class ElliottWaveEngine:
    """
    Simplified Elliott Wave analyzer.
    Detects impulse (5-wave) and corrective (ABC) patterns
    using deterministic rules.
    """

    def __init__(self, config: Optional[ElliottConfig] = None):
        self.cfg = config or ELLIOTT_CFG

    def analyze(
        self,
        df: pd.DataFrame,
        swing_points: Optional[List[SwingPoint]] = None,
        degree: WaveDegree = WaveDegree.MINOR,
    ) -> ElliottResult:
        """
        Analyze price data for Elliott Wave patterns.

        Args:
            df: OHLCV DataFrame
            swing_points: Pre-computed swing points (optional)
            degree: Wave degree to analyze at

        Returns:
            ElliottResult with wave type, labels, validity
        """
        if df.empty or len(df) < self.cfg.min_wave_bars * 3:
            return ElliottResult(degree=degree)

        if swing_points is None:
            swing_points = self._compute_swings(df)

        if len(swing_points) < 5:
            return ElliottResult(degree=degree)

        # Try impulse detection first
        impulse_result = self._detect_impulse(df, swing_points, degree)
        if impulse_result.is_valid and impulse_result.wave_type == WaveType.IMPULSE:
            return impulse_result

        # Try corrective detection
        corrective_result = self._detect_corrective(df, swing_points, degree)
        if corrective_result.is_valid and corrective_result.wave_type == WaveType.CORRECTIVE:
            return corrective_result

        # No clear wave pattern
        return ElliottResult(degree=degree, score=10.0)

    def _compute_swings(self, df: pd.DataFrame, lookback: int = 5) -> List[SwingPoint]:
        """Compute swing highs and lows from price data."""
        highs = df["High"].values
        lows = df["Low"].values
        points: List[SwingPoint] = []

        for i in range(lookback, len(df) - lookback):
            # Swing high
            if highs[i] == max(highs[i - lookback:i + lookback + 1]):
                points.append(SwingPoint(
                    index=i, price=highs[i],
                    swing_type=SwingType.HH,
                    bar_date=str(df.index[i]),
                ))
            # Swing low
            if lows[i] == min(lows[i - lookback:i + lookback + 1]):
                points.append(SwingPoint(
                    index=i, price=lows[i],
                    swing_type=SwingType.LL,
                    bar_date=str(df.index[i]),
                ))

        points.sort(key=lambda sp: sp.index)
        return points

    def _detect_impulse(
        self,
        df: pd.DataFrame,
        swing_points: List[SwingPoint],
        degree: WaveDegree,
    ) -> ElliottResult:
        """
        Detect 5-wave impulse pattern.

        Rules:
        - Wave 2 cannot retrace beyond start of Wave 1
        - Wave 3 cannot be the shortest impulse wave
        - Wave 4 cannot overlap Wave 1 price territory
        """
        # We need alternating high/low swings to map waves
        # Separate into highs and lows
        swing_highs = [sp for sp in swing_points if sp.swing_type in (SwingType.HH, SwingType.LH)]
        swing_lows = [sp for sp in swing_points if sp.swing_type in (SwingType.HL, SwingType.LL)]

        # Try bullish impulse: low-high-low-high-low-high (0-1-2-3-4-5)
        bullish = self._try_bullish_impulse(swing_highs, swing_lows, df, degree)
        if bullish.is_valid and bullish.wave_type == WaveType.IMPULSE:
            return bullish

        # Try bearish impulse: high-low-high-low-high-low (0-1-2-3-4-5)
        bearish = self._try_bearish_impulse(swing_highs, swing_lows, df, degree)
        if bearish.is_valid and bearish.wave_type == WaveType.IMPULSE:
            return bearish

        return ElliottResult(degree=degree)

    def _try_bullish_impulse(
        self,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
        df: pd.DataFrame,
        degree: WaveDegree,
    ) -> ElliottResult:
        """Attempt to fit a bullish 5-wave impulse."""
        if len(swing_lows) < 3 or len(swing_highs) < 2:
            return ElliottResult(degree=degree)

        # Use the last significant swings
        # Wave pattern: L0 -> H1 -> L2 -> H3 -> L4 -> H5
        lows = swing_lows[-3:]   # points 0, 2, 4
        highs = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs[-2:]

        if len(highs) < 2:
            return ElliottResult(degree=degree)

        # Map wave points
        w0 = lows[0].price   # Start
        w1 = highs[0].price  # Wave 1 top
        w2 = lows[1].price   # Wave 2 bottom
        w3 = highs[1].price  # Wave 3 top
        w4 = lows[2].price if len(lows) > 2 else None
        w5 = highs[2].price if len(highs) > 2 else None

        # Validate rules
        labels: List[WaveLabel] = []
        is_valid = True
        is_extended = False

        # Rule 1: Wave 2 cannot retrace beyond Wave 0
        if w2 <= w0:
            is_valid = False

        # Rule 2: Wave 3 end must be above Wave 1
        if w3 <= w1:
            is_valid = False

        # Wave sizes
        wave1_size = w1 - w0
        wave3_size = w3 - w2

        # Rule 3: Wave 3 cannot be shortest (check vs wave 1)
        if wave1_size > 0 and wave3_size > 0:
            ratio = wave3_size / wave1_size
            if ratio < self.cfg.wave3_min_ratio:
                is_valid = False
            if ratio > self.cfg.wave3_max_ratio:
                is_extended = True

        # Rule 4: Wave 4 cannot overlap Wave 1 territory
        if w4 is not None and w4 <= w1:
            is_valid = False

        if is_valid:
            labels = [
                WaveLabel(index=lows[0].index, price=w0, label="0", degree=degree),
                WaveLabel(index=highs[0].index, price=w1, label="1", degree=degree),
                WaveLabel(index=lows[1].index, price=w2, label="2", degree=degree),
                WaveLabel(index=highs[1].index, price=w3, label="3", degree=degree),
            ]
            if w4 is not None:
                labels.append(WaveLabel(index=lows[2].index, price=w4, label="4", degree=degree))
            if w5 is not None and len(highs) > 2:
                labels.append(WaveLabel(index=highs[2].index, price=w5, label="5", degree=degree))

        # Determine current wave
        current_wave = self._determine_current_wave(labels, df)

        # Invalidation level
        invalidation = w0 if is_valid else None

        # Score
        score = self._score_impulse(is_valid, is_extended, wave1_size, wave3_size, labels)

        return ElliottResult(
            wave_type=WaveType.IMPULSE if is_valid else None,
            current_wave=current_wave,
            degree=degree,
            labels=labels,
            is_extended=is_extended,
            is_valid=is_valid,
            invalidation_level=invalidation,
            score=score,
        )

    def _try_bearish_impulse(
        self,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
        df: pd.DataFrame,
        degree: WaveDegree,
    ) -> ElliottResult:
        """Attempt to fit a bearish 5-wave impulse."""
        if len(swing_highs) < 3 or len(swing_lows) < 2:
            return ElliottResult(degree=degree)

        highs = swing_highs[-3:]
        lows = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows[-2:]

        if len(lows) < 2:
            return ElliottResult(degree=degree)

        # Wave pattern: H0 -> L1 -> H2 -> L3 -> H4 -> L5
        w0 = highs[0].price
        w1 = lows[0].price
        w2 = highs[1].price
        w3 = lows[1].price
        w4 = highs[2].price if len(highs) > 2 else None
        w5 = lows[2].price if len(lows) > 2 else None

        labels: List[WaveLabel] = []
        is_valid = True
        is_extended = False

        # Rule 1: Wave 2 cannot retrace beyond Wave 0
        if w2 >= w0:
            is_valid = False

        # Rule 2: Wave 3 must go below Wave 1
        if w3 >= w1:
            is_valid = False

        wave1_size = w0 - w1
        wave3_size = w2 - w3

        if wave1_size > 0 and wave3_size > 0:
            ratio = wave3_size / wave1_size
            if ratio < self.cfg.wave3_min_ratio:
                is_valid = False
            if ratio > self.cfg.wave3_max_ratio:
                is_extended = True

        # Rule 4: Wave 4 cannot overlap Wave 1
        if w4 is not None and w4 >= w1:
            is_valid = False

        if is_valid:
            labels = [
                WaveLabel(index=highs[0].index, price=w0, label="0", degree=degree),
                WaveLabel(index=lows[0].index, price=w1, label="1", degree=degree),
                WaveLabel(index=highs[1].index, price=w2, label="2", degree=degree),
                WaveLabel(index=lows[1].index, price=w3, label="3", degree=degree),
            ]
            if w4 is not None:
                labels.append(WaveLabel(index=highs[2].index, price=w4, label="4", degree=degree))
            if w5 is not None and len(lows) > 2:
                labels.append(WaveLabel(index=lows[2].index, price=w5, label="5", degree=degree))

        current_wave = self._determine_current_wave(labels, df)
        invalidation = w0 if is_valid else None
        score = self._score_impulse(is_valid, is_extended, wave1_size, wave3_size, labels)

        return ElliottResult(
            wave_type=WaveType.IMPULSE if is_valid else None,
            current_wave=current_wave,
            degree=degree,
            labels=labels,
            is_extended=is_extended,
            is_valid=is_valid,
            invalidation_level=invalidation,
            score=score,
        )

    def _detect_corrective(
        self,
        df: pd.DataFrame,
        swing_points: List[SwingPoint],
        degree: WaveDegree,
    ) -> ElliottResult:
        """
        Detect ABC corrective pattern.
        """
        swing_highs = [sp for sp in swing_points if sp.swing_type in (SwingType.HH, SwingType.LH)]
        swing_lows = [sp for sp in swing_points if sp.swing_type in (SwingType.HL, SwingType.LL)]

        # Try bearish ABC (after uptrend)
        if len(swing_highs) >= 2 and len(swing_lows) >= 1:
            result = self._try_abc_correction(swing_highs, swing_lows, df, degree, bullish_prior=True)
            if result.is_valid:
                return result

        # Try bullish ABC (after downtrend)
        if len(swing_lows) >= 2 and len(swing_highs) >= 1:
            result = self._try_abc_correction(swing_highs, swing_lows, df, degree, bullish_prior=False)
            if result.is_valid:
                return result

        return ElliottResult(degree=degree)

    def _try_abc_correction(
        self,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
        df: pd.DataFrame,
        degree: WaveDegree,
        bullish_prior: bool,
    ) -> ElliottResult:
        """Try to fit an ABC correction."""
        labels: List[WaveLabel] = []
        is_valid = True

        if bullish_prior:
            # After uptrend: A down, B up, C down
            if len(swing_highs) < 1 or len(swing_lows) < 2:
                return ElliottResult(degree=degree)

            top = swing_highs[-1]  # Start of correction
            a_low = swing_lows[-2] if len(swing_lows) >= 2 else swing_lows[-1]

            # B should retrace 38.2%-78.6% of A
            # Find a swing high between A low and the end
            b_candidates = [sh for sh in swing_highs if sh.index > a_low.index]
            if not b_candidates:
                return ElliottResult(degree=degree)

            b_high = b_candidates[0]
            wave_a_size = top.price - a_low.price

            if wave_a_size > 0:
                b_retrace = (b_high.price - a_low.price) / wave_a_size
                if b_retrace < self.cfg.correction_min_ratio or b_retrace > self.cfg.correction_max_ratio:
                    is_valid = False

            # C low
            c_candidates = [sl for sl in swing_lows if sl.index > b_high.index]
            c_low = c_candidates[0] if c_candidates else None

            if is_valid:
                labels = [
                    WaveLabel(index=a_low.index, price=a_low.price, label="A", degree=degree),
                    WaveLabel(index=b_high.index, price=b_high.price, label="B", degree=degree),
                ]
                if c_low:
                    labels.append(WaveLabel(index=c_low.index, price=c_low.price, label="C", degree=degree))
        else:
            # After downtrend: A up, B down, C up
            if len(swing_lows) < 1 or len(swing_highs) < 2:
                return ElliottResult(degree=degree)

            bottom = swing_lows[-1]
            a_high = swing_highs[-2] if len(swing_highs) >= 2 else swing_highs[-1]

            b_candidates = [sl for sl in swing_lows if sl.index > a_high.index]
            if not b_candidates:
                return ElliottResult(degree=degree)

            b_low = b_candidates[0]
            wave_a_size = a_high.price - bottom.price

            if wave_a_size > 0:
                b_retrace = (a_high.price - b_low.price) / wave_a_size
                if b_retrace < self.cfg.correction_min_ratio or b_retrace > self.cfg.correction_max_ratio:
                    is_valid = False

            c_candidates = [sh for sh in swing_highs if sh.index > b_low.index]
            c_high = c_candidates[0] if c_candidates else None

            if is_valid:
                labels = [
                    WaveLabel(index=a_high.index, price=a_high.price, label="A", degree=degree),
                    WaveLabel(index=b_low.index, price=b_low.price, label="B", degree=degree),
                ]
                if c_high:
                    labels.append(WaveLabel(index=c_high.index, price=c_high.price, label="C", degree=degree))

        current_wave = self._determine_current_wave(labels, df)
        score = 40.0 if is_valid and len(labels) >= 3 else 15.0

        return ElliottResult(
            wave_type=WaveType.CORRECTIVE if is_valid else None,
            current_wave=current_wave,
            degree=degree,
            labels=labels,
            is_valid=is_valid,
            score=score,
        )

    def _determine_current_wave(self, labels: List[WaveLabel], df: pd.DataFrame) -> str:
        """Determine which wave we're currently in."""
        if not labels:
            return ""

        last_label = labels[-1]
        bar_count = len(df)

        # If the last label's index is near the end, we're at that wave
        if bar_count - last_label.index < 5:
            return last_label.label

        # Otherwise, we're past the last labeled wave
        label_map = {"0": "1", "1": "2", "2": "3", "3": "4", "4": "5", "5": "complete",
                     "A": "B", "B": "C", "C": "complete"}
        return label_map.get(last_label.label, last_label.label)

    def _score_impulse(
        self,
        is_valid: bool,
        is_extended: bool,
        wave1_size: float,
        wave3_size: float,
        labels: List[WaveLabel],
    ) -> float:
        """Score an impulse wave 0-100."""
        if not is_valid:
            return 10.0

        score = 30.0  # Base for valid impulse

        # Extended wave 3 is strong
        if is_extended:
            score += 15.0

        # More labels = more complete pattern
        score += len(labels) * 5

        # Wave 3 being significantly larger than wave 1 is ideal
        if wave1_size > 0:
            ratio = wave3_size / wave1_size
            if 1.618 <= ratio <= 2.618:
                score += 20.0
            elif 1.0 <= ratio < 1.618:
                score += 10.0

        return min(score, 100.0)
