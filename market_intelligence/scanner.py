"""
Phase 12 — Scanner
Scans selected universe for specific trade setups:
- Wave Expansion Setup
- Near Support
- Breakout Setup
- Trend Continuation
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from market_intelligence.config import NIFTY50_SYMBOLS
from market_intelligence.models import (
    Direction,
    ScannerSetup,
    ScanResult,
    TrendDirection,
)
from market_intelligence.market_structure import MarketStructureAnalyzer
from market_intelligence.liquidity_model import LiquidityAnalyzer
from market_intelligence.timing_engine import TimingEngine

logger = logging.getLogger(__name__)


class Scanner:
    """
    Scans a universe of stocks for specific trade setups.
    Designed for efficiency — avoids full analysis on every stock.
    """

    def __init__(self):
        self.structure_analyzer = MarketStructureAnalyzer()
        self.liquidity_analyzer = LiquidityAnalyzer()
        self.timing_engine = TimingEngine()

    def scan(
        self,
        data: Dict[str, pd.DataFrame],
        setups: Optional[List[ScannerSetup]] = None,
        min_confidence: float = 40.0,
    ) -> List[ScanResult]:
        """
        Scan multiple stocks for trade setups.

        Args:
            data: Dict of symbol -> OHLCV DataFrame
            setups: Which setups to scan for (default: all)
            min_confidence: Minimum confidence to include in results

        Returns:
            List of ScanResult sorted by confidence (descending)
        """
        if setups is None:
            setups = list(ScannerSetup)

        results: List[ScanResult] = []

        for symbol, df in data.items():
            if df.empty or len(df) < 30:
                continue

            for setup in setups:
                try:
                    result = self._check_setup(symbol, df, setup)
                    if result and result.confidence >= min_confidence:
                        results.append(result)
                except Exception as e:
                    logger.warning(f"Scanner error for {symbol}/{setup.value}: {e}")

        # Sort by confidence descending
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    def _check_setup(
        self, symbol: str, df: pd.DataFrame, setup: ScannerSetup
    ) -> Optional[ScanResult]:
        """Check if a stock matches a specific setup."""
        if setup == ScannerSetup.WAVE_EXPANSION:
            return self._scan_wave_expansion(symbol, df)
        elif setup == ScannerSetup.NEAR_SUPPORT:
            return self._scan_near_support(symbol, df)
        elif setup == ScannerSetup.BREAKOUT:
            return self._scan_breakout(symbol, df)
        elif setup == ScannerSetup.TREND_CONTINUATION:
            return self._scan_trend_continuation(symbol, df)
        return None

    def _scan_wave_expansion(self, symbol: str, df: pd.DataFrame) -> Optional[ScanResult]:
        """
        Wave Expansion: ATR is contracting, suggesting imminent expansion.
        Combined with clear trend structure.
        """
        timing = self.timing_engine.analyze(df)
        structure = self.structure_analyzer.analyze(df)

        if not timing.atr_contracted and not timing.range_compressed:
            return None

        confidence = 0.0
        direction = Direction.WAIT

        if timing.atr_contracted:
            confidence += 25.0
        if timing.range_compressed:
            confidence += 20.0

        if structure.trend == TrendDirection.BULLISH:
            direction = Direction.LONG
            confidence += 20.0
        elif structure.trend == TrendDirection.BEARISH:
            direction = Direction.SHORT
            confidence += 20.0
        else:
            confidence += 5.0

        if structure.bos_detected:
            confidence += 15.0

        return ScanResult(
            symbol=symbol,
            setup=ScannerSetup.WAVE_EXPANSION,
            direction=direction,
            confidence=min(confidence, 100),
            note=f"ATR contracted: {timing.atr_contracted}, Range compressed: {timing.range_compressed}",
        )

    def _scan_near_support(self, symbol: str, df: pd.DataFrame) -> Optional[ScanResult]:
        """
        Near Support: Price is near a significant support/liquidity level.
        """
        liquidity = self.liquidity_analyzer.analyze(df)
        current_price = df["Close"].iloc[-1]

        # Find nearest support
        all_supports = liquidity.support_levels + liquidity.equal_lows
        if not all_supports:
            return None

        # Filter supports below current price
        below = [s for s in all_supports if s < current_price]
        if not below:
            return None

        nearest = max(below)
        dist_pct = (current_price - nearest) / current_price

        if dist_pct > 0.03:  # More than 3% away
            return None

        confidence = 30.0
        if dist_pct < 0.01:
            confidence += 30.0
        elif dist_pct < 0.02:
            confidence += 20.0
        else:
            confidence += 10.0

        # Bonus for multiple support levels nearby
        nearby_supports = [s for s in all_supports if abs(s - nearest) / nearest < 0.01]
        confidence += min(len(nearby_supports) * 10, 20)

        return ScanResult(
            symbol=symbol,
            setup=ScannerSetup.NEAR_SUPPORT,
            direction=Direction.LONG,
            confidence=min(confidence, 100),
            note=f"Support at {nearest:.2f}, price {current_price:.2f} ({dist_pct*100:.1f}% above)",
        )

    def _scan_breakout(self, symbol: str, df: pd.DataFrame) -> Optional[ScanResult]:
        """
        Breakout Setup: Price breaking above resistance or below support
        with velocity.
        """
        timing = self.timing_engine.analyze(df)
        liquidity = self.liquidity_analyzer.analyze(df)
        current_price = df["Close"].iloc[-1]

        if not timing.breakout_velocity_met:
            return None

        confidence = 35.0
        direction = Direction.WAIT

        # Check if breaking above resistance
        for level in liquidity.resistance_levels + liquidity.equal_highs:
            if current_price > level and (current_price - level) / level < 0.02:
                direction = Direction.LONG
                confidence += 25.0
                break

        # Check if breaking below support
        if direction == Direction.WAIT:
            for level in liquidity.support_levels + liquidity.equal_lows:
                if current_price < level and (level - current_price) / level < 0.02:
                    direction = Direction.SHORT
                    confidence += 25.0
                    break

        if direction == Direction.WAIT:
            return None

        if timing.atr_contracted:
            confidence += 15.0  # Breakout from contraction is stronger

        structure = self.structure_analyzer.analyze(df)
        if structure.bos_detected:
            confidence += 15.0

        return ScanResult(
            symbol=symbol,
            setup=ScannerSetup.BREAKOUT,
            direction=direction,
            confidence=min(confidence, 100),
            note=f"Breakout with velocity, BOS: {structure.bos_detected}",
        )

    def _scan_trend_continuation(self, symbol: str, df: pd.DataFrame) -> Optional[ScanResult]:
        """
        Trend Continuation: Strong trend with a pullback that's resuming.
        """
        structure = self.structure_analyzer.analyze(df)

        if structure.trend == TrendDirection.SIDEWAYS:
            return None

        closes = df["Close"].values
        if len(closes) < 10:
            return None

        confidence = 30.0
        direction = Direction.LONG if structure.trend == TrendDirection.BULLISH else Direction.SHORT

        # Check for pullback + resumption in last 5 candles
        recent = closes[-5:]

        if direction == Direction.LONG:
            # Pullback: at least one down candle, then up
            had_pullback = any(recent[i] < recent[i - 1] for i in range(1, 4))
            resuming = recent[-1] > recent[-2]
            if had_pullback and resuming:
                confidence += 25.0
            else:
                return None
        else:
            had_pullback = any(recent[i] > recent[i - 1] for i in range(1, 4))
            resuming = recent[-1] < recent[-2]
            if had_pullback and resuming:
                confidence += 25.0
            else:
                return None

        # Trend strength
        if structure.phase.value == "TRENDING":
            confidence += 15.0

        if structure.bos_detected:
            confidence += 15.0

        return ScanResult(
            symbol=symbol,
            setup=ScannerSetup.TREND_CONTINUATION,
            direction=direction,
            confidence=min(confidence, 100),
            note=f"Trend: {structure.trend.value}, Phase: {structure.phase.value}",
        )
