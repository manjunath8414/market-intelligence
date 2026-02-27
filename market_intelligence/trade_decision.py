"""
Phase 11 — Trade Decision Output
Generates final trade decisions with direction, entry, targets,
stop loss, and holding period estimates.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from market_intelligence.config import TRADE_CFG, TradeConfig
from market_intelligence.models import (
    CrossMarketResult,
    Direction,
    ElliottResult,
    EntryType,
    FibonacciResult,
    IntentResult,
    LiquidityResult,
    StructureResult,
    TimingResult,
    TradeDecision,
    TradeType,
    TrendDirection,
    WaveType,
)
from market_intelligence.scoring_engine import ScoreBreakdown

logger = logging.getLogger(__name__)


class TradeDecisionEngine:
    """
    Synthesizes all analysis results into actionable trade decisions.
    """

    def __init__(self, config: Optional[TradeConfig] = None):
        self.cfg = config or TRADE_CFG

    def generate(
        self,
        symbol: str,
        df: pd.DataFrame,
        scores: ScoreBreakdown,
        structure: Optional[StructureResult] = None,
        elliott: Optional[ElliottResult] = None,
        liquidity: Optional[LiquidityResult] = None,
        fibonacci: Optional[FibonacciResult] = None,
        timing: Optional[TimingResult] = None,
        intent: Optional[IntentResult] = None,
        cross_market: Optional[CrossMarketResult] = None,
    ) -> TradeDecision:
        """
        Generate a complete trade decision.

        Args:
            symbol: Stock symbol
            df: OHLCV DataFrame
            scores: Score breakdown from scoring engine
            structure, elliott, liquidity, fibonacci, timing, intent, cross_market:
                Individual analysis results

        Returns:
            TradeDecision with all trade parameters
        """
        if df.empty:
            return TradeDecision(
                symbol=symbol,
                direction=Direction.WAIT,
                confidence=0.0,
                reasoning="Insufficient data",
            )

        current_price = df["Close"].iloc[-1]
        confidence = scores.final_confidence

        # Determine direction
        direction = self._determine_direction(structure, intent, elliott, confidence)

        # If confidence too low, WAIT
        if confidence < self.cfg.min_confidence_trade:
            return TradeDecision(
                symbol=symbol,
                direction=Direction.WAIT,
                confidence=confidence,
                reasoning=f"Confidence {confidence:.0f}% below minimum {self.cfg.min_confidence_trade}%",
                structure_score=scores.structure_raw,
                wave_score=scores.wave_raw,
                liquidity_score=scores.liquidity_raw,
                timing_score=scores.timing_raw,
                market_alignment_score=scores.market_alignment_raw,
                fibonacci_score=scores.fibonacci_raw,
            )

        # Determine entry type
        entry_type = self._determine_entry_type(structure, liquidity, timing)

        # Determine trade type and holding period
        trade_type, holding_period = self._determine_trade_type(df, timing)

        # Calculate entry zone, stop loss, and targets
        entry_zone = self._calculate_entry(current_price, direction, entry_type, fibonacci, liquidity)
        stop_loss = self._calculate_stop_loss(current_price, direction, structure, liquidity, df)
        target_1, target_2, target_3 = self._calculate_targets(
            entry_zone, stop_loss, direction, fibonacci, elliott
        )

        # Validate risk-reward
        if stop_loss and target_1 and entry_zone:
            risk = abs(entry_zone - stop_loss)
            reward = abs(target_1 - entry_zone)
            if risk > 0 and reward / risk < self.cfg.risk_reward_min:
                # Adjust: still report but note poor R:R
                pass

        # Build reasoning
        reasoning = self._build_reasoning(
            direction, entry_type, structure, elliott, liquidity, timing, intent
        )

        return TradeDecision(
            symbol=symbol,
            direction=direction,
            entry_type=entry_type,
            trade_type=trade_type,
            holding_period=holding_period,
            entry_zone=round(entry_zone, 2) if entry_zone else None,
            stop_loss=round(stop_loss, 2) if stop_loss else None,
            target_1=round(target_1, 2) if target_1 else None,
            target_2=round(target_2, 2) if target_2 else None,
            target_3=round(target_3, 2) if target_3 else None,
            confidence=round(confidence, 1),
            reasoning=reasoning,
            structure_score=scores.structure_raw,
            wave_score=scores.wave_raw,
            liquidity_score=scores.liquidity_raw,
            timing_score=scores.timing_raw,
            market_alignment_score=scores.market_alignment_raw,
            fibonacci_score=scores.fibonacci_raw,
        )

    def _determine_direction(
        self,
        structure: Optional[StructureResult],
        intent: Optional[IntentResult],
        elliott: Optional[ElliottResult],
        confidence: float,
    ) -> Direction:
        """Determine trade direction from analysis signals."""
        votes = {Direction.LONG: 0, Direction.SHORT: 0}

        # Structure trend (strongest weight)
        if structure:
            if structure.trend == TrendDirection.BULLISH:
                votes[Direction.LONG] += 3
            elif structure.trend == TrendDirection.BEARISH:
                votes[Direction.SHORT] += 3

            # BOS in trend direction
            if structure.bos_detected:
                if structure.trend == TrendDirection.BULLISH:
                    votes[Direction.LONG] += 2
                elif structure.trend == TrendDirection.BEARISH:
                    votes[Direction.SHORT] += 2

            # CHoCH suggests reversal
            if structure.choch_detected:
                if structure.trend == TrendDirection.BULLISH:
                    votes[Direction.SHORT] += 2
                elif structure.trend == TrendDirection.BEARISH:
                    votes[Direction.LONG] += 2

        # Intent
        if intent and intent.inferred_direction != Direction.WAIT:
            votes[intent.inferred_direction] += 2

        # Elliott wave
        if elliott and elliott.is_valid:
            if elliott.wave_type == WaveType.IMPULSE:
                if elliott.current_wave in ("3", "5"):
                    # In active impulse waves
                    # Assume bullish impulse for simplicity based on wave direction
                    votes[Direction.LONG] += 1
            elif elliott.wave_type == WaveType.CORRECTIVE:
                if elliott.current_wave == "C":
                    # End of correction, expect reversal
                    votes[Direction.LONG] += 1

        if votes[Direction.LONG] > votes[Direction.SHORT]:
            return Direction.LONG
        elif votes[Direction.SHORT] > votes[Direction.LONG]:
            return Direction.SHORT
        return Direction.WAIT

    def _determine_entry_type(
        self,
        structure: Optional[StructureResult],
        liquidity: Optional[LiquidityResult],
        timing: Optional[TimingResult],
    ) -> EntryType:
        """Determine entry type: Pullback, Breakout, or Reversal."""
        # Reversal if structure shift detected
        if structure and structure.choch_detected:
            return EntryType.REVERSAL

        # Breakout if timing shows velocity
        if timing and timing.breakout_velocity_met:
            return EntryType.BREAKOUT

        # Sweep reversal
        if liquidity and liquidity.sweep_detected:
            return EntryType.REVERSAL

        # Default: Pullback (safest entry)
        return EntryType.PULLBACK

    def _determine_trade_type(
        self, df: pd.DataFrame, timing: Optional[TimingResult]
    ) -> tuple:
        """Determine Intraday vs Swing and estimated holding period."""
        # Check timeframe from data frequency
        if len(df) >= 2:
            time_diff = df.index[-1] - df.index[-2]
            minutes = time_diff.total_seconds() / 60

            if minutes <= 15:
                return TradeType.INTRADAY, "1-4 Hours"
            elif minutes <= 60:
                return TradeType.INTRADAY, "4-8 Hours"

        # Daily timeframe = Swing
        return TradeType.SWING, "4-7 Days"

    def _calculate_entry(
        self,
        current_price: float,
        direction: Direction,
        entry_type: EntryType,
        fibonacci: Optional[FibonacciResult],
        liquidity: Optional[LiquidityResult],
    ) -> float:
        """Calculate entry zone."""
        if entry_type == EntryType.BREAKOUT:
            # Enter at or slightly beyond current price
            if direction == Direction.LONG:
                return current_price * 1.002  # Slight premium
            else:
                return current_price * 0.998

        if entry_type == EntryType.PULLBACK and fibonacci:
            # Look for nearest Fib retracement level
            for level in fibonacci.retracement_levels:
                if direction == Direction.LONG and level.price < current_price:
                    return level.price
                elif direction == Direction.SHORT and level.price > current_price:
                    return level.price

        # Default: current price
        return current_price

    def _calculate_stop_loss(
        self,
        current_price: float,
        direction: Direction,
        structure: Optional[StructureResult],
        liquidity: Optional[LiquidityResult],
        df: pd.DataFrame,
    ) -> Optional[float]:
        """Calculate stop loss level."""
        # Method 1: Below/above recent swing point
        if structure and structure.swing_points:
            if direction == Direction.LONG:
                recent_lows = [
                    sp.price for sp in structure.swing_points
                    if sp.swing_type.value in ("Higher Low", "Lower Low")
                ]
                if recent_lows:
                    sl = min(recent_lows[-3:])  # Recent 3 swing lows
                    # Add buffer
                    return sl * 0.995
            else:
                recent_highs = [
                    sp.price for sp in structure.swing_points
                    if sp.swing_type.value in ("Higher High", "Lower High")
                ]
                if recent_highs:
                    sl = max(recent_highs[-3:])
                    return sl * 1.005

        # Method 2: ATR-based stop
        if len(df) >= 14:
            highs = df["High"].values
            lows = df["Low"].values
            closes = df["Close"].values

            tr = np.maximum(
                highs[1:] - lows[1:],
                np.maximum(
                    np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:] - closes[:-1])
                )
            )
            atr = np.mean(tr[-14:])

            if direction == Direction.LONG:
                return current_price - (atr * 1.5)
            else:
                return current_price + (atr * 1.5)

        # Fallback: percentage-based
        if direction == Direction.LONG:
            return current_price * (1 - self.cfg.max_stop_loss_pct)
        else:
            return current_price * (1 + self.cfg.max_stop_loss_pct)

    def _calculate_targets(
        self,
        entry: Optional[float],
        stop_loss: Optional[float],
        direction: Direction,
        fibonacci: Optional[FibonacciResult],
        elliott: Optional[ElliottResult],
    ) -> tuple:
        """Calculate T1, T2, T3 targets."""
        if entry is None or stop_loss is None:
            return None, None, None

        risk = abs(entry - stop_loss)

        if direction == Direction.LONG:
            # Use Fibonacci extensions if available
            if fibonacci and fibonacci.extension_levels:
                ext_prices = sorted([lv.price for lv in fibonacci.extension_levels if lv.price > entry])
                if len(ext_prices) >= 3:
                    return ext_prices[0], ext_prices[1], ext_prices[2]
                elif len(ext_prices) >= 2:
                    return ext_prices[0], ext_prices[1], entry + risk * 3.0
                elif len(ext_prices) >= 1:
                    return ext_prices[0], entry + risk * 2.0, entry + risk * 3.0

            # R:R based targets
            return (
                entry + risk * 1.5,
                entry + risk * 2.5,
                entry + risk * 3.5,
            )
        else:
            if fibonacci and fibonacci.extension_levels:
                ext_prices = sorted([lv.price for lv in fibonacci.extension_levels if lv.price < entry], reverse=True)
                if len(ext_prices) >= 3:
                    return ext_prices[0], ext_prices[1], ext_prices[2]
                elif len(ext_prices) >= 2:
                    return ext_prices[0], ext_prices[1], entry - risk * 3.0
                elif len(ext_prices) >= 1:
                    return ext_prices[0], entry - risk * 2.0, entry - risk * 3.0

            return (
                entry - risk * 1.5,
                entry - risk * 2.5,
                entry - risk * 3.5,
            )

    def _build_reasoning(
        self,
        direction: Direction,
        entry_type: EntryType,
        structure: Optional[StructureResult],
        elliott: Optional[ElliottResult],
        liquidity: Optional[LiquidityResult],
        timing: Optional[TimingResult],
        intent: Optional[IntentResult],
    ) -> str:
        """Build human-readable reasoning for the trade decision."""
        parts = []

        if structure:
            parts.append(f"Trend: {structure.trend.value}")
            if structure.bos_detected:
                parts.append("Break of Structure confirmed")
            if structure.choch_detected:
                parts.append("Change of Character detected")

        if elliott and elliott.is_valid:
            parts.append(f"Wave: {elliott.wave_type.value if elliott.wave_type else 'N/A'} "
                        f"(current: {elliott.current_wave})")

        if liquidity and liquidity.sweep_detected:
            parts.append(f"Liquidity sweep ({liquidity.sweep_direction.value if liquidity.sweep_direction else 'N/A'})")

        if timing:
            if timing.atr_contracted:
                parts.append("ATR contracted (expansion likely)")
            if timing.breakout_velocity_met:
                parts.append("Breakout velocity confirmed")

        if intent and intent.inferred_direction != Direction.WAIT:
            parts.append(f"Intent: {intent.inferred_direction.value}")

        return " | ".join(parts) if parts else "Standard analysis"
