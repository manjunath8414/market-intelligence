"""
Phase 10 — Scoring Engine
Assigns weighted scores from all analysis modules
and computes a final confidence score (0-100).
"""

import logging
from dataclasses import dataclass
from typing import Optional

from market_intelligence.config import SCORING_WEIGHTS, ScoringWeights
from market_intelligence.models import (
    CrossMarketResult,
    ElliottResult,
    FibonacciResult,
    GannResult,
    IntentResult,
    LiquidityResult,
    StructureResult,
    TimingResult,
)

logger = logging.getLogger(__name__)


@dataclass
class ScoreBreakdown:
    """Detailed score breakdown for transparency."""
    structure_raw: float = 0.0
    wave_raw: float = 0.0
    liquidity_raw: float = 0.0
    timing_raw: float = 0.0
    market_alignment_raw: float = 0.0
    fibonacci_raw: float = 0.0

    structure_weighted: float = 0.0
    wave_weighted: float = 0.0
    liquidity_weighted: float = 0.0
    timing_weighted: float = 0.0
    market_alignment_weighted: float = 0.0
    fibonacci_weighted: float = 0.0

    final_confidence: float = 0.0

    def summary(self) -> str:
        return (
            f"Structure: {self.structure_raw:.0f} (w: {self.structure_weighted:.1f}) | "
            f"Wave: {self.wave_raw:.0f} (w: {self.wave_weighted:.1f}) | "
            f"Liquidity: {self.liquidity_raw:.0f} (w: {self.liquidity_weighted:.1f}) | "
            f"Timing: {self.timing_raw:.0f} (w: {self.timing_weighted:.1f}) | "
            f"Market: {self.market_alignment_raw:.0f} (w: {self.market_alignment_weighted:.1f}) | "
            f"Fib: {self.fibonacci_raw:.0f} (w: {self.fibonacci_weighted:.1f}) | "
            f"=> Confidence: {self.final_confidence:.0f}%"
        )


class ScoringEngine:
    """
    Combines individual module scores into a final confidence score
    using configurable weights.
    """

    def __init__(self, weights: Optional[ScoringWeights] = None):
        self.weights = weights or SCORING_WEIGHTS
        self._validate_weights()

    def _validate_weights(self) -> None:
        """Ensure weights sum to ~1.0."""
        total = self.weights.total()
        if abs(total - 1.0) > 0.01:
            logger.warning(
                f"Scoring weights sum to {total:.3f}, expected ~1.0. "
                "Results may be scaled unexpectedly."
            )

    def calculate(
        self,
        structure: Optional[StructureResult] = None,
        elliott: Optional[ElliottResult] = None,
        liquidity: Optional[LiquidityResult] = None,
        timing: Optional[TimingResult] = None,
        cross_market: Optional[CrossMarketResult] = None,
        fibonacci: Optional[FibonacciResult] = None,
        gann: Optional[GannResult] = None,
        intent: Optional[IntentResult] = None,
    ) -> ScoreBreakdown:
        """
        Calculate final confidence score from all module scores.

        Each module provides a raw score (0-100).
        Final score = weighted sum of all raw scores.

        Args:
            structure: Market structure result
            elliott: Elliott wave result
            liquidity: Liquidity analysis result
            timing: Timing engine result
            cross_market: Cross-market alignment result
            fibonacci: Fibonacci confluence result
            gann: Gann time cycles result (bonus)
            intent: Market intent result (bonus)

        Returns:
            ScoreBreakdown with detailed scores and final confidence
        """
        breakdown = ScoreBreakdown()

        # Raw scores (0-100 each)
        breakdown.structure_raw = structure.score if structure else 0.0
        breakdown.wave_raw = elliott.score if elliott else 0.0
        breakdown.liquidity_raw = liquidity.score if liquidity else 0.0
        breakdown.timing_raw = timing.score if timing else 0.0
        breakdown.market_alignment_raw = cross_market.score if cross_market else 0.0
        breakdown.fibonacci_raw = fibonacci.score if fibonacci else 0.0

        # Weighted scores
        breakdown.structure_weighted = breakdown.structure_raw * self.weights.structure
        breakdown.wave_weighted = breakdown.wave_raw * self.weights.wave
        breakdown.liquidity_weighted = breakdown.liquidity_raw * self.weights.liquidity
        breakdown.timing_weighted = breakdown.timing_raw * self.weights.timing
        breakdown.market_alignment_weighted = breakdown.market_alignment_raw * self.weights.market_alignment
        breakdown.fibonacci_weighted = breakdown.fibonacci_raw * self.weights.fibonacci

        # Base confidence = weighted sum
        confidence = (
            breakdown.structure_weighted
            + breakdown.wave_weighted
            + breakdown.liquidity_weighted
            + breakdown.timing_weighted
            + breakdown.market_alignment_weighted
            + breakdown.fibonacci_weighted
        )

        # Bonus modifiers (small additive boosts, not weighted)
        if gann and gann.cycle_confluence:
            confidence += 5.0  # Gann confluence bonus

        if intent and intent.score > 60:
            confidence += 5.0  # Strong intent bonus

        # Clamp to 0-100
        breakdown.final_confidence = max(0.0, min(100.0, confidence))

        return breakdown

    def update_weights(self, new_weights: ScoringWeights) -> None:
        """Update scoring weights (used by learning system)."""
        self.weights = new_weights
        self._validate_weights()
