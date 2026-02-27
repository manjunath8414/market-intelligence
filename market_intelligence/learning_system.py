"""
Phase 14 — Light Learning System
Stores predictions locally and adjusts scoring weights
based on target hit / stop loss hit outcomes.
No neural network training.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from market_intelligence.config import (
    LEARNING_CFG,
    PREDICTIONS_DIR,
    SCORING_WEIGHTS,
    LearningConfig,
    ScoringWeights,
)
from market_intelligence.models import (
    Direction,
    PredictionRecord,
    TradeDecision,
)

logger = logging.getLogger(__name__)


class LearningSystem:
    """
    Lightweight learning system that:
    1. Stores predictions with their scoring weights
    2. Evaluates outcomes (target hit / SL hit)
    3. Adjusts scoring weights incrementally based on outcomes
    """

    def __init__(
        self,
        predictions_dir: Optional[Path] = None,
        config: Optional[LearningConfig] = None,
    ):
        self.predictions_dir = predictions_dir or PREDICTIONS_DIR
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = config or LEARNING_CFG
        self._predictions_file = self.predictions_dir / "predictions.json"
        self._weights_file = self.predictions_dir / "adjusted_weights.json"

    # ── Store Predictions ──────────────────────────────

    def store_prediction(self, trade: TradeDecision) -> None:
        """Store a trade prediction for future evaluation."""
        record = PredictionRecord(
            symbol=trade.symbol,
            timestamp=datetime.now().isoformat(),
            direction=trade.direction,
            entry_price=trade.entry_zone or 0.0,
            stop_loss=trade.stop_loss or 0.0,
            target_1=trade.target_1 or 0.0,
            confidence=trade.confidence,
            weights_snapshot={
                "structure": trade.structure_score,
                "wave": trade.wave_score,
                "liquidity": trade.liquidity_score,
                "timing": trade.timing_score,
                "market_alignment": trade.market_alignment_score,
                "fibonacci": trade.fibonacci_score,
            },
            outcome="PENDING",
        )

        predictions = self._load_predictions()
        predictions.append(self._record_to_dict(record))
        self._save_predictions(predictions)

        logger.info(f"Stored prediction: {trade.symbol} {trade.direction.value} @ {trade.entry_zone}")

    def evaluate_predictions(self, current_data: Dict[str, pd.DataFrame]) -> Dict[str, int]:
        """
        Evaluate pending predictions against current market data.

        Args:
            current_data: Dict of symbol -> current OHLCV DataFrame

        Returns:
            Dict with counts: {"target_hit": N, "stoploss_hit": N, "expired": N}
        """
        predictions = self._load_predictions()
        counts = {"target_hit": 0, "stoploss_hit": 0, "expired": 0, "still_pending": 0}

        for pred in predictions:
            if pred["outcome"] != "PENDING":
                continue

            symbol = pred["symbol"]
            if symbol not in current_data:
                continue

            df = current_data[symbol]
            if df.empty:
                continue

            current_price = df["Close"].iloc[-1]
            entry = pred["entry_price"]
            sl = pred["stop_loss"]
            target = pred["target_1"]
            direction = pred["direction"]

            # Check high/low since prediction
            pred_time = pd.Timestamp(pred["timestamp"])
            recent = df[df.index >= pred_time] if pred_time in df.index or True else df.tail(20)

            if recent.empty:
                counts["still_pending"] += 1
                continue

            high_since = recent["High"].max()
            low_since = recent["Low"].min()

            if direction == "LONG":
                if target > 0 and high_since >= target:
                    pred["outcome"] = "TARGET_HIT"
                    pred["actual_exit_price"] = target
                    counts["target_hit"] += 1
                elif sl > 0 and low_since <= sl:
                    pred["outcome"] = "STOPLOSS_HIT"
                    pred["actual_exit_price"] = sl
                    counts["stoploss_hit"] += 1
                else:
                    # Check if expired (more than 30 bars old)
                    if len(recent) > 30:
                        pred["outcome"] = "EXPIRED"
                        pred["actual_exit_price"] = current_price
                        counts["expired"] += 1
                    else:
                        counts["still_pending"] += 1
            elif direction == "SHORT":
                if target > 0 and low_since <= target:
                    pred["outcome"] = "TARGET_HIT"
                    pred["actual_exit_price"] = target
                    counts["target_hit"] += 1
                elif sl > 0 and high_since >= sl:
                    pred["outcome"] = "STOPLOSS_HIT"
                    pred["actual_exit_price"] = sl
                    counts["stoploss_hit"] += 1
                else:
                    if len(recent) > 30:
                        pred["outcome"] = "EXPIRED"
                        pred["actual_exit_price"] = current_price
                        counts["expired"] += 1
                    else:
                        counts["still_pending"] += 1

        self._save_predictions(predictions)
        return counts

    # ── Adjust Weights ─────────────────────────────────

    def adjust_weights(self) -> ScoringWeights:
        """
        Adjust scoring weights based on prediction outcomes.
        Components that contributed to winning trades get boosted;
        those that led to losses get reduced.

        Returns:
            Updated ScoringWeights
        """
        predictions = self._load_predictions()
        evaluated = [p for p in predictions if p["outcome"] in ("TARGET_HIT", "STOPLOSS_HIT")]

        if len(evaluated) < self.cfg.min_predictions_to_adjust:
            logger.info(
                f"Only {len(evaluated)} evaluated predictions, "
                f"need {self.cfg.min_predictions_to_adjust} to adjust weights"
            )
            return self.load_weights()

        # Current weights
        weights = self.load_weights()

        # Analyze which components correlated with wins/losses
        component_names = ["structure", "wave", "liquidity", "timing", "market_alignment", "fibonacci"]

        for pred in evaluated:
            snapshot = pred.get("weights_snapshot", {})
            is_win = pred["outcome"] == "TARGET_HIT"

            for component in component_names:
                component_score = snapshot.get(component, 0)
                current_weight = getattr(weights, component)

                if is_win and component_score > 50:
                    # High-scoring component led to a win — boost
                    new_weight = current_weight + self.cfg.target_hit_boost
                    setattr(weights, component, new_weight)
                elif not is_win and component_score > 50:
                    # High-scoring component led to a loss — penalize
                    new_weight = max(0.05, current_weight - self.cfg.stoploss_hit_penalty)
                    setattr(weights, component, new_weight)

        # Normalize weights to sum to 1.0
        weights = self._normalize_weights(weights, component_names)

        # Save adjusted weights
        self.save_weights(weights)
        logger.info(f"Adjusted weights: {self._weights_to_dict(weights)}")

        return weights

    def _normalize_weights(self, weights: ScoringWeights, components: List[str]) -> ScoringWeights:
        """Normalize weights to sum to 1.0."""
        total = sum(getattr(weights, c) for c in components)
        if total > 0:
            for c in components:
                setattr(weights, c, getattr(weights, c) / total)
        return weights

    # ── Persistence ────────────────────────────────────

    def _load_predictions(self) -> List[dict]:
        if self._predictions_file.exists():
            with open(self._predictions_file, "r") as f:
                return json.load(f)
        return []

    def _save_predictions(self, predictions: List[dict]) -> None:
        with open(self._predictions_file, "w") as f:
            json.dump(predictions, f, indent=2, default=str)

    def load_weights(self) -> ScoringWeights:
        """Load adjusted weights or return defaults."""
        if self._weights_file.exists():
            with open(self._weights_file, "r") as f:
                data = json.load(f)
                return ScoringWeights(**data)
        return ScoringWeights()

    def save_weights(self, weights: ScoringWeights) -> None:
        """Save adjusted weights to disk."""
        with open(self._weights_file, "w") as f:
            json.dump(self._weights_to_dict(weights), f, indent=2)

    def _weights_to_dict(self, weights: ScoringWeights) -> dict:
        return {
            "structure": weights.structure,
            "wave": weights.wave,
            "liquidity": weights.liquidity,
            "timing": weights.timing,
            "market_alignment": weights.market_alignment,
            "fibonacci": weights.fibonacci,
        }

    def _record_to_dict(self, record: PredictionRecord) -> dict:
        return {
            "symbol": record.symbol,
            "timestamp": record.timestamp,
            "direction": record.direction.value if isinstance(record.direction, Direction) else record.direction,
            "entry_price": record.entry_price,
            "stop_loss": record.stop_loss,
            "target_1": record.target_1,
            "confidence": record.confidence,
            "weights_snapshot": record.weights_snapshot,
            "outcome": record.outcome,
            "actual_exit_price": record.actual_exit_price,
        }

    def get_stats(self) -> dict:
        """Get prediction statistics."""
        predictions = self._load_predictions()
        total = len(predictions)
        outcomes = {}
        for p in predictions:
            outcome = p.get("outcome", "PENDING")
            outcomes[outcome] = outcomes.get(outcome, 0) + 1

        win_rate = 0.0
        wins = outcomes.get("TARGET_HIT", 0)
        losses = outcomes.get("STOPLOSS_HIT", 0)
        if wins + losses > 0:
            win_rate = wins / (wins + losses) * 100

        return {
            "total_predictions": total,
            "outcomes": outcomes,
            "win_rate": f"{win_rate:.1f}%",
        }
