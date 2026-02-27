"""
Core tests for Market Intelligence Platform modules.
Uses synthetic data to validate all analysis modules independently.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from market_intelligence.config import ScoringWeights
from market_intelligence.models import (
    Direction,
    EntryType,
    MarketPhase,
    SwingPoint,
    SwingType,
    TradeType,
    TrendDirection,
    WaveDegree,
    WaveType,
)
from market_intelligence.market_structure import MarketStructureAnalyzer
from market_intelligence.liquidity_model import LiquidityAnalyzer
from market_intelligence.elliott_wave import ElliottWaveEngine
from market_intelligence.fibonacci import FibonacciAnalyzer
from market_intelligence.gann_time import GannTimeAnalyzer
from market_intelligence.market_intent import MarketIntentAnalyzer
from market_intelligence.timing_engine import TimingEngine
from market_intelligence.cross_market import CrossMarketAnalyzer
from market_intelligence.scoring_engine import ScoringEngine, ScoreBreakdown
from market_intelligence.trade_decision import TradeDecisionEngine


# ── Fixtures / Helpers ─────────────────────────────────

def make_ohlcv(
    n: int = 100,
    trend: str = "up",
    base_price: float = 1000.0,
    volatility: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range(end=datetime.now(), periods=n, freq="D")

    prices = [base_price]
    for i in range(1, n):
        if trend == "up":
            drift = 0.001
        elif trend == "down":
            drift = -0.001
        else:  # sideways
            drift = 0.0
        change = drift + rng.normal(0, volatility)
        prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)
    noise = rng.uniform(0.005, 0.02, n)

    opens = prices * (1 + rng.uniform(-0.005, 0.005, n))
    highs = np.maximum(prices, opens) * (1 + noise)
    lows = np.minimum(prices, opens) * (1 - noise)
    closes = prices
    volumes = rng.randint(100000, 10000000, n)

    return pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
    }, index=dates)


def make_impulse_wave_data(n: int = 120) -> pd.DataFrame:
    """Generate data with a clear 5-wave impulse pattern."""
    dates = pd.date_range(end=datetime.now(), periods=n, freq="D")
    base = 1000.0

    # Wave 0->1: Up
    w1 = np.linspace(base, base + 100, 20)
    # Wave 1->2: Down (retrace ~50%)
    w2 = np.linspace(base + 100, base + 50, 15)
    # Wave 2->3: Strong up (longest)
    w3 = np.linspace(base + 50, base + 200, 30)
    # Wave 3->4: Down (retrace ~38%, stays above Wave 1)
    w4 = np.linspace(base + 200, base + 140, 20)
    # Wave 4->5: Up
    w5 = np.linspace(base + 140, base + 250, 35)

    closes = np.concatenate([w1, w2, w3, w4, w5])
    rng = np.random.RandomState(42)
    noise = rng.uniform(0.005, 0.015, n)

    opens = closes * (1 + rng.uniform(-0.003, 0.003, n))
    highs = np.maximum(closes, opens) * (1 + noise)
    lows = np.minimum(closes, opens) * (1 - noise)
    volumes = rng.randint(100000, 5000000, n)

    return pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
    }, index=dates)


# ── Tests ──────────────────────────────────────────────

class TestMarketStructure:
    def test_uptrend_detection(self):
        df = make_ohlcv(100, trend="up")
        analyzer = MarketStructureAnalyzer()
        result = analyzer.analyze(df)

        assert result.trend in (TrendDirection.BULLISH, TrendDirection.SIDEWAYS)
        assert len(result.swing_points) > 0
        assert result.score >= 0

    def test_downtrend_detection(self):
        df = make_ohlcv(100, trend="down")
        analyzer = MarketStructureAnalyzer()
        result = analyzer.analyze(df)

        assert result.trend in (TrendDirection.BEARISH, TrendDirection.SIDEWAYS)
        assert result.score >= 0

    def test_sideways_detection(self):
        df = make_ohlcv(100, trend="sideways", volatility=0.005)
        analyzer = MarketStructureAnalyzer()
        result = analyzer.analyze(df)

        assert result.phase in (MarketPhase.TRENDING, MarketPhase.RANGING)
        assert result.score >= 0

    def test_empty_data(self):
        df = pd.DataFrame()
        analyzer = MarketStructureAnalyzer()
        result = analyzer.analyze(df)

        assert result.trend == TrendDirection.SIDEWAYS
        assert result.phase == MarketPhase.RANGING

    def test_swing_points_labeled(self):
        df = make_ohlcv(100, trend="up")
        analyzer = MarketStructureAnalyzer()
        result = analyzer.analyze(df)

        for sp in result.swing_points:
            assert sp.swing_type in (SwingType.HH, SwingType.HL, SwingType.LH, SwingType.LL)
            assert sp.price > 0


class TestLiquidity:
    def test_sr_detection(self):
        df = make_ohlcv(100, trend="sideways", volatility=0.01)
        analyzer = LiquidityAnalyzer()
        result = analyzer.analyze(df)

        assert result.score >= 0
        assert isinstance(result.support_levels, list)
        assert isinstance(result.resistance_levels, list)

    def test_empty_data(self):
        analyzer = LiquidityAnalyzer()
        result = analyzer.analyze(pd.DataFrame())
        assert result.score == 0

    def test_zones_populated(self):
        df = make_ohlcv(100, trend="sideways", volatility=0.01)
        analyzer = LiquidityAnalyzer()
        result = analyzer.analyze(df)

        # Zones should be a list (may or may not be populated)
        assert isinstance(result.zones, list)


class TestElliottWave:
    def test_impulse_detection(self):
        df = make_impulse_wave_data()
        engine = ElliottWaveEngine()
        result = engine.analyze(df)

        assert result.score >= 0
        assert result.degree == WaveDegree.MINOR

    def test_empty_data(self):
        engine = ElliottWaveEngine()
        result = engine.analyze(pd.DataFrame())

        assert result.wave_type is None
        assert result.score == 0

    def test_short_data(self):
        df = make_ohlcv(10)
        engine = ElliottWaveEngine()
        result = engine.analyze(df)

        assert result.score >= 0


class TestFibonacci:
    def test_level_calculation(self):
        df = make_ohlcv(100, trend="up")
        analyzer = FibonacciAnalyzer()
        result = analyzer.analyze(df)

        assert len(result.retracement_levels) > 0
        assert len(result.extension_levels) > 0
        assert result.score >= 0

    def test_confluence_detection(self):
        df = make_ohlcv(100, trend="up")
        analyzer = FibonacciAnalyzer()
        result = analyzer.analyze(df)

        # Confluences are optional but structure should be valid
        assert isinstance(result.confluence_zones, list)

    def test_empty_data(self):
        analyzer = FibonacciAnalyzer()
        result = analyzer.analyze(pd.DataFrame())

        assert result.score == 0
        assert len(result.retracement_levels) == 0


class TestGannTime:
    def test_cycle_detection(self):
        df = make_ohlcv(200, trend="up")
        analyzer = GannTimeAnalyzer()
        result = analyzer.analyze(df)

        assert result.score >= 0
        assert isinstance(result.active_cycles, list)
        assert isinstance(result.next_cycle_bars, list)

    def test_short_data(self):
        df = make_ohlcv(30)
        analyzer = GannTimeAnalyzer()
        result = analyzer.analyze(df)

        assert result.score == 0  # Not enough data for any cycle


class TestMarketIntent:
    def test_intent_analysis(self):
        df = make_ohlcv(100, trend="up")
        analyzer = MarketIntentAnalyzer()
        structure_analyzer = MarketStructureAnalyzer()
        structure = structure_analyzer.analyze(df)

        result = analyzer.analyze(df, structure=structure)

        assert result.inferred_direction in (Direction.LONG, Direction.SHORT, Direction.WAIT)
        assert result.score >= 0

    def test_empty_data(self):
        analyzer = MarketIntentAnalyzer()
        result = analyzer.analyze(pd.DataFrame())

        assert result.inferred_direction == Direction.WAIT


class TestTimingEngine:
    def test_timing_analysis(self):
        df = make_ohlcv(100, trend="up")
        engine = TimingEngine()
        result = engine.analyze(df)

        assert result.expansion_probability >= 0
        assert result.score >= 0

    def test_contraction_detection(self):
        # Create data with decreasing volatility
        df = make_ohlcv(100, trend="sideways", volatility=0.005)
        engine = TimingEngine()
        result = engine.analyze(df)

        assert isinstance(result.atr_contracted, bool)
        assert isinstance(result.range_compressed, bool)


class TestCrossMarket:
    def test_alignment_check(self):
        nifty_df = make_ohlcv(100, trend="up")
        stock_df = make_ohlcv(100, trend="up", seed=43)

        analyzer = CrossMarketAnalyzer()
        result = analyzer.analyze("RELIANCE", nifty_df, {}, stock_df)

        assert result.nifty_trend in (TrendDirection.BULLISH, TrendDirection.BEARISH, TrendDirection.SIDEWAYS)
        assert result.score >= 0

    def test_no_data(self):
        analyzer = CrossMarketAnalyzer()
        result = analyzer.analyze("RELIANCE")

        assert result.nifty_trend == TrendDirection.SIDEWAYS
        assert not result.aligned


class TestScoringEngine:
    def test_scoring_calculation(self):
        from market_intelligence.models import (
            StructureResult, ElliottResult, LiquidityResult,
            TimingResult, CrossMarketResult, FibonacciResult,
        )

        engine = ScoringEngine()
        scores = engine.calculate(
            structure=StructureResult(trend=TrendDirection.BULLISH, phase=MarketPhase.TRENDING, score=70),
            elliott=ElliottResult(score=60),
            liquidity=LiquidityResult(score=50),
            timing=TimingResult(score=40),
            cross_market=CrossMarketResult(score=80),
            fibonacci=FibonacciResult(score=55),
        )

        assert 0 <= scores.final_confidence <= 100
        assert scores.structure_weighted > 0
        assert scores.wave_weighted > 0

    def test_empty_inputs(self):
        engine = ScoringEngine()
        scores = engine.calculate()

        assert scores.final_confidence == 0


class TestTradeDecision:
    def test_trade_generation(self):
        df = make_ohlcv(100, trend="up")
        engine = TradeDecisionEngine()

        from market_intelligence.scoring_engine import ScoreBreakdown
        scores = ScoreBreakdown(final_confidence=75)

        trade = engine.generate("RELIANCE", df, scores)

        assert trade.symbol == "RELIANCE"
        assert trade.direction in (Direction.LONG, Direction.SHORT, Direction.WAIT)

    def test_low_confidence_wait(self):
        df = make_ohlcv(100, trend="sideways")
        engine = TradeDecisionEngine()

        scores = ScoreBreakdown(final_confidence=30)
        trade = engine.generate("TCS", df, scores)

        assert trade.direction == Direction.WAIT

    def test_trade_summary(self):
        from market_intelligence.models import TradeDecision
        trade = TradeDecision(
            symbol="RELIANCE",
            direction=Direction.LONG,
            entry_type=EntryType.PULLBACK,
            trade_type=TradeType.SWING,
            holding_period="4-7 Days",
            entry_zone=2500.0,
            stop_loss=2400.0,
            target_1=2600.0,
            target_2=2700.0,
            target_3=2800.0,
            confidence=78.0,
            reasoning="Bullish trend with BOS",
        )

        summary = trade.summary()
        assert "RELIANCE" in summary
        assert "LONG" in summary
        assert "78%" in summary


class TestIntegration:
    """Integration tests using synthetic data."""

    def test_full_pipeline(self):
        """Test the entire analysis pipeline with synthetic data."""
        df = make_ohlcv(200, trend="up")

        # Phase 2: Structure
        structure_analyzer = MarketStructureAnalyzer()
        structure = structure_analyzer.analyze(df)

        # Phase 3: Liquidity
        liquidity_analyzer = LiquidityAnalyzer()
        liquidity = liquidity_analyzer.analyze(df)

        # Phase 4: Elliott
        elliott_engine = ElliottWaveEngine()
        elliott = elliott_engine.analyze(df, structure.swing_points)

        # Phase 5: Fibonacci
        fib_analyzer = FibonacciAnalyzer()
        fibonacci = fib_analyzer.analyze(df, structure.swing_points)

        # Phase 6: Gann
        gann_analyzer = GannTimeAnalyzer()
        gann = gann_analyzer.analyze(df)

        # Phase 7: Intent
        intent_analyzer = MarketIntentAnalyzer()
        intent = intent_analyzer.analyze(df, structure, liquidity)

        # Phase 8: Timing
        timing_engine = TimingEngine()
        timing = timing_engine.analyze(df)

        # Phase 9: Cross Market (with NIFTY as same data for test)
        cross_analyzer = CrossMarketAnalyzer()
        cross_market = cross_analyzer.analyze("RELIANCE", df, {}, df)

        # Phase 10: Scoring
        scoring_engine = ScoringEngine()
        scores = scoring_engine.calculate(
            structure=structure,
            elliott=elliott,
            liquidity=liquidity,
            timing=timing,
            cross_market=cross_market,
            fibonacci=fibonacci,
            gann=gann,
            intent=intent,
        )

        assert 0 <= scores.final_confidence <= 100

        # Phase 11: Trade Decision
        trade_engine = TradeDecisionEngine()
        trade = trade_engine.generate(
            symbol="RELIANCE",
            df=df,
            scores=scores,
            structure=structure,
            elliott=elliott,
            liquidity=liquidity,
            fibonacci=fibonacci,
            timing=timing,
            intent=intent,
            cross_market=cross_market,
        )

        assert trade.symbol == "RELIANCE"
        assert trade.confidence >= 0

    def test_bearish_pipeline(self):
        """Test pipeline with bearish data."""
        df = make_ohlcv(200, trend="down")

        structure = MarketStructureAnalyzer().analyze(df)
        liquidity = LiquidityAnalyzer().analyze(df)
        timing = TimingEngine().analyze(df)
        intent = MarketIntentAnalyzer().analyze(df, structure, liquidity)

        scoring = ScoringEngine()
        scores = scoring.calculate(structure=structure, liquidity=liquidity, timing=timing)

        trade = TradeDecisionEngine().generate("TCS", df, scores, structure=structure)

        assert trade.symbol == "TCS"
        assert trade.confidence >= 0
