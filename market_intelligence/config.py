"""
Configuration module for Market Intelligence Platform.
Central place for all tunable parameters and constants.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# ── Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
PREDICTIONS_DIR = DATA_DIR / "predictions"
CHARTS_DIR = DATA_DIR / "charts"

for _dir in (DATA_DIR, CACHE_DIR, PREDICTIONS_DIR, CHARTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ── NSE Symbol Suffix ─────────────────────────────────
NSE_SUFFIX = ".NS"
NFO_SUFFIX = ".NS"  # yfinance uses same suffix for derivatives


# ── Universe ───────────────────────────────────────────
NIFTY50_SYMBOLS: List[str] = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "HCLTECH",
    "SUNPHARMA", "TITAN", "BAJFINANCE", "WIPRO", "ULTRACEMCO",
    "NESTLEIND", "NTPC", "POWERGRID", "M&M", "TATAMOTORS",
    "ONGC", "JSWSTEEL", "TATASTEEL", "ADANIENT", "ADANIPORTS",
    "TECHM", "HDFCLIFE", "SBILIFE", "BAJAJFINSV", "COALINDIA",
    "GRASIM", "INDUSINDBK", "BRITANNIA", "CIPLA", "DIVISLAB",
    "DRREDDY", "EICHERMOT", "HEROMOTOCO", "APOLLOHOSP", "TATACONSUM",
    "BPCL", "LTIM", "BAJAJ-AUTO", "UPL", "HINDALCO",
]

# Sector mapping for cross-market checks
SECTOR_MAP: Dict[str, List[str]] = {
    "BANKING": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK"],
    "IT": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM"],
    "PHARMA": ["SUNPHARMA", "CIPLA", "DIVISLAB", "DRREDDY", "APOLLOHOSP"],
    "AUTO": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO"],
    "ENERGY": ["RELIANCE", "ONGC", "BPCL", "NTPC", "POWERGRID"],
    "METALS": ["JSWSTEEL", "TATASTEEL", "HINDALCO", "COALINDIA"],
    "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM"],
    "INFRA": ["LT", "ADANIENT", "ADANIPORTS", "ULTRACEMCO", "GRASIM"],
    "FINANCE": ["BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE"],
}

# Index symbols
NIFTY_INDEX = "^NSEI"
BANKNIFTY_INDEX = "^NSEBANK"


# ── Timeframes ─────────────────────────────────────────
DEFAULT_TIMEFRAME = "1d"
SUPPORTED_TIMEFRAMES = ["5m", "15m", "1h", "1d", "1wk"]


# ── Market Structure ───────────────────────────────────
@dataclass
class MarketStructureConfig:
    swing_lookback: int = 5          # Bars to confirm a swing high/low
    bos_min_bars: int = 3            # Minimum bars for Break of Structure
    choch_confirmation_bars: int = 2  # Bars to confirm Change of Character
    range_threshold_atr: float = 1.5  # ATR multiplier for range detection


# ── Liquidity ──────────────────────────────────────────
@dataclass
class LiquidityConfig:
    equal_level_tolerance: float = 0.002  # 0.2% tolerance for equal highs/lows
    cluster_tolerance: float = 0.005      # 0.5% for S/R clustering
    sweep_wick_ratio: float = 0.6         # Min wick-to-body ratio for sweep candle
    lookback_bars: int = 50               # Bars to look back for liquidity


# ── Elliott Wave ───────────────────────────────────────
@dataclass
class ElliottConfig:
    min_wave_bars: int = 5           # Minimum bars for a wave
    wave3_min_ratio: float = 1.0     # Wave 3 must be >= 1.0x Wave 1
    wave3_max_ratio: float = 2.618   # Wave 3 max extension
    wave5_max_ratio: float = 1.618   # Wave 5 max extension relative to Wave 1
    correction_min_ratio: float = 0.382  # Min retracement for correction
    correction_max_ratio: float = 0.786  # Max retracement for correction


# ── Fibonacci ──────────────────────────────────────────
FIBONACCI_RETRACEMENT_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]
FIBONACCI_EXTENSION_LEVELS = [1.0, 1.272, 1.618, 2.0, 2.618]
FIBONACCI_CONFLUENCE_TOLERANCE = 0.005  # 0.5% zone tolerance


# ── Gann Time Cycles ──────────────────────────────────
GANN_CYCLES = [45, 60, 90, 120, 144, 180]
GANN_CYCLE_TOLERANCE = 3  # +/- bars tolerance for cycle match


# ── Timing Engine ──────────────────────────────────────
@dataclass
class TimingConfig:
    atr_period: int = 14
    atr_contraction_threshold: float = 0.7   # ATR < 70% of 20-bar avg
    range_compression_bars: int = 10          # Bars for range compression check
    breakout_velocity_threshold: float = 1.5  # Price move > 1.5x ATR


# ── Scoring Weights ────────────────────────────────────
@dataclass
class ScoringWeights:
    structure: float = 0.25
    wave: float = 0.20
    liquidity: float = 0.20
    timing: float = 0.15
    market_alignment: float = 0.10
    fibonacci: float = 0.10

    def total(self) -> float:
        return (self.structure + self.wave + self.liquidity +
                self.timing + self.market_alignment + self.fibonacci)


# ── Trade Decision ─────────────────────────────────────
@dataclass
class TradeConfig:
    min_confidence_trade: float = 55.0   # Min confidence to issue a trade
    min_confidence_high: float = 75.0    # High confidence threshold
    risk_reward_min: float = 1.5         # Minimum R:R ratio
    max_stop_loss_pct: float = 0.05      # 5% max stop loss
    intraday_max_bars: int = 78          # ~6.5 hours of 5-min candles


# ── Learning System ────────────────────────────────────
@dataclass
class LearningConfig:
    weight_adjustment_rate: float = 0.02  # How much to adjust weights per outcome
    min_predictions_to_adjust: int = 10   # Min predictions before adjusting
    target_hit_boost: float = 0.01        # Boost weight on target hit
    stoploss_hit_penalty: float = 0.01    # Reduce weight on SL hit


# ── Default instances ──────────────────────────────────
MARKET_STRUCTURE_CFG = MarketStructureConfig()
LIQUIDITY_CFG = LiquidityConfig()
ELLIOTT_CFG = ElliottConfig()
TIMING_CFG = TimingConfig()
SCORING_WEIGHTS = ScoringWeights()
TRADE_CFG = TradeConfig()
LEARNING_CFG = LearningConfig()
