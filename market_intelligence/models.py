"""
Data models / typed containers used across all modules.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


# ── Enums ──────────────────────────────────────────────

class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    WAIT = "WAIT"


class EntryType(Enum):
    PULLBACK = "Pullback"
    BREAKOUT = "Breakout"
    REVERSAL = "Reversal"


class TradeType(Enum):
    INTRADAY = "Intraday"
    SWING = "Swing"


class TrendDirection(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"


class MarketPhase(Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"


class WaveType(Enum):
    IMPULSE = "IMPULSE"
    CORRECTIVE = "CORRECTIVE"


class WaveDegree(Enum):
    PRIMARY = "PRIMARY"
    INTERMEDIATE = "INTERMEDIATE"
    MINOR = "MINOR"


class SwingType(Enum):
    HH = "Higher High"
    HL = "Higher Low"
    LH = "Lower High"
    LL = "Lower Low"


class ScannerSetup(Enum):
    WAVE_EXPANSION = "Wave Expansion Setup"
    NEAR_SUPPORT = "Near Support"
    BREAKOUT = "Breakout Setup"
    TREND_CONTINUATION = "Trend Continuation"


# ── Data Models ────────────────────────────────────────

@dataclass
class SwingPoint:
    index: int
    price: float
    swing_type: SwingType
    bar_date: str = ""


@dataclass
class StructureResult:
    trend: TrendDirection
    phase: MarketPhase
    swing_points: List[SwingPoint] = field(default_factory=list)
    bos_detected: bool = False
    bos_level: Optional[float] = None
    choch_detected: bool = False
    choch_level: Optional[float] = None
    score: float = 0.0  # 0-100


@dataclass
class LiquidityZone:
    price_level: float
    zone_type: str  # "equal_high", "equal_low", "sr_cluster", "sweep"
    strength: int = 1  # Number of touches / confirmations


@dataclass
class LiquidityResult:
    equal_highs: List[float] = field(default_factory=list)
    equal_lows: List[float] = field(default_factory=list)
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    sweep_detected: bool = False
    sweep_direction: Optional[Direction] = None
    zones: List[LiquidityZone] = field(default_factory=list)
    score: float = 0.0


@dataclass
class WaveLabel:
    index: int
    price: float
    label: str  # "1", "2", "3", "4", "5", "A", "B", "C"
    degree: WaveDegree = WaveDegree.MINOR


@dataclass
class ElliottResult:
    wave_type: Optional[WaveType] = None
    current_wave: str = ""  # e.g., "3", "C"
    degree: WaveDegree = WaveDegree.MINOR
    labels: List[WaveLabel] = field(default_factory=list)
    is_extended: bool = False
    is_valid: bool = True
    invalidation_level: Optional[float] = None
    score: float = 0.0


@dataclass
class FibLevel:
    ratio: float
    price: float
    level_type: str  # "retracement" or "extension"


@dataclass
class FibonacciResult:
    retracement_levels: List[FibLevel] = field(default_factory=list)
    extension_levels: List[FibLevel] = field(default_factory=list)
    confluence_zones: List[Tuple[float, float]] = field(default_factory=list)  # (low, high) price zones
    score: float = 0.0


@dataclass
class GannResult:
    active_cycles: List[int] = field(default_factory=list)  # Cycle lengths currently active
    next_cycle_bars: List[int] = field(default_factory=list)  # Bars to next cycle turn
    cycle_confluence: bool = False  # Multiple cycles aligning
    score: float = 0.0


@dataclass
class IntentResult:
    sweep_detected: bool = False
    structure_shift: bool = False
    rejection_candle: bool = False
    continuation: bool = False
    inferred_direction: Direction = Direction.WAIT
    score: float = 0.0


@dataclass
class TimingResult:
    atr_contracted: bool = False
    range_compressed: bool = False
    breakout_velocity_met: bool = False
    expansion_probability: float = 0.0  # 0-100
    score: float = 0.0


@dataclass
class CrossMarketResult:
    nifty_trend: TrendDirection = TrendDirection.SIDEWAYS
    sector_trend: TrendDirection = TrendDirection.SIDEWAYS
    sector_name: str = ""
    aligned: bool = False
    score: float = 0.0


@dataclass
class TradeDecision:
    symbol: str
    direction: Direction
    entry_type: EntryType = EntryType.PULLBACK
    trade_type: TradeType = TradeType.SWING
    holding_period: str = ""  # e.g., "4-7 Days"
    entry_zone: Optional[float] = None
    stop_loss: Optional[float] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    target_3: Optional[float] = None
    confidence: float = 0.0
    reasoning: str = ""

    # Component scores
    structure_score: float = 0.0
    wave_score: float = 0.0
    liquidity_score: float = 0.0
    timing_score: float = 0.0
    market_alignment_score: float = 0.0
    fibonacci_score: float = 0.0

    def summary(self) -> str:
        lines = [
            f"{'=' * 50}",
            f"  {self.symbol}",
            f"{'=' * 50}",
            f"  Direction     : {self.direction.value}",
            f"  Entry         : {self.entry_type.value}",
            f"  Trade Type    : {self.trade_type.value}",
            f"  Holding       : {self.holding_period}",
            f"  Entry Zone    : {self.entry_zone:.2f}" if self.entry_zone else "",
            f"  Stop Loss     : {self.stop_loss:.2f}" if self.stop_loss else "",
            f"  Target 1      : {self.target_1:.2f}" if self.target_1 else "",
            f"  Target 2      : {self.target_2:.2f}" if self.target_2 else "",
            f"  Target 3      : {self.target_3:.2f}" if self.target_3 else "",
            f"  Confidence    : {self.confidence:.0f}%",
            f"{'─' * 50}",
            f"  Structure     : {self.structure_score:.0f}",
            f"  Wave          : {self.wave_score:.0f}",
            f"  Liquidity     : {self.liquidity_score:.0f}",
            f"  Timing        : {self.timing_score:.0f}",
            f"  Mkt Alignment : {self.market_alignment_score:.0f}",
            f"  Fibonacci     : {self.fibonacci_score:.0f}",
            f"{'─' * 50}",
            f"  {self.reasoning}",
            f"{'=' * 50}",
        ]
        return "\n".join(line for line in lines if line)


@dataclass
class ScanResult:
    symbol: str
    setup: ScannerSetup
    direction: Direction
    confidence: float
    note: str = ""


@dataclass
class PredictionRecord:
    """Stored prediction for learning system."""
    symbol: str
    timestamp: str
    direction: Direction
    entry_price: float
    stop_loss: float
    target_1: float
    confidence: float
    weights_snapshot: dict = field(default_factory=dict)
    outcome: str = "PENDING"  # "TARGET_HIT", "STOPLOSS_HIT", "EXPIRED", "PENDING"
    actual_exit_price: Optional[float] = None
