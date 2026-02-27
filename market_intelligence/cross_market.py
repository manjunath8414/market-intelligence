"""
Phase 9 — Cross Market Check
Checks NIFTY trend and sector trend alignment
for simple bias confirmation.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from market_intelligence.config import (
    NIFTY_INDEX,
    SECTOR_MAP,
)
from market_intelligence.models import (
    CrossMarketResult,
    TrendDirection,
)

logger = logging.getLogger(__name__)


class CrossMarketAnalyzer:
    """
    Checks broader market (NIFTY) and sector trend
    for alignment with individual stock analysis.
    """

    def __init__(self, sector_map: Optional[Dict[str, List[str]]] = None):
        self.sector_map = sector_map or SECTOR_MAP

    def analyze(
        self,
        symbol: str,
        nifty_df: Optional[pd.DataFrame] = None,
        sector_data: Optional[Dict[str, pd.DataFrame]] = None,
        stock_df: Optional[pd.DataFrame] = None,
    ) -> CrossMarketResult:
        """
        Cross-market alignment check.

        Args:
            symbol: Stock symbol being analyzed
            nifty_df: NIFTY index OHLCV data
            sector_data: Dict of symbol -> OHLCV for sector peers
            stock_df: The stock's own OHLCV data (for fallback sector trend)

        Returns:
            CrossMarketResult with alignment status
        """
        # NIFTY trend
        nifty_trend = self._compute_trend(nifty_df) if nifty_df is not None else TrendDirection.SIDEWAYS

        # Find the sector for this symbol
        sector_name = self._find_sector(symbol)
        sector_trend = TrendDirection.SIDEWAYS

        if sector_name and sector_data:
            sector_trend = self._compute_sector_trend(sector_data, symbol)

        # Alignment: stock trend should match NIFTY and sector
        stock_trend = self._compute_trend(stock_df) if stock_df is not None else TrendDirection.SIDEWAYS
        aligned = self._check_alignment(stock_trend, nifty_trend, sector_trend)

        score = self._calculate_score(nifty_trend, sector_trend, stock_trend, aligned)

        return CrossMarketResult(
            nifty_trend=nifty_trend,
            sector_trend=sector_trend,
            sector_name=sector_name,
            aligned=aligned,
            score=score,
        )

    def _compute_trend(self, df: Optional[pd.DataFrame]) -> TrendDirection:
        """Compute simple trend from price data using EMAs."""
        if df is None or df.empty or len(df) < 20:
            return TrendDirection.SIDEWAYS

        closes = df["Close"].values

        # 20-period and 50-period simple moving averages
        sma20 = np.mean(closes[-20:])
        sma50 = np.mean(closes[-min(50, len(closes)):])
        current = closes[-1]

        # Bullish: price > SMA20 > SMA50
        if current > sma20 > sma50:
            return TrendDirection.BULLISH

        # Bearish: price < SMA20 < SMA50
        if current < sma20 < sma50:
            return TrendDirection.BEARISH

        return TrendDirection.SIDEWAYS

    def _find_sector(self, symbol: str) -> str:
        """Find which sector a symbol belongs to."""
        for sector, symbols in self.sector_map.items():
            if symbol in symbols:
                return sector
        return ""

    def _compute_sector_trend(
        self, sector_data: Dict[str, pd.DataFrame], exclude_symbol: str
    ) -> TrendDirection:
        """Compute sector trend from peer stocks."""
        trends = []
        for sym, df in sector_data.items():
            if sym == exclude_symbol:
                continue
            trend = self._compute_trend(df)
            trends.append(trend)

        if not trends:
            return TrendDirection.SIDEWAYS

        bullish = sum(1 for t in trends if t == TrendDirection.BULLISH)
        bearish = sum(1 for t in trends if t == TrendDirection.BEARISH)
        total = len(trends)

        if bullish > total * 0.6:
            return TrendDirection.BULLISH
        elif bearish > total * 0.6:
            return TrendDirection.BEARISH

        return TrendDirection.SIDEWAYS

    def _check_alignment(
        self,
        stock_trend: TrendDirection,
        nifty_trend: TrendDirection,
        sector_trend: TrendDirection,
    ) -> bool:
        """Check if stock, NIFTY, and sector trends are aligned."""
        if stock_trend == TrendDirection.SIDEWAYS:
            return False

        # Full alignment
        if stock_trend == nifty_trend == sector_trend:
            return True

        # Partial alignment (stock + one of NIFTY/sector)
        if stock_trend == nifty_trend or stock_trend == sector_trend:
            return True

        return False

    def _calculate_score(
        self,
        nifty_trend: TrendDirection,
        sector_trend: TrendDirection,
        stock_trend: TrendDirection,
        aligned: bool,
    ) -> float:
        """Calculate cross-market alignment score 0-100."""
        score = 0.0

        # NIFTY trend is clear
        if nifty_trend != TrendDirection.SIDEWAYS:
            score += 20.0

        # Sector trend is clear
        if sector_trend != TrendDirection.SIDEWAYS:
            score += 20.0

        # Full alignment: all three match
        if stock_trend == nifty_trend == sector_trend and stock_trend != TrendDirection.SIDEWAYS:
            score += 40.0
        elif aligned:
            score += 20.0

        return min(score, 100.0)
