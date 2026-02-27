"""
Phase 1 — Data Engine
Fetch OHLCV data with caching and incremental updates.
Supports NSE stocks, NIFTY50, NIFTY200, NFO.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from market_intelligence.config import (
    CACHE_DIR,
    DEFAULT_TIMEFRAME,
    NIFTY50_SYMBOLS,
    NIFTY_INDEX,
    NSE_SUFFIX,
    SUPPORTED_TIMEFRAMES,
)

logger = logging.getLogger(__name__)


class DataEngine:
    """
    Lightweight data engine with local CSV caching and incremental updates.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._meta_file = self.cache_dir / "_meta.json"
        self._meta = self._load_meta()

    # ── Cache metadata ─────────────────────────────────

    def _load_meta(self) -> dict:
        if self._meta_file.exists():
            with open(self._meta_file, "r") as f:
                return json.load(f)
        return {}

    def _save_meta(self) -> None:
        with open(self._meta_file, "w") as f:
            json.dump(self._meta, f, indent=2, default=str)

    def _cache_key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol}_{timeframe}"

    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        safe_symbol = symbol.replace("^", "IDX_").replace("&", "_AND_")
        return self.cache_dir / f"{safe_symbol}_{timeframe}.csv"

    # ── Public API ─────────────────────────────────────

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = DEFAULT_TIMEFRAME,
        days: int = 365,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol. Uses cache with incremental updates.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE") or index ("^NSEI")
            timeframe: Candle interval (1d, 1h, 15m, 5m, 1wk)
            days: Historical days to fetch on first load
            force_refresh: Skip cache entirely

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Use one of {SUPPORTED_TIMEFRAMES}")

        yf_symbol = self._to_yf_symbol(symbol)
        cache_path = self._cache_path(symbol, timeframe)
        cache_key = self._cache_key(symbol, timeframe)

        if not force_refresh and cache_path.exists():
            df = self._load_cached(cache_path)
            df = self._incremental_update(df, yf_symbol, timeframe, cache_key)
        else:
            df = self._full_fetch(yf_symbol, timeframe, days)

        if df is not None and not df.empty:
            self._save_cached(df, cache_path, cache_key)

        return df if df is not None else pd.DataFrame()

    def get_multiple(
        self,
        symbols: List[str],
        timeframe: str = DEFAULT_TIMEFRAME,
        days: int = 365,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV for multiple symbols."""
        result = {}
        for sym in symbols:
            try:
                df = self.get_ohlcv(sym, timeframe, days)
                if not df.empty:
                    result[sym] = df
            except Exception as e:
                logger.warning(f"Failed to fetch {sym}: {e}")
        return result

    def get_nifty50(self, timeframe: str = DEFAULT_TIMEFRAME) -> Dict[str, pd.DataFrame]:
        """Fetch all NIFTY50 stocks."""
        return self.get_multiple(NIFTY50_SYMBOLS, timeframe)

    def get_index(self, index_symbol: str = NIFTY_INDEX, timeframe: str = DEFAULT_TIMEFRAME) -> pd.DataFrame:
        """Fetch index data (NIFTY, BANKNIFTY)."""
        return self.get_ohlcv(index_symbol, timeframe)

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Clear cache for a symbol or all."""
        if symbol:
            for tf in SUPPORTED_TIMEFRAMES:
                path = self._cache_path(symbol, tf)
                if path.exists():
                    path.unlink()
                key = self._cache_key(symbol, tf)
                self._meta.pop(key, None)
        else:
            for f in self.cache_dir.glob("*.csv"):
                f.unlink()
            self._meta = {}
        self._save_meta()

    # ── Internal ───────────────────────────────────────

    def _to_yf_symbol(self, symbol: str) -> str:
        """Convert local symbol to yfinance format."""
        if symbol.startswith("^"):
            return symbol  # Already an index symbol
        if symbol.endswith(".NS"):
            return symbol
        return f"{symbol}{NSE_SUFFIX}"

    def _full_fetch(self, yf_symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        """Full historical fetch."""
        end = datetime.now()
        start = end - timedelta(days=days)

        # yfinance limits intraday data
        if timeframe in ("5m", "15m"):
            days = min(days, 59)
            start = end - timedelta(days=days)
        elif timeframe == "1h":
            days = min(days, 729)
            start = end - timedelta(days=days)

        logger.info(f"Full fetch: {yf_symbol} | {timeframe} | {days}d")
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(start=start, end=end, interval=timeframe)
        return self._clean_df(df)

    def _incremental_update(
        self, cached_df: pd.DataFrame, yf_symbol: str, timeframe: str, cache_key: str
    ) -> pd.DataFrame:
        """Fetch only new data since last update."""
        meta = self._meta.get(cache_key, {})
        last_update = meta.get("last_update")

        if last_update:
            last_dt = pd.Timestamp(last_update)
            now = pd.Timestamp.now()

            # Skip if updated recently (within timeframe granularity)
            min_gap = self._min_update_gap(timeframe)
            if (now - last_dt) < min_gap:
                logger.debug(f"Cache fresh for {yf_symbol}, skipping update")
                return cached_df

        # Fetch from last cached date
        if not cached_df.empty:
            start_date = cached_df.index[-1]
        else:
            start_date = datetime.now() - timedelta(days=365)

        logger.info(f"Incremental update: {yf_symbol} from {start_date}")
        ticker = yf.Ticker(yf_symbol)
        new_df = ticker.history(start=start_date, interval=timeframe)
        new_df = self._clean_df(new_df)

        if new_df.empty:
            return cached_df

        # Merge: keep old data, append/overwrite with new
        combined = pd.concat([cached_df, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        return combined

    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame columns."""
        if df is None or df.empty:
            return pd.DataFrame()

        # Keep only OHLCV
        keep_cols = ["Open", "High", "Low", "Close", "Volume"]
        available = [c for c in keep_cols if c in df.columns]
        df = df[available].copy()

        # Drop rows with NaN in price columns
        price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
        df.dropna(subset=price_cols, inplace=True)

        return df

    def _load_cached(self, path: Path) -> pd.DataFrame:
        """Load cached CSV."""
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache {path}: {e}")
            return pd.DataFrame()

    def _save_cached(self, df: pd.DataFrame, path: Path, cache_key: str) -> None:
        """Save DataFrame to CSV and update metadata."""
        df.to_csv(path)
        self._meta[cache_key] = {
            "last_update": datetime.now().isoformat(),
            "rows": len(df),
            "start": str(df.index[0]) if not df.empty else "",
            "end": str(df.index[-1]) if not df.empty else "",
        }
        self._save_meta()

    def _min_update_gap(self, timeframe: str) -> timedelta:
        """Minimum time between incremental updates."""
        gaps = {
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "1d": timedelta(hours=6),
            "1wk": timedelta(days=1),
        }
        return gaps.get(timeframe, timedelta(hours=6))
