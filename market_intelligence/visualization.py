"""
Phase 13 — Visualization
Plots price charts with wave labels, support/resistance,
and target levels.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from market_intelligence.config import CHARTS_DIR
from market_intelligence.models import (
    Direction,
    ElliottResult,
    FibLevel,
    FibonacciResult,
    LiquidityResult,
    StructureResult,
    SwingPoint,
    TradeDecision,
    WaveLabel,
)

logger = logging.getLogger(__name__)


class ChartVisualizer:
    """
    Generates annotated price charts with technical analysis overlays.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or CHARTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_analysis(
        self,
        symbol: str,
        df: pd.DataFrame,
        structure: Optional[StructureResult] = None,
        elliott: Optional[ElliottResult] = None,
        fibonacci: Optional[FibonacciResult] = None,
        liquidity: Optional[LiquidityResult] = None,
        trade: Optional[TradeDecision] = None,
        last_n_bars: int = 100,
        save: bool = True,
        show: bool = False,
    ) -> Optional[str]:
        """
        Generate comprehensive analysis chart.

        Args:
            symbol: Stock symbol
            df: OHLCV DataFrame
            structure: Market structure result (swing points)
            elliott: Elliott wave result (wave labels)
            fibonacci: Fibonacci result (levels)
            liquidity: Liquidity result (S/R levels)
            trade: Trade decision (entry, SL, targets)
            last_n_bars: Number of bars to display
            save: Save chart to file
            show: Display chart (for interactive use)

        Returns:
            File path if saved, else None
        """
        if df.empty:
            return None

        # Trim to last N bars
        plot_df = df.iloc[-last_n_bars:].copy()

        fig, axes = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1],
                                  gridspec_kw={"hspace": 0.1})

        ax_price = axes[0]
        ax_volume = axes[1]

        # Plot candlestick-style (simplified as OHLC line)
        self._plot_price(ax_price, plot_df, symbol)

        # Overlay: Swing points
        if structure and structure.swing_points:
            self._plot_swing_points(ax_price, plot_df, structure.swing_points)

        # Overlay: Elliott wave labels
        if elliott and elliott.labels:
            self._plot_wave_labels(ax_price, plot_df, elliott.labels)

        # Overlay: Fibonacci levels
        if fibonacci:
            self._plot_fibonacci(ax_price, plot_df, fibonacci)

        # Overlay: Support/Resistance
        if liquidity:
            self._plot_sr_levels(ax_price, plot_df, liquidity)

        # Overlay: Trade levels
        if trade and trade.direction != Direction.WAIT:
            self._plot_trade_levels(ax_price, plot_df, trade)

        # Volume
        self._plot_volume(ax_volume, plot_df)

        # Title and legend
        title = f"{symbol} Analysis"
        if trade:
            title += f" | {trade.direction.value} | Confidence: {trade.confidence:.0f}%"
        ax_price.set_title(title, fontsize=14, fontweight="bold")

        ax_price.legend(loc="upper left", fontsize=8)
        ax_price.grid(True, alpha=0.3)
        ax_volume.grid(True, alpha=0.3)

        plt.tight_layout()

        file_path = None
        if save:
            file_path = str(self.output_dir / f"{symbol}_analysis.png")
            plt.savefig(file_path, dpi=150, bbox_inches="tight")
            logger.info(f"Chart saved: {file_path}")

        if show:
            plt.show()

        plt.close(fig)
        return file_path

    def _plot_price(self, ax: plt.Axes, df: pd.DataFrame, symbol: str) -> None:
        """Plot price as candlestick-style colored bars."""
        dates = range(len(df))
        opens = df["Open"].values
        highs = df["High"].values
        lows = df["Low"].values
        closes = df["Close"].values

        for i in range(len(df)):
            color = "#26a69a" if closes[i] >= opens[i] else "#ef5350"

            # Wick
            ax.plot([i, i], [lows[i], highs[i]], color=color, linewidth=0.5)

            # Body
            body_bottom = min(opens[i], closes[i])
            body_height = abs(closes[i] - opens[i])
            ax.bar(i, body_height, bottom=body_bottom, width=0.6,
                   color=color, edgecolor=color, linewidth=0.5)

        # X-axis labels (show every Nth date)
        n = max(len(df) // 10, 1)
        tick_positions = list(range(0, len(df), n))
        tick_labels = [str(df.index[i].date()) if hasattr(df.index[i], 'date') else str(df.index[i])[:10]
                       for i in tick_positions if i < len(df)]
        ax.set_xticks(tick_positions[:len(tick_labels)])
        ax.set_xticklabels(tick_labels, rotation=45, fontsize=7)

        ax.set_ylabel("Price", fontsize=10)

    def _plot_swing_points(
        self, ax: plt.Axes, df: pd.DataFrame, swing_points: List[SwingPoint]
    ) -> None:
        """Plot swing high/low markers."""
        start_idx = df.index[0] if not df.empty else None
        bar_count = len(df)

        for sp in swing_points:
            # Map swing point index to plot position
            # Find the position in the trimmed df
            plot_pos = None
            for i, idx in enumerate(df.index):
                if str(idx) == sp.bar_date:
                    plot_pos = i
                    break

            if plot_pos is None:
                continue

            if "High" in sp.swing_type.value:
                ax.annotate(
                    sp.swing_type.value[:2],
                    xy=(plot_pos, sp.price),
                    fontsize=7,
                    fontweight="bold",
                    color="#1565c0" if "Higher" in sp.swing_type.value else "#e65100",
                    ha="center",
                    va="bottom",
                )
            else:
                ax.annotate(
                    sp.swing_type.value[:2],
                    xy=(plot_pos, sp.price),
                    fontsize=7,
                    fontweight="bold",
                    color="#1565c0" if "Higher" in sp.swing_type.value else "#e65100",
                    ha="center",
                    va="top",
                )

    def _plot_wave_labels(
        self, ax: plt.Axes, df: pd.DataFrame, labels: List[WaveLabel]
    ) -> None:
        """Plot Elliott wave labels on the chart."""
        for wl in labels:
            if 0 <= wl.index < len(df):
                # Map to trimmed position
                plot_pos = wl.index - (len(df.index) - len(df))
                if 0 <= plot_pos < len(df):
                    ax.annotate(
                        f"({wl.label})",
                        xy=(plot_pos, wl.price),
                        fontsize=9,
                        fontweight="bold",
                        color="#6a1b9a",
                        ha="center",
                        va="bottom" if wl.label in ("1", "3", "5", "B") else "top",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#f3e5f5", alpha=0.7),
                    )

    def _plot_fibonacci(
        self, ax: plt.Axes, df: pd.DataFrame, fib: FibonacciResult
    ) -> None:
        """Plot Fibonacci retracement and extension levels."""
        xlim = (0, len(df) - 1)

        for level in fib.retracement_levels:
            if level.ratio in (0.382, 0.5, 0.618):
                ax.axhline(
                    y=level.price, color="#ff9800", linestyle="--",
                    alpha=0.5, linewidth=0.8,
                    label=f"Fib {level.ratio:.1%}: {level.price:.2f}" if level.ratio == 0.382 else "",
                )
                ax.text(
                    xlim[1] + 0.5, level.price,
                    f"{level.ratio:.1%}",
                    fontsize=7, color="#ff9800", va="center",
                )

        for level in fib.extension_levels:
            if level.ratio in (1.272, 1.618):
                ax.axhline(
                    y=level.price, color="#2196f3", linestyle=":",
                    alpha=0.4, linewidth=0.8,
                )
                ax.text(
                    xlim[1] + 0.5, level.price,
                    f"Ext {level.ratio:.3f}",
                    fontsize=7, color="#2196f3", va="center",
                )

        # Confluence zones as shaded areas
        for zone_low, zone_high in fib.confluence_zones:
            ax.axhspan(zone_low, zone_high, alpha=0.1, color="#ff9800",
                       label="Confluence Zone" if zone_low == fib.confluence_zones[0][0] else "")

    def _plot_sr_levels(
        self, ax: plt.Axes, df: pd.DataFrame, liquidity: LiquidityResult
    ) -> None:
        """Plot support and resistance levels."""
        for level in liquidity.support_levels[:5]:  # Limit to top 5
            ax.axhline(y=level, color="#4caf50", linestyle="-", alpha=0.3, linewidth=1)

        for level in liquidity.resistance_levels[:5]:
            ax.axhline(y=level, color="#f44336", linestyle="-", alpha=0.3, linewidth=1)

        # Equal highs/lows
        for level in liquidity.equal_highs[:3]:
            ax.axhline(y=level, color="#f44336", linestyle="--", alpha=0.5, linewidth=1.2)

        for level in liquidity.equal_lows[:3]:
            ax.axhline(y=level, color="#4caf50", linestyle="--", alpha=0.5, linewidth=1.2)

    def _plot_trade_levels(
        self, ax: plt.Axes, df: pd.DataFrame, trade: TradeDecision
    ) -> None:
        """Plot entry, stop loss, and target levels."""
        xlim = (0, len(df) - 1)

        if trade.entry_zone:
            ax.axhline(y=trade.entry_zone, color="#2196f3", linewidth=1.5, linestyle="-",
                       label=f"Entry: {trade.entry_zone:.2f}")

        if trade.stop_loss:
            ax.axhline(y=trade.stop_loss, color="#f44336", linewidth=1.5, linestyle="-",
                       label=f"SL: {trade.stop_loss:.2f}")
            ax.axhspan(
                min(trade.stop_loss, trade.entry_zone or trade.stop_loss),
                max(trade.stop_loss, trade.entry_zone or trade.stop_loss),
                alpha=0.05, color="#f44336",
            )

        targets = [
            (trade.target_1, "T1"),
            (trade.target_2, "T2"),
            (trade.target_3, "T3"),
        ]
        colors = ["#4caf50", "#66bb6a", "#81c784"]

        for (target, label), color in zip(targets, colors):
            if target:
                ax.axhline(y=target, color=color, linewidth=1, linestyle="--",
                           label=f"{label}: {target:.2f}")

    def _plot_volume(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Plot volume bars."""
        if "Volume" not in df.columns:
            return

        volumes = df["Volume"].values
        colors = ["#26a69a" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#ef5350"
                  for i in range(len(df))]

        ax.bar(range(len(df)), volumes, color=colors, alpha=0.5, width=0.6)
        ax.set_ylabel("Volume", fontsize=10)

        # Format y-axis for volume
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
