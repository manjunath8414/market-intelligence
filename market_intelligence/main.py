"""
Main CLI entry point for Market Intelligence Platform.
Orchestrates all modules and provides a unified interface.
"""

import logging
import sys
from typing import Dict, List, Optional

import click
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from market_intelligence.config import (
    NIFTY50_SYMBOLS,
    NIFTY_INDEX,
    SCORING_WEIGHTS,
    SECTOR_MAP,
)
from market_intelligence.data_engine import DataEngine
from market_intelligence.market_structure import MarketStructureAnalyzer
from market_intelligence.liquidity_model import LiquidityAnalyzer
from market_intelligence.elliott_wave import ElliottWaveEngine
from market_intelligence.fibonacci import FibonacciAnalyzer
from market_intelligence.gann_time import GannTimeAnalyzer
from market_intelligence.market_intent import MarketIntentAnalyzer
from market_intelligence.timing_engine import TimingEngine
from market_intelligence.cross_market import CrossMarketAnalyzer
from market_intelligence.scoring_engine import ScoringEngine
from market_intelligence.trade_decision import TradeDecisionEngine
from market_intelligence.scanner import Scanner
from market_intelligence.visualization import ChartVisualizer
from market_intelligence.learning_system import LearningSystem
from market_intelligence.models import Direction, TradeDecision

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

console = Console()


class MarketIntelligencePlatform:
    """
    Central orchestrator that connects all analysis modules.
    """

    def __init__(self):
        self.data_engine = DataEngine()
        self.structure_analyzer = MarketStructureAnalyzer()
        self.liquidity_analyzer = LiquidityAnalyzer()
        self.elliott_engine = ElliottWaveEngine()
        self.fibonacci_analyzer = FibonacciAnalyzer()
        self.gann_analyzer = GannTimeAnalyzer()
        self.intent_analyzer = MarketIntentAnalyzer()
        self.timing_engine = TimingEngine()
        self.cross_market_analyzer = CrossMarketAnalyzer()
        self.scoring_engine = ScoringEngine()
        self.trade_engine = TradeDecisionEngine()
        self.scanner = Scanner()
        self.visualizer = ChartVisualizer()
        self.learning = LearningSystem()

        # Load adjusted weights if available
        adjusted_weights = self.learning.load_weights()
        self.scoring_engine.update_weights(adjusted_weights)

    def analyze_stock(
        self,
        symbol: str,
        timeframe: str = "1d",
        days: int = 365,
        plot: bool = True,
        store_prediction: bool = True,
    ) -> TradeDecision:
        """
        Run full analysis pipeline on a single stock.

        Args:
            symbol: NSE stock symbol (e.g., "RELIANCE")
            timeframe: Candle timeframe
            days: Historical days to analyze
            plot: Generate chart
            store_prediction: Store prediction for learning

        Returns:
            TradeDecision with complete analysis
        """
        console.print(f"\n[bold cyan]Analyzing {symbol}...[/bold cyan]")

        # Phase 1: Fetch data
        df = self.data_engine.get_ohlcv(symbol, timeframe, days)
        if df.empty:
            console.print(f"[red]No data available for {symbol}[/red]")
            return TradeDecision(
                symbol=symbol, direction=Direction.WAIT,
                confidence=0.0, reasoning="No data"
            )

        console.print(f"  Data: {len(df)} bars loaded ({df.index[0]} to {df.index[-1]})")

        # Phase 2: Market Structure
        structure = self.structure_analyzer.analyze(df)
        console.print(f"  Structure: {structure.trend.value} | {structure.phase.value} | "
                      f"BOS: {structure.bos_detected} | CHoCH: {structure.choch_detected}")

        # Phase 3: Liquidity
        liquidity = self.liquidity_analyzer.analyze(df)
        console.print(f"  Liquidity: S/R clusters: {len(liquidity.support_levels) + len(liquidity.resistance_levels)} | "
                      f"Sweep: {liquidity.sweep_detected}")

        # Phase 4: Elliott Wave
        elliott = self.elliott_engine.analyze(df, structure.swing_points)
        wave_info = f"{elliott.wave_type.value if elliott.wave_type else 'None'}"
        if elliott.current_wave:
            wave_info += f" (wave {elliott.current_wave})"
        console.print(f"  Elliott: {wave_info} | Valid: {elliott.is_valid}")

        # Phase 5: Fibonacci
        fibonacci = self.fibonacci_analyzer.analyze(df, structure.swing_points)
        console.print(f"  Fibonacci: {len(fibonacci.retracement_levels)} ret, "
                      f"{len(fibonacci.extension_levels)} ext, "
                      f"{len(fibonacci.confluence_zones)} confluences")

        # Phase 6: Gann Time
        gann = self.gann_analyzer.analyze(df)
        console.print(f"  Gann: Active cycles: {gann.active_cycles} | "
                      f"Confluence: {gann.cycle_confluence}")

        # Phase 7: Market Intent
        intent = self.intent_analyzer.analyze(df, structure, liquidity)
        console.print(f"  Intent: {intent.inferred_direction.value} | "
                      f"Sweep: {intent.sweep_detected} | Rejection: {intent.rejection_candle}")

        # Phase 8: Timing
        timing = self.timing_engine.analyze(df)
        console.print(f"  Timing: ATR contracted: {timing.atr_contracted} | "
                      f"Compressed: {timing.range_compressed} | "
                      f"Expansion prob: {timing.expansion_probability:.0f}%")

        # Phase 9: Cross Market
        nifty_df = self.data_engine.get_ohlcv(NIFTY_INDEX, timeframe, days)
        sector_data = self._get_sector_data(symbol, timeframe, days)
        cross_market = self.cross_market_analyzer.analyze(
            symbol, nifty_df, sector_data, df
        )
        console.print(f"  Cross-Market: NIFTY {cross_market.nifty_trend.value} | "
                      f"Sector ({cross_market.sector_name}): {cross_market.sector_trend.value} | "
                      f"Aligned: {cross_market.aligned}")

        # Phase 10: Scoring
        scores = self.scoring_engine.calculate(
            structure=structure,
            elliott=elliott,
            liquidity=liquidity,
            timing=timing,
            cross_market=cross_market,
            fibonacci=fibonacci,
            gann=gann,
            intent=intent,
        )
        console.print(f"  [bold]Confidence: {scores.final_confidence:.0f}%[/bold]")

        # Phase 11: Trade Decision
        trade = self.trade_engine.generate(
            symbol=symbol,
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

        # Display the trade decision
        self._display_trade(trade)

        # Phase 13: Visualization
        if plot:
            chart_path = self.visualizer.plot_analysis(
                symbol=symbol,
                df=df,
                structure=structure,
                elliott=elliott,
                fibonacci=fibonacci,
                liquidity=liquidity,
                trade=trade,
            )
            if chart_path:
                console.print(f"  [dim]Chart saved: {chart_path}[/dim]")

        # Phase 14: Store prediction
        if store_prediction and trade.direction != Direction.WAIT:
            self.learning.store_prediction(trade)

        return trade

    def scan_universe(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "1d",
        min_confidence: float = 40.0,
    ) -> None:
        """
        Scan a universe of stocks for trade setups.
        """
        symbols = symbols or NIFTY50_SYMBOLS
        console.print(f"\n[bold cyan]Scanning {len(symbols)} stocks...[/bold cyan]")

        # Fetch data for all symbols
        data = self.data_engine.get_multiple(symbols, timeframe)
        console.print(f"  Data loaded for {len(data)} symbols")

        # Run scanner
        results = self.scanner.scan(data, min_confidence=min_confidence)

        if not results:
            console.print("[yellow]No setups found meeting criteria[/yellow]")
            return

        # Display results
        table = Table(title="Scanner Results", show_header=True, header_style="bold magenta")
        table.add_column("Symbol", style="cyan", width=12)
        table.add_column("Setup", width=22)
        table.add_column("Direction", width=10)
        table.add_column("Confidence", justify="right", width=12)
        table.add_column("Note", width=40)

        for r in results:
            dir_color = "green" if r.direction == Direction.LONG else "red" if r.direction == Direction.SHORT else "yellow"
            conf_color = "green" if r.confidence >= 70 else "yellow" if r.confidence >= 50 else "white"

            table.add_row(
                r.symbol,
                r.setup.value,
                f"[{dir_color}]{r.direction.value}[/{dir_color}]",
                f"[{conf_color}]{r.confidence:.0f}%[/{conf_color}]",
                r.note,
            )

        console.print(table)

    def evaluate_and_learn(self, symbols: Optional[List[str]] = None) -> None:
        """Evaluate past predictions and adjust weights."""
        symbols = symbols or NIFTY50_SYMBOLS
        console.print("\n[bold cyan]Evaluating predictions...[/bold cyan]")

        data = self.data_engine.get_multiple(symbols)
        counts = self.learning.evaluate_predictions(data)
        console.print(f"  Results: {counts}")

        adjusted = self.learning.adjust_weights()
        self.scoring_engine.update_weights(adjusted)
        console.print(f"  Updated weights applied")

        stats = self.learning.get_stats()
        console.print(f"  Stats: {stats}")

    def _get_sector_data(
        self, symbol: str, timeframe: str, days: int
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for sector peers."""
        for sector, peers in SECTOR_MAP.items():
            if symbol in peers:
                other_peers = [p for p in peers if p != symbol][:3]  # Limit to 3 peers
                return self.data_engine.get_multiple(other_peers, timeframe, days)
        return {}

    def _display_trade(self, trade: TradeDecision) -> None:
        """Display trade decision using Rich formatting."""
        if trade.direction == Direction.WAIT:
            console.print(Panel(
                f"[yellow]WAIT[/yellow] — {trade.reasoning}",
                title=f"[bold]{trade.symbol}[/bold]",
                border_style="yellow",
            ))
            return

        dir_color = "green" if trade.direction == Direction.LONG else "red"
        conf_color = "green" if trade.confidence >= 75 else "yellow" if trade.confidence >= 55 else "red"

        lines = [
            f"  Direction     : [{dir_color}]{trade.direction.value}[/{dir_color}]",
            f"  Entry         : {trade.entry_type.value}",
            f"  Trade Type    : {trade.trade_type.value}",
            f"  Holding       : {trade.holding_period}",
        ]

        if trade.entry_zone:
            lines.append(f"  Entry Zone    : {trade.entry_zone:.2f}")
        if trade.stop_loss:
            lines.append(f"  Stop Loss     : [red]{trade.stop_loss:.2f}[/red]")
        if trade.target_1:
            lines.append(f"  Target 1      : [green]{trade.target_1:.2f}[/green]")
        if trade.target_2:
            lines.append(f"  Target 2      : [green]{trade.target_2:.2f}[/green]")
        if trade.target_3:
            lines.append(f"  Target 3      : [green]{trade.target_3:.2f}[/green]")

        lines.append(f"  Confidence    : [{conf_color}]{trade.confidence:.0f}%[/{conf_color}]")
        lines.append("")
        lines.append(f"  [dim]{trade.reasoning}[/dim]")

        console.print(Panel(
            "\n".join(lines),
            title=f"[bold]{trade.symbol}[/bold]",
            border_style=dir_color,
        ))


# ── CLI ────────────────────────────────────────────────

@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def cli(debug: bool) -> None:
    """Market Intelligence Platform — Institutional-grade trade decisions."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.argument("symbol")
@click.option("--timeframe", "-t", default="1d", help="Candle timeframe (5m, 15m, 1h, 1d, 1wk)")
@click.option("--days", "-d", default=365, help="Historical days to fetch")
@click.option("--no-plot", is_flag=True, help="Skip chart generation")
@click.option("--no-store", is_flag=True, help="Skip prediction storage")
def analyze(symbol: str, timeframe: str, days: int, no_plot: bool, no_store: bool) -> None:
    """Analyze a single stock. Example: market-intel analyze RELIANCE"""
    platform = MarketIntelligencePlatform()
    platform.analyze_stock(
        symbol=symbol.upper(),
        timeframe=timeframe,
        days=days,
        plot=not no_plot,
        store_prediction=not no_store,
    )


@cli.command()
@click.option("--symbols", "-s", default=None, help="Comma-separated symbols (default: NIFTY50)")
@click.option("--timeframe", "-t", default="1d", help="Candle timeframe")
@click.option("--min-confidence", "-c", default=40.0, help="Minimum confidence filter")
def scan(symbols: Optional[str], timeframe: str, min_confidence: float) -> None:
    """Scan universe for trade setups."""
    platform = MarketIntelligencePlatform()
    sym_list = symbols.upper().split(",") if symbols else None
    platform.scan_universe(
        symbols=sym_list,
        timeframe=timeframe,
        min_confidence=min_confidence,
    )


@cli.command()
@click.argument("symbols", nargs=-1)
def batch(symbols: tuple) -> None:
    """Analyze multiple stocks. Example: market-intel batch RELIANCE TCS INFY"""
    platform = MarketIntelligencePlatform()
    sym_list = [s.upper() for s in symbols] if symbols else NIFTY50_SYMBOLS[:10]

    results: List[TradeDecision] = []
    for sym in sym_list:
        trade = platform.analyze_stock(sym, plot=True)
        results.append(trade)

    # Summary table
    console.print("\n")
    table = Table(title="Batch Analysis Summary", show_header=True, header_style="bold magenta")
    table.add_column("Symbol", style="cyan", width=12)
    table.add_column("Direction", width=10)
    table.add_column("Entry", width=10)
    table.add_column("Type", width=10)
    table.add_column("Holding", width=12)
    table.add_column("Entry Zone", justify="right", width=12)
    table.add_column("Stop Loss", justify="right", width=12)
    table.add_column("Target 1", justify="right", width=12)
    table.add_column("Confidence", justify="right", width=12)

    for t in results:
        dir_color = "green" if t.direction == Direction.LONG else "red" if t.direction == Direction.SHORT else "yellow"
        table.add_row(
            t.symbol,
            f"[{dir_color}]{t.direction.value}[/{dir_color}]",
            t.entry_type.value if t.direction != Direction.WAIT else "-",
            t.trade_type.value if t.direction != Direction.WAIT else "-",
            t.holding_period or "-",
            f"{t.entry_zone:.2f}" if t.entry_zone else "-",
            f"{t.stop_loss:.2f}" if t.stop_loss else "-",
            f"{t.target_1:.2f}" if t.target_1 else "-",
            f"{t.confidence:.0f}%",
        )

    console.print(table)


@cli.command()
def learn() -> None:
    """Evaluate past predictions and adjust scoring weights."""
    platform = MarketIntelligencePlatform()
    platform.evaluate_and_learn()


@cli.command()
def stats() -> None:
    """Show prediction statistics."""
    learning = LearningSystem()
    stats_data = learning.get_stats()

    console.print(Panel(
        f"Total Predictions: {stats_data['total_predictions']}\n"
        f"Outcomes: {stats_data['outcomes']}\n"
        f"Win Rate: {stats_data['win_rate']}",
        title="[bold]Prediction Statistics[/bold]",
    ))


@cli.command()
@click.option("--symbol", "-s", default=None, help="Clear cache for specific symbol")
def clear_cache(symbol: Optional[str]) -> None:
    """Clear cached data."""
    engine = DataEngine()
    engine.clear_cache(symbol)
    if symbol:
        console.print(f"[green]Cache cleared for {symbol}[/green]")
    else:
        console.print("[green]All cache cleared[/green]")


if __name__ == "__main__":
    cli()
