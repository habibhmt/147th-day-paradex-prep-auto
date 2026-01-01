"""Enhanced statistics dashboard for trading bot."""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text


class DashboardTheme(Enum):
    """Dashboard color themes."""

    DARK = "dark"
    LIGHT = "light"
    HACKER = "hacker"


@dataclass
class MetricValue:
    """A metric with history tracking."""

    name: str
    current: float = 0.0
    previous: float = 0.0
    unit: str = ""
    format_spec: str = ".2f"

    @property
    def change(self) -> float:
        """Calculate change from previous."""
        return self.current - self.previous

    @property
    def change_pct(self) -> float:
        """Calculate percentage change."""
        if self.previous == 0:
            return 0.0
        return ((self.current - self.previous) / abs(self.previous)) * 100

    @property
    def trend(self) -> str:
        """Get trend indicator."""
        if self.change > 0:
            return "↑"
        elif self.change < 0:
            return "↓"
        return "→"

    @property
    def trend_color(self) -> str:
        """Get color for trend."""
        if self.change > 0:
            return "green"
        elif self.change < 0:
            return "red"
        return "yellow"

    def formatted_value(self) -> str:
        """Get formatted value string."""
        return f"{self.current:{self.format_spec}}{self.unit}"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "current": self.current,
            "previous": self.previous,
            "change": self.change,
            "change_pct": self.change_pct,
            "unit": self.unit,
        }


@dataclass
class StatsDashboardConfig:
    """Configuration for stats dashboard."""

    refresh_rate: float = 1.0
    theme: DashboardTheme = DashboardTheme.DARK
    show_charts: bool = True
    max_history: int = 60  # Keep 60 data points
    compact_mode: bool = False


@dataclass
class StatsDashboard:
    """Enhanced statistics dashboard.

    Features:
    - Real-time metrics with trends
    - Performance charts (ASCII)
    - Trade history table
    - Risk indicators
    - XP progress tracking
    """

    config: StatsDashboardConfig = field(default_factory=StatsDashboardConfig)
    _console: Console = field(default_factory=Console)
    _running: bool = False
    _start_time: float = 0.0
    _metrics: Dict[str, MetricValue] = field(default_factory=dict)
    _pnl_history: List[float] = field(default_factory=list)
    _volume_history: List[float] = field(default_factory=list)
    _trade_count: int = 0
    _data_provider: Any = None  # Will be set with actual data sources

    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._console = Console()
        self._running = False
        self._start_time = time.time()
        self._metrics = {}
        self._pnl_history = []
        self._volume_history = []
        self._trade_count = 0
        self._init_metrics()

    def _init_metrics(self):
        """Initialize tracked metrics."""
        self._metrics = {
            "pnl": MetricValue("PnL", unit="$"),
            "volume_24h": MetricValue("Volume 24h", unit="$", format_spec=",.0f"),
            "volume_7d": MetricValue("Volume 7d", unit="$", format_spec=",.0f"),
            "win_rate": MetricValue("Win Rate", unit="%", format_spec=".1f"),
            "delta_pct": MetricValue("Delta", unit="%", format_spec=".2f"),
            "trades_today": MetricValue("Trades", format_spec=".0f"),
            "estimated_xp": MetricValue("Est. XP", format_spec=",.0f"),
            "drawdown": MetricValue("Drawdown", unit="%", format_spec=".2f"),
            "profit_factor": MetricValue("Profit Factor", format_spec=".2f"),
            "accounts_active": MetricValue("Active Accounts", format_spec=".0f"),
        }

    def set_data_provider(self, provider: Any) -> None:
        """Set the data provider for dashboard updates."""
        self._data_provider = provider

    def update_metric(
        self,
        name: str,
        value: float,
    ) -> None:
        """Update a metric value.

        Args:
            name: Metric name
            value: New value
        """
        if name in self._metrics:
            metric = self._metrics[name]
            metric.previous = metric.current
            metric.current = value

    def update_metrics(self, data: Dict[str, float]) -> None:
        """Update multiple metrics at once.

        Args:
            data: Dictionary of metric name to value
        """
        for name, value in data.items():
            self.update_metric(name, value)

    def record_pnl(self, pnl: float) -> None:
        """Record PnL for history tracking."""
        self._pnl_history.append(pnl)
        if len(self._pnl_history) > self.config.max_history:
            self._pnl_history = self._pnl_history[-self.config.max_history:]

    def record_volume(self, volume: float) -> None:
        """Record volume for history tracking."""
        self._volume_history.append(volume)
        if len(self._volume_history) > self.config.max_history:
            self._volume_history = self._volume_history[-self.config.max_history:]

    def create_layout(self) -> Layout:
        """Create enhanced dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        if self.config.compact_mode:
            layout["body"].split_row(
                Layout(name="metrics", ratio=2),
                Layout(name="charts", ratio=1),
            )
        else:
            layout["body"].split_column(
                Layout(name="top_row", size=12),
                Layout(name="bottom_row"),
            )

            layout["top_row"].split_row(
                Layout(name="key_metrics"),
                Layout(name="performance"),
            )

            layout["bottom_row"].split_row(
                Layout(name="trades"),
                Layout(name="risk"),
            )

        return layout

    def render_header(self) -> Panel:
        """Render header with status."""
        uptime = time.time() - self._start_time
        uptime_str = str(timedelta(seconds=int(uptime)))

        status = "[green]● RUNNING[/green]" if self._running else "[red]● STOPPED[/red]"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        header_text = Text()
        header_text.append("Paradex Delta-Neutral Bot ", style="bold magenta")
        header_text.append(f"{status} ", style="")
        header_text.append(f"| Uptime: {uptime_str} ", style="dim")
        header_text.append(f"| {timestamp}", style="dim")

        return Panel(header_text, style="blue")

    def render_footer(self) -> Panel:
        """Render footer."""
        footer_text = Text()
        footer_text.append("Press ", style="dim")
        footer_text.append("Ctrl+C", style="bold")
        footer_text.append(" to exit | ", style="dim")
        footer_text.append("R", style="bold")
        footer_text.append(" to refresh | ", style="dim")
        footer_text.append("T", style="bold")
        footer_text.append(" to toggle theme", style="dim")

        return Panel(footer_text, style="dim")

    def render_key_metrics(self) -> Panel:
        """Render key metrics panel."""
        table = Table(show_header=True, header_style="bold cyan", expand=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_column("Change", justify="right")
        table.add_column("Trend", justify="center")

        key_metrics = ["pnl", "volume_24h", "win_rate", "delta_pct", "estimated_xp"]

        for name in key_metrics:
            metric = self._metrics.get(name)
            if metric:
                trend_styled = f"[{metric.trend_color}]{metric.trend}[/{metric.trend_color}]"
                change_styled = f"[{metric.trend_color}]{metric.change:+.2f}[/{metric.trend_color}]"

                table.add_row(
                    metric.name,
                    metric.formatted_value(),
                    change_styled,
                    trend_styled,
                )

        return Panel(table, title="Key Metrics", border_style="green")

    def render_performance(self) -> Panel:
        """Render performance panel with mini charts."""
        content_lines = []

        # PnL mini chart
        if self._pnl_history:
            chart = self._create_sparkline(self._pnl_history, width=30)
            content_lines.append(f"PnL Trend: {chart}")

        # Volume mini chart
        if self._volume_history:
            chart = self._create_sparkline(self._volume_history, width=30)
            content_lines.append(f"Volume:    {chart}")

        # Performance metrics
        pnl = self._metrics.get("pnl")
        if pnl:
            pnl_color = "green" if pnl.current >= 0 else "red"
            content_lines.append(f"Total PnL: [{pnl_color}]${pnl.current:,.2f}[/{pnl_color}]")

        profit_factor = self._metrics.get("profit_factor")
        if profit_factor:
            pf_color = "green" if profit_factor.current > 1 else "red"
            content_lines.append(
                f"Profit Factor: [{pf_color}]{profit_factor.current:.2f}[/{pf_color}]"
            )

        drawdown = self._metrics.get("drawdown")
        if drawdown:
            dd_color = "green" if drawdown.current < 5 else "yellow" if drawdown.current < 10 else "red"
            content_lines.append(f"Drawdown: [{dd_color}]{drawdown.current:.2f}%[/{dd_color}]")

        content = "\n".join(content_lines) if content_lines else "No data available"
        return Panel(content, title="Performance", border_style="blue")

    def render_trades_table(self, trades: List[Dict] = None) -> Panel:
        """Render recent trades table."""
        table = Table(show_header=True, header_style="bold cyan", expand=True)
        table.add_column("Time", style="dim")
        table.add_column("Market")
        table.add_column("Side")
        table.add_column("Size", justify="right")
        table.add_column("PnL", justify="right")

        # Use sample trades if no real data
        sample_trades = trades or []

        for trade in sample_trades[-10:]:  # Show last 10
            side_style = "green" if trade.get("side") == "BUY" else "red"
            pnl = trade.get("pnl", 0)
            pnl_style = "green" if pnl >= 0 else "red"

            table.add_row(
                trade.get("time", ""),
                trade.get("market", ""),
                f"[{side_style}]{trade.get('side', '')}[/{side_style}]",
                f"{trade.get('size', 0):,.2f}",
                f"[{pnl_style}]${pnl:,.2f}[/{pnl_style}]",
            )

        if not sample_trades:
            table.add_row("No trades", "-", "-", "-", "-")

        return Panel(table, title=f"Recent Trades ({len(sample_trades)})", border_style="cyan")

    def render_risk_panel(self) -> Panel:
        """Render risk indicators panel."""
        lines = []

        # Delta status
        delta = self._metrics.get("delta_pct")
        if delta:
            delta_status = (
                "[green]● NEUTRAL[/green]" if delta.current < 5
                else "[yellow]● WARNING[/yellow]" if delta.current < 10
                else "[red]● CRITICAL[/red]"
            )
            lines.append(f"Delta Status: {delta_status} ({delta.current:.2f}%)")

        # Drawdown status
        drawdown = self._metrics.get("drawdown")
        if drawdown:
            dd_status = (
                "[green]● LOW[/green]" if drawdown.current < 5
                else "[yellow]● ELEVATED[/yellow]" if drawdown.current < 10
                else "[red]● HIGH[/red]"
            )
            lines.append(f"Drawdown: {dd_status} ({drawdown.current:.2f}%)")

        # Progress bars
        lines.append("")
        lines.append("Risk Utilization:")

        # XP Progress
        xp = self._metrics.get("estimated_xp")
        if xp:
            weekly_target = 100000  # Example target
            progress = min(xp.current / weekly_target * 100, 100)
            bar = self._create_progress_bar(progress, width=20)
            lines.append(f"XP Target: {bar} {progress:.0f}%")

        # Volume Progress
        vol = self._metrics.get("volume_24h")
        if vol:
            daily_target = 500000  # Example target
            progress = min(vol.current / daily_target * 100, 100)
            bar = self._create_progress_bar(progress, width=20)
            lines.append(f"Volume:    {bar} {progress:.0f}%")

        content = "\n".join(lines) if lines else "No risk data"
        return Panel(content, title="Risk Monitor", border_style="yellow")

    def _create_sparkline(self, data: List[float], width: int = 20) -> str:
        """Create ASCII sparkline chart."""
        if not data:
            return "─" * width

        chars = "▁▂▃▄▅▆▇█"

        # Normalize data
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val if max_val != min_val else 1

        # Sample if too many points
        if len(data) > width:
            step = len(data) / width
            data = [data[int(i * step)] for i in range(width)]

        # Create sparkline
        result = ""
        for val in data:
            normalized = (val - min_val) / range_val
            char_idx = min(int(normalized * (len(chars) - 1)), len(chars) - 1)
            result += chars[char_idx]

        return result

    def _create_progress_bar(
        self,
        percent: float,
        width: int = 20,
    ) -> str:
        """Create ASCII progress bar."""
        filled = int(width * percent / 100)
        empty = width - filled

        bar = "█" * filled + "░" * empty
        return f"[{bar}]"

    async def run(self, refresh_data_callback=None) -> None:
        """Run the live dashboard.

        Args:
            refresh_data_callback: Optional async callback to refresh data
        """
        self._running = True
        layout = self.create_layout()

        with Live(
            layout,
            refresh_per_second=1 / self.config.refresh_rate,
            console=self._console,
        ) as live:
            while self._running:
                try:
                    # Refresh data if callback provided
                    if refresh_data_callback:
                        await refresh_data_callback()

                    # Update layout
                    layout["header"].update(self.render_header())
                    layout["footer"].update(self.render_footer())

                    if not self.config.compact_mode:
                        layout["key_metrics"].update(self.render_key_metrics())
                        layout["performance"].update(self.render_performance())
                        layout["trades"].update(self.render_trades_table())
                        layout["risk"].update(self.render_risk_panel())

                    await asyncio.sleep(self.config.refresh_rate)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self._console.print(f"[red]Dashboard error: {e}[/red]")
                    await asyncio.sleep(1)

    def stop(self) -> None:
        """Stop the dashboard."""
        self._running = False

    def get_metrics(self) -> Dict[str, Dict]:
        """Get all current metrics.

        Returns:
            Dictionary of metric data
        """
        return {name: metric.to_dict() for name, metric in self._metrics.items()}

    def get_status(self) -> Dict:
        """Get dashboard status.

        Returns:
            Status dictionary
        """
        return {
            "running": self._running,
            "uptime_seconds": time.time() - self._start_time,
            "refresh_rate": self.config.refresh_rate,
            "theme": self.config.theme.value,
            "pnl_history_points": len(self._pnl_history),
            "volume_history_points": len(self._volume_history),
        }
