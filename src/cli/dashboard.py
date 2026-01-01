"""Rich terminal dashboard for bot status."""

import asyncio
from dataclasses import dataclass
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.core.account_manager import AccountManager
from src.core.delta_calculator import DeltaCalculator
from src.core.position_manager import PositionManager
from src.xp.volume_tracker import VolumeTracker


@dataclass
class Dashboard:
    """Rich terminal dashboard for bot status.

    Displays real-time information about:
    - Account status
    - Position summary
    - Delta exposure
    - Volume and XP estimates
    """

    account_manager: AccountManager
    position_manager: PositionManager
    delta_calculator: DeltaCalculator
    volume_tracker: VolumeTracker
    refresh_rate: float = 1.0

    def __post_init__(self):
        """Initialize console."""
        self.console = Console()
        self._running = False

    def create_layout(self) -> Layout:
        """Create dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )

        layout["left"].split_column(
            Layout(name="accounts"),
            Layout(name="positions"),
        )

        layout["right"].split_column(
            Layout(name="delta"),
            Layout(name="volume"),
        )

        return layout

    def render_header(self) -> Panel:
        """Render header panel."""
        return Panel(
            Text("Paradex Delta-Neutral Bot", style="bold magenta", justify="center"),
            style="blue",
        )

    def render_footer(self) -> Panel:
        """Render footer panel."""
        return Panel(
            Text("Press Ctrl+C to exit", style="dim", justify="center"),
            style="dim",
        )

    def render_accounts_table(self) -> Panel:
        """Render accounts status table."""
        table = Table(title="Accounts", expand=True)
        table.add_column("Alias", style="cyan")
        table.add_column("Role", style="green")
        table.add_column("Position", justify="right")
        table.add_column("PnL", justify="right")

        for account in self.account_manager.get_active_accounts():
            positions = self.position_manager.get_account_positions(account.account_id)
            total_pnl = sum(float(p.unrealized_pnl) for p in positions)
            total_size = sum(float(p.size) for p in positions)

            pnl_style = "green" if total_pnl >= 0 else "red"

            table.add_row(
                account.alias,
                account.role.value.upper(),
                f"${total_size:,.2f}",
                f"[{pnl_style}]${total_pnl:,.2f}[/{pnl_style}]",
            )

        return Panel(table, title="Account Status")

    def render_positions_table(self) -> Panel:
        """Render positions table."""
        table = Table(title="Positions", expand=True)
        table.add_column("Account", style="cyan")
        table.add_column("Market")
        table.add_column("Side")
        table.add_column("Size", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Duration", justify="right")

        for pos in self.position_manager.get_all_positions():
            side_style = "green" if pos.side == "LONG" else "red"
            table.add_row(
                pos.account_id[:8],
                pos.market,
                f"[{side_style}]{pos.side}[/{side_style}]",
                f"{pos.size}",
                f"${pos.entry_price:,.2f}",
                f"{pos.duration_hours:.1f}h",
            )

        return Panel(table, title="Open Positions")

    def render_delta_panel(self) -> Panel:
        """Render delta exposure panel."""
        exposure = self.account_manager.get_total_exposure()

        delta_pct = 0.0
        if exposure["gross"] > 0:
            delta_pct = abs(exposure["net"]) / exposure["gross"] * 100

        status = "[green]NEUTRAL[/green]" if delta_pct < 5 else "[red]UNBALANCED[/red]"

        content = f"""
Long Exposure:  ${exposure['long']:,.2f}
Short Exposure: ${exposure['short']:,.2f}
Net Delta:      ${exposure['net']:,.2f}
Delta %:        {delta_pct:.2f}%

Status: {status}
        """

        border_style = "green" if delta_pct < 5 else "red"
        return Panel(content.strip(), title="Delta Exposure", border_style=border_style)

    def render_volume_panel(self) -> Panel:
        """Render volume and XP panel."""
        volume_24h = self.volume_tracker.get_total_volume_24h()
        volume_7d = self.volume_tracker.get_total_volume_7d()

        # Rough XP estimate
        weekly_pool = 4_000_000
        estimated_xp = weekly_pool * 0.001  # Placeholder calculation

        content = f"""
Volume (24h): ${volume_24h:,.2f}
Volume (7d):  ${volume_7d:,.2f}

Estimated XP: ~{estimated_xp:,.0f}
Weekly Pool:  4,000,000
        """

        return Panel(content.strip(), title="Volume & XP")

    async def run(self) -> None:
        """Run the live dashboard."""
        self._running = True
        layout = self.create_layout()

        with Live(layout, refresh_per_second=1/self.refresh_rate, console=self.console) as live:
            while self._running:
                try:
                    # Update layout components
                    layout["header"].update(self.render_header())
                    layout["accounts"].update(self.render_accounts_table())
                    layout["positions"].update(self.render_positions_table())
                    layout["delta"].update(self.render_delta_panel())
                    layout["volume"].update(self.render_volume_panel())
                    layout["footer"].update(self.render_footer())

                    await asyncio.sleep(self.refresh_rate)

                except asyncio.CancelledError:
                    break

    def stop(self) -> None:
        """Stop the dashboard."""
        self._running = False
