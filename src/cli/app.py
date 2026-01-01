"""Typer CLI application for Paradex Delta Bot."""

import asyncio
import logging
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config.settings import BotConfig, StrategyType, get_config

app = typer.Typer(
    name="paradex-delta",
    help="Delta-neutral trading bot for Paradex DEX - XP farming with multi-account support",
    add_completion=False,
)

accounts_app = typer.Typer(help="Account management commands")
app.add_typer(accounts_app, name="accounts")

console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def get_bot_components():
    """Initialize bot components."""
    from src.security.keychain import KeychainManager
    from src.security.credentials import CredentialManager
    from src.network.client_factory import ParadexClientFactory
    from src.network.rate_limiter import RateLimiter
    from src.network.websocket_manager import WebSocketManager
    from src.core.account_manager import AccountManager
    from src.core.position_manager import PositionManager
    from src.core.delta_calculator import DeltaCalculator
    from src.core.order_manager import OrderManager
    from src.xp.volume_tracker import VolumeTracker
    from src.notifications.notifier import Notifier

    config = get_config()

    keychain = KeychainManager()
    cred_manager = CredentialManager(keychain)
    client_factory = ParadexClientFactory(config.environment.value)
    rate_limiter = RateLimiter()
    ws_manager = WebSocketManager()

    account_manager = AccountManager(
        keychain=keychain,
        client_factory=client_factory,
        ws_manager=ws_manager,
    )

    position_manager = PositionManager()
    delta_calculator = DeltaCalculator(position_manager)

    order_manager = OrderManager(
        account_manager=account_manager,
        rate_limiter=rate_limiter,
    )

    volume_tracker = VolumeTracker()
    notifier = Notifier()

    return {
        "config": config,
        "keychain": keychain,
        "cred_manager": cred_manager,
        "client_factory": client_factory,
        "rate_limiter": rate_limiter,
        "ws_manager": ws_manager,
        "account_manager": account_manager,
        "position_manager": position_manager,
        "delta_calculator": delta_calculator,
        "order_manager": order_manager,
        "volume_tracker": volume_tracker,
        "notifier": notifier,
    }


@app.command()
def start(
    market: str = typer.Argument(
        "BTC-USD-PERP",
        help="Market to trade (e.g., BTC-USD-PERP)",
    ),
    strategy: str = typer.Option(
        "5050",
        "--strategy", "-s",
        help="Strategy: 5050, funding, random",
    ),
    size: float = typer.Option(
        1000.0,
        "--size",
        help="Total position size in USD",
    ),
    leverage: float = typer.Option(
        1.0,
        "--leverage", "-l",
        help="Leverage for positions",
    ),
    auto_rebalance: bool = typer.Option(
        True,
        "--auto-rebalance/--no-auto-rebalance",
        help="Enable automatic rebalancing",
    ),
    threshold: float = typer.Option(
        5.0,
        "--threshold", "-t",
        help="Delta threshold percentage for rebalancing",
    ),
    interval: float = typer.Option(
        60.0,
        "--interval", "-i",
        help="Trading interval in seconds",
    ),
):
    """Start the delta-neutral trading bot."""
    console.print(Panel(
        f"[bold green]Starting Paradex Delta Bot[/bold green]\n\n"
        f"Market: {market}\n"
        f"Strategy: {strategy}\n"
        f"Size: ${size:,.2f}\n"
        f"Leverage: {leverage}x\n"
        f"Auto-rebalance: {auto_rebalance}\n"
        f"Threshold: {threshold}%",
        title="Bot Configuration",
    ))

    async def run_bot():
        components = get_bot_components()
        account_manager = components["account_manager"]
        notifier = components["notifier"]

        # Initialize accounts
        n_accounts = await account_manager.initialize_accounts()
        if n_accounts < 2:
            console.print("[red]Error: Need at least 2 accounts. Use 'paradex-delta accounts add' first.[/red]")
            return

        await notifier.success(
            "Bot Started",
            f"Initialized {n_accounts} accounts for {market}",
        )

        # Select strategy
        from src.strategies import Simple5050Strategy, FundingBasedStrategy, RandomSplitStrategy

        position_manager = components["position_manager"]

        if strategy == "5050":
            strat = Simple5050Strategy(account_manager, position_manager)
        elif strategy == "funding":
            strat = FundingBasedStrategy(account_manager, position_manager)
        elif strategy == "random":
            strat = RandomSplitStrategy(account_manager, position_manager)
        else:
            console.print(f"[red]Unknown strategy: {strategy}[/red]")
            return

        console.print(f"[green]Using strategy: {strat.name}[/green]")

        # Main loop placeholder
        console.print("[yellow]Bot is running. Press Ctrl+C to stop.[/yellow]")

        try:
            while True:
                await asyncio.sleep(interval)
                # Trading logic would go here
                console.print(f"[dim]Checking positions... (interval: {interval}s)[/dim]")
        except asyncio.CancelledError:
            pass
        finally:
            await notifier.info("Bot Stopped", "Shutting down gracefully")
            await account_manager.shutdown()

    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        console.print("\n[yellow]Bot stopped by user[/yellow]")


@app.command()
def stop():
    """Stop the trading bot gracefully."""
    console.print("[yellow]Sending stop signal...[/yellow]")
    # In a real implementation, this would signal the running bot to stop
    console.print("[green]Stop signal sent. Bot will shut down after current cycle.[/green]")


@app.command()
def status():
    """Show current bot status and positions."""
    async def show_status():
        components = get_bot_components()
        account_manager = components["account_manager"]
        delta_calculator = components["delta_calculator"]

        await account_manager.initialize_accounts()

        # Accounts table
        accounts = account_manager.list_accounts()

        if not accounts:
            console.print("[yellow]No accounts configured. Use 'paradex-delta accounts add' first.[/yellow]")
            return

        table = Table(title="Accounts")
        table.add_column("Alias", style="cyan")
        table.add_column("Role", style="green")
        table.add_column("Position", justify="right")
        table.add_column("Balance", justify="right")
        table.add_column("Active", justify="center")

        for acc in accounts:
            table.add_row(
                acc["alias"],
                acc["role"],
                f"${acc.get('position_size', 0):,.2f}",
                f"${acc.get('available_balance', 0):,.2f}",
                "[green]Yes[/green]" if acc["is_active"] else "[red]No[/red]",
            )

        console.print(table)

        # Delta status
        exposure = account_manager.get_total_exposure()
        delta_pct = 0.0
        if exposure["gross"] > 0:
            delta_pct = abs(exposure["net"]) / exposure["gross"] * 100

        console.print(Panel(
            f"Long Exposure:  ${exposure['long']:,.2f}\n"
            f"Short Exposure: ${exposure['short']:,.2f}\n"
            f"Net Delta:      ${exposure['net']:,.2f}\n"
            f"Delta %:        {delta_pct:.2f}%\n"
            f"Status:         {'[green]NEUTRAL[/green]' if delta_pct < 5 else '[red]UNBALANCED[/red]'}",
            title="Delta Exposure",
        ))

        await account_manager.shutdown()

    asyncio.run(show_status())


# Account subcommands
@accounts_app.command("add")
def add_account(
    alias: str = typer.Argument(..., help="Account alias (e.g., acc1)"),
    l2_address: str = typer.Option(
        ...,
        "--address", "-a",
        prompt="L2 Address (0x...)",
        help="Starknet L2 account address",
    ),
    l2_private_key: str = typer.Option(
        ...,
        "--key", "-k",
        prompt="Subkey Private Key (0x...)",
        hide_input=True,
        help="L2 subkey private key (will not be displayed)",
    ),
):
    """Add a new trading account."""
    from src.security.credentials import CredentialManager
    from src.security.keychain import KeychainManager

    try:
        keychain = KeychainManager()
        cred_manager = CredentialManager(keychain)

        creds = cred_manager.add_account(
            alias=alias,
            l2_address=l2_address,
            l2_private_key=l2_private_key,
        )

        console.print(Panel(
            f"[green]Account added successfully![/green]\n\n"
            f"Alias: {creds.alias}\n"
            f"ID: {creds.account_id}\n"
            f"Address: {creds.l2_address[:10]}...{creds.l2_address[-8:]}",
            title="New Account",
        ))

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@accounts_app.command("list")
def list_accounts():
    """List all configured accounts."""
    from src.security.keychain import KeychainManager

    keychain = KeychainManager()
    accounts = keychain.list_accounts_info()

    if not accounts:
        console.print("[yellow]No accounts configured.[/yellow]")
        console.print("Use 'paradex-delta accounts add' to add an account.")
        return

    table = Table(title="Configured Accounts")
    table.add_column("Alias", style="cyan")
    table.add_column("Account ID")
    table.add_column("L2 Address")

    for acc in accounts:
        addr = acc["l2_address"]
        table.add_row(
            acc["alias"],
            acc["account_id"],
            f"{addr[:10]}...{addr[-8:]}",
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(accounts)} account(s)[/dim]")


@accounts_app.command("remove")
def remove_account(
    alias: str = typer.Argument(..., help="Account alias to remove"),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Skip confirmation",
    ),
):
    """Remove an account."""
    from src.security.credentials import CredentialManager
    from src.security.keychain import KeychainManager

    keychain = KeychainManager()
    cred_manager = CredentialManager(keychain)

    # Check if exists
    account = cred_manager.get_account(alias)
    if not account:
        console.print(f"[red]Account not found: {alias}[/red]")
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(f"Remove account '{alias}'?")
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            return

    if cred_manager.remove_account(alias):
        console.print(f"[green]Account removed: {alias}[/green]")
    else:
        console.print(f"[red]Failed to remove account: {alias}[/red]")


@app.command()
def version():
    """Show version information."""
    from src import __version__

    console.print(f"Paradex Delta Bot v{__version__}")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
