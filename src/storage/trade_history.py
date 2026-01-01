"""SQLite storage for trade history and analytics."""

import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TradeEntry:
    """A single trade entry."""

    id: Optional[int]
    account_id: str
    market: str
    side: str  # BUY or SELL
    size: Decimal
    price: Decimal
    order_id: str
    client_id: str
    timestamp: float
    pnl: Optional[Decimal] = None
    fee: Optional[Decimal] = None
    strategy: str = ""

    @property
    def volume(self) -> Decimal:
        """Calculate trade volume."""
        return self.size * self.price

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "account_id": self.account_id,
            "market": self.market,
            "side": self.side,
            "size": str(self.size),
            "price": str(self.price),
            "volume": str(self.volume),
            "order_id": self.order_id,
            "timestamp": self.timestamp,
            "pnl": str(self.pnl) if self.pnl else None,
            "fee": str(self.fee) if self.fee else None,
            "strategy": self.strategy,
        }


class TradeHistoryDB:
    """SQLite database for trade history.

    Features:
    - Persistent trade storage
    - Performance analytics
    - Account-specific queries
    - PnL tracking
    """

    def __init__(self, db_path: str = "trade_history.db"):
        """Initialize trade history database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    market TEXT NOT NULL,
                    side TEXT NOT NULL,
                    size TEXT NOT NULL,
                    price TEXT NOT NULL,
                    order_id TEXT,
                    client_id TEXT,
                    timestamp REAL NOT NULL,
                    pnl TEXT,
                    fee TEXT,
                    strategy TEXT DEFAULT '',
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_account
                ON trades(account_id)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_market
                ON trades(market)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_timestamp
                ON trades(timestamp)
            """)

            # Rebalance history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rebalances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    delta_before TEXT NOT NULL,
                    delta_after TEXT,
                    orders_planned INTEGER,
                    orders_executed INTEGER,
                    success INTEGER,
                    timestamp REAL NOT NULL,
                    error_message TEXT
                )
            """)

            # Daily summary table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    account_id TEXT NOT NULL,
                    total_volume TEXT NOT NULL,
                    trade_count INTEGER,
                    total_pnl TEXT,
                    total_fees TEXT,
                    avg_trade_size TEXT
                )
            """)

            conn.commit()
            logger.info(f"Database initialized: {self.db_path}")

    @contextmanager
    def _connect(self):
        """Context manager for database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def add_trade(self, trade: TradeEntry) -> int:
        """Add a trade to history.

        Args:
            trade: TradeEntry to add

        Returns:
            ID of inserted trade
        """
        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO trades
                (account_id, market, side, size, price, order_id, client_id,
                 timestamp, pnl, fee, strategy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.account_id,
                trade.market,
                trade.side,
                str(trade.size),
                str(trade.price),
                trade.order_id,
                trade.client_id,
                trade.timestamp,
                str(trade.pnl) if trade.pnl else None,
                str(trade.fee) if trade.fee else None,
                trade.strategy,
            ))
            conn.commit()
            return cursor.lastrowid

    def get_trades(
        self,
        account_id: Optional[str] = None,
        market: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100,
    ) -> List[TradeEntry]:
        """Get trades with optional filters.

        Args:
            account_id: Filter by account
            market: Filter by market
            start_time: Filter by start timestamp
            end_time: Filter by end timestamp
            limit: Maximum trades to return

        Returns:
            List of TradeEntry objects
        """
        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if account_id:
            query += " AND account_id = ?"
            params.append(account_id)

        if market:
            query += " AND market = ?"
            params.append(market)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_trade(row) for row in rows]

    def _row_to_trade(self, row: sqlite3.Row) -> TradeEntry:
        """Convert database row to TradeEntry."""
        return TradeEntry(
            id=row["id"],
            account_id=row["account_id"],
            market=row["market"],
            side=row["side"],
            size=Decimal(row["size"]),
            price=Decimal(row["price"]),
            order_id=row["order_id"] or "",
            client_id=row["client_id"] or "",
            timestamp=row["timestamp"],
            pnl=Decimal(row["pnl"]) if row["pnl"] else None,
            fee=Decimal(row["fee"]) if row["fee"] else None,
            strategy=row["strategy"] or "",
        )

    def get_volume_by_account(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Dict[str, Decimal]:
        """Get total volume per account.

        Args:
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            Dictionary of {account_id: total_volume}
        """
        query = """
            SELECT account_id,
                   SUM(CAST(size AS REAL) * CAST(price AS REAL)) as volume
            FROM trades
            WHERE 1=1
        """
        params = []

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " GROUP BY account_id"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return {row["account_id"]: Decimal(str(row["volume"])) for row in rows}

    def get_total_volume(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Decimal:
        """Get total volume across all accounts.

        Args:
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            Total volume
        """
        volumes = self.get_volume_by_account(start_time, end_time)
        return sum(volumes.values(), Decimal("0"))

    def get_trade_count(
        self,
        account_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> int:
        """Get trade count.

        Args:
            account_id: Filter by account
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            Number of trades
        """
        query = "SELECT COUNT(*) as count FROM trades WHERE 1=1"
        params = []

        if account_id:
            query += " AND account_id = ?"
            params.append(account_id)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()

        return row["count"]

    def get_total_pnl(
        self,
        account_id: Optional[str] = None,
        start_time: Optional[float] = None,
    ) -> Decimal:
        """Get total PnL.

        Args:
            account_id: Filter by account
            start_time: Start timestamp

        Returns:
            Total PnL
        """
        query = "SELECT COALESCE(SUM(CAST(pnl AS REAL)), 0) as total FROM trades WHERE pnl IS NOT NULL"
        params = []

        if account_id:
            query += " AND account_id = ?"
            params.append(account_id)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()

        return Decimal(str(row["total"]))

    def add_rebalance(
        self,
        market: str,
        trigger_type: str,
        delta_before: Decimal,
        delta_after: Optional[Decimal],
        orders_planned: int,
        orders_executed: int,
        success: bool,
        error_message: Optional[str] = None,
    ) -> int:
        """Add rebalance event to history.

        Args:
            market: Market symbol
            trigger_type: What triggered rebalance
            delta_before: Delta before rebalance
            delta_after: Delta after rebalance
            orders_planned: Orders planned
            orders_executed: Orders executed
            success: Whether rebalance succeeded
            error_message: Error message if failed

        Returns:
            ID of inserted rebalance
        """
        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO rebalances
                (market, trigger_type, delta_before, delta_after,
                 orders_planned, orders_executed, success, timestamp, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                market,
                trigger_type,
                str(delta_before),
                str(delta_after) if delta_after else None,
                orders_planned,
                orders_executed,
                1 if success else 0,
                time.time(),
                error_message,
            ))
            conn.commit()
            return cursor.lastrowid

    def get_rebalances(
        self,
        market: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """Get rebalance history.

        Args:
            market: Filter by market
            limit: Maximum entries to return

        Returns:
            List of rebalance dictionaries
        """
        query = "SELECT * FROM rebalances WHERE 1=1"
        params = []

        if market:
            query += " AND market = ?"
            params.append(market)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [dict(row) for row in rows]

    def get_daily_summary(self, date: str) -> Optional[Dict]:
        """Get summary for a specific date.

        Args:
            date: Date string (YYYY-MM-DD)

        Returns:
            Summary dictionary or None
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM daily_summaries WHERE date = ?",
                (date,)
            ).fetchone()

        return dict(row) if row else None

    def generate_daily_summary(self, date: str, account_id: str) -> Dict:
        """Generate and store daily summary.

        Args:
            date: Date string (YYYY-MM-DD)
            account_id: Account ID

        Returns:
            Summary dictionary
        """
        # Parse date to timestamps
        dt = datetime.strptime(date, "%Y-%m-%d")
        start_ts = dt.timestamp()
        end_ts = start_ts + 86400

        trades = self.get_trades(
            account_id=account_id,
            start_time=start_ts,
            end_time=end_ts,
            limit=10000,
        )

        total_volume = sum(t.volume for t in trades)
        trade_count = len(trades)
        total_pnl = sum(t.pnl or Decimal("0") for t in trades)
        total_fees = sum(t.fee or Decimal("0") for t in trades)
        avg_size = total_volume / trade_count if trade_count > 0 else Decimal("0")

        summary = {
            "date": date,
            "account_id": account_id,
            "total_volume": str(total_volume),
            "trade_count": trade_count,
            "total_pnl": str(total_pnl),
            "total_fees": str(total_fees),
            "avg_trade_size": str(avg_size),
        }

        # Store summary
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO daily_summaries
                (date, account_id, total_volume, trade_count, total_pnl, total_fees, avg_trade_size)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                date,
                account_id,
                summary["total_volume"],
                summary["trade_count"],
                summary["total_pnl"],
                summary["total_fees"],
                summary["avg_trade_size"],
            ))
            conn.commit()

        return summary

    def export_to_csv(
        self,
        filepath: str,
        account_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> int:
        """Export trades to CSV.

        Args:
            filepath: Output CSV file path
            account_id: Filter by account
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            Number of trades exported
        """
        import csv

        trades = self.get_trades(
            account_id=account_id,
            start_time=start_time,
            end_time=end_time,
            limit=100000,
        )

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "account_id", "market", "side", "size", "price",
                "volume", "order_id", "timestamp", "pnl", "fee", "strategy"
            ])

            for trade in trades:
                writer.writerow([
                    trade.id,
                    trade.account_id,
                    trade.market,
                    trade.side,
                    str(trade.size),
                    str(trade.price),
                    str(trade.volume),
                    trade.order_id,
                    datetime.fromtimestamp(trade.timestamp).isoformat(),
                    str(trade.pnl) if trade.pnl else "",
                    str(trade.fee) if trade.fee else "",
                    trade.strategy,
                ])

        logger.info(f"Exported {len(trades)} trades to {filepath}")
        return len(trades)

    def get_statistics(self) -> Dict:
        """Get overall database statistics.

        Returns:
            Statistics dictionary
        """
        with self._connect() as conn:
            trade_count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            account_count = conn.execute(
                "SELECT COUNT(DISTINCT account_id) FROM trades"
            ).fetchone()[0]
            rebalance_count = conn.execute(
                "SELECT COUNT(*) FROM rebalances"
            ).fetchone()[0]

        total_volume = self.get_total_volume()
        total_pnl = self.get_total_pnl()

        return {
            "total_trades": trade_count,
            "unique_accounts": account_count,
            "total_rebalances": rebalance_count,
            "total_volume": str(total_volume),
            "total_pnl": str(total_pnl),
            "db_path": str(self.db_path),
        }

    def clear_all(self) -> None:
        """Clear all data from database (use with caution)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM trades")
            conn.execute("DELETE FROM rebalances")
            conn.execute("DELETE FROM daily_summaries")
            conn.commit()
        logger.warning("All trade history cleared")
