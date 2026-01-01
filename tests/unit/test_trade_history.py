"""Unit tests for Trade History Database."""

import pytest
import os
import time
from decimal import Decimal
from pathlib import Path
import tempfile

from src.storage.trade_history import TradeHistoryDB, TradeEntry


class TestTradeEntry:
    """Tests for TradeEntry dataclass."""

    def test_create_trade_entry(self):
        """Should create trade entry correctly."""
        entry = TradeEntry(
            id=1,
            account_id="acc1",
            market="BTC-USD-PERP",
            side="BUY",
            size=Decimal("1.5"),
            price=Decimal("50000"),
            order_id="order-123",
            client_id="client-456",
            timestamp=time.time(),
        )

        assert entry.account_id == "acc1"
        assert entry.market == "BTC-USD-PERP"
        assert entry.side == "BUY"
        assert entry.size == Decimal("1.5")

    def test_volume_property(self):
        """Should calculate volume correctly."""
        entry = TradeEntry(
            id=1,
            account_id="acc1",
            market="BTC-USD-PERP",
            side="BUY",
            size=Decimal("2"),
            price=Decimal("50000"),
            order_id="order-123",
            client_id="client-456",
            timestamp=time.time(),
        )

        assert entry.volume == Decimal("100000")

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        entry = TradeEntry(
            id=1,
            account_id="acc1",
            market="BTC-USD-PERP",
            side="SELL",
            size=Decimal("1"),
            price=Decimal("50000"),
            order_id="order-123",
            client_id="client-456",
            timestamp=1234567890.0,
            pnl=Decimal("100"),
            fee=Decimal("5"),
        )

        d = entry.to_dict()

        assert d["account_id"] == "acc1"
        assert d["volume"] == "50000"
        assert d["pnl"] == "100"
        assert d["fee"] == "5"


class TestTradeHistoryDB:
    """Tests for TradeHistoryDB."""

    @pytest.fixture
    def db(self, tmp_path):
        """Create temporary database."""
        db_path = tmp_path / "test_trades.db"
        return TradeHistoryDB(str(db_path))

    def test_init_creates_db(self, db):
        """Should create database file."""
        assert db.db_path.exists()

    def test_add_trade(self, db):
        """Should add trade to database."""
        entry = TradeEntry(
            id=None,
            account_id="acc1",
            market="BTC-USD-PERP",
            side="BUY",
            size=Decimal("1"),
            price=Decimal("50000"),
            order_id="order-123",
            client_id="client-456",
            timestamp=time.time(),
        )

        trade_id = db.add_trade(entry)

        assert trade_id is not None
        assert trade_id > 0

    def test_get_trades(self, db):
        """Should retrieve trades."""
        now = time.time()
        for i in range(5):
            entry = TradeEntry(
                id=None,
                account_id="acc1",
                market="BTC-USD-PERP",
                side="BUY" if i % 2 == 0 else "SELL",
                size=Decimal("1"),
                price=Decimal("50000"),
                order_id=f"order-{i}",
                client_id=f"client-{i}",
                timestamp=now - i * 60,
            )
            db.add_trade(entry)

        trades = db.get_trades()

        assert len(trades) == 5

    def test_get_trades_by_account(self, db):
        """Should filter trades by account."""
        now = time.time()
        for account in ["acc1", "acc1", "acc2"]:
            entry = TradeEntry(
                id=None,
                account_id=account,
                market="BTC-USD-PERP",
                side="BUY",
                size=Decimal("1"),
                price=Decimal("50000"),
                order_id="order-123",
                client_id="client-456",
                timestamp=now,
            )
            db.add_trade(entry)

        acc1_trades = db.get_trades(account_id="acc1")
        acc2_trades = db.get_trades(account_id="acc2")

        assert len(acc1_trades) == 2
        assert len(acc2_trades) == 1

    def test_get_trades_by_market(self, db):
        """Should filter trades by market."""
        now = time.time()
        for market in ["BTC-USD-PERP", "ETH-USD-PERP", "BTC-USD-PERP"]:
            entry = TradeEntry(
                id=None,
                account_id="acc1",
                market=market,
                side="BUY",
                size=Decimal("1"),
                price=Decimal("50000"),
                order_id="order-123",
                client_id="client-456",
                timestamp=now,
            )
            db.add_trade(entry)

        btc_trades = db.get_trades(market="BTC-USD-PERP")
        eth_trades = db.get_trades(market="ETH-USD-PERP")

        assert len(btc_trades) == 2
        assert len(eth_trades) == 1

    def test_get_trades_by_time_range(self, db):
        """Should filter trades by time range."""
        now = time.time()
        for i in range(5):
            entry = TradeEntry(
                id=None,
                account_id="acc1",
                market="BTC-USD-PERP",
                side="BUY",
                size=Decimal("1"),
                price=Decimal("50000"),
                order_id=f"order-{i}",
                client_id=f"client-{i}",
                timestamp=now - i * 3600,  # Each trade 1 hour apart
            )
            db.add_trade(entry)

        # Get trades from last 2.5 hours
        recent_trades = db.get_trades(start_time=now - 9000)

        assert len(recent_trades) == 3

    def test_get_volume_by_account(self, db):
        """Should calculate volume per account."""
        now = time.time()
        # acc1: 2 trades, acc2: 1 trade
        db.add_trade(TradeEntry(None, "acc1", "BTC-USD-PERP", "BUY",
                                Decimal("1"), Decimal("50000"), "", "", now))
        db.add_trade(TradeEntry(None, "acc1", "BTC-USD-PERP", "SELL",
                                Decimal("2"), Decimal("50000"), "", "", now))
        db.add_trade(TradeEntry(None, "acc2", "BTC-USD-PERP", "BUY",
                                Decimal("1"), Decimal("50000"), "", "", now))

        volumes = db.get_volume_by_account()

        assert volumes["acc1"] == Decimal("150000")
        assert volumes["acc2"] == Decimal("50000")

    def test_get_total_volume(self, db):
        """Should calculate total volume."""
        now = time.time()
        db.add_trade(TradeEntry(None, "acc1", "BTC-USD-PERP", "BUY",
                                Decimal("1"), Decimal("50000"), "", "", now))
        db.add_trade(TradeEntry(None, "acc2", "BTC-USD-PERP", "SELL",
                                Decimal("1"), Decimal("50000"), "", "", now))

        total = db.get_total_volume()

        assert total == Decimal("100000")

    def test_get_trade_count(self, db):
        """Should count trades."""
        now = time.time()
        for i in range(7):
            db.add_trade(TradeEntry(None, "acc1", "BTC-USD-PERP", "BUY",
                                    Decimal("1"), Decimal("50000"), "", "", now))

        count = db.get_trade_count()

        assert count == 7

    def test_get_trade_count_by_account(self, db):
        """Should count trades per account."""
        now = time.time()
        for i in range(3):
            db.add_trade(TradeEntry(None, "acc1", "BTC-USD-PERP", "BUY",
                                    Decimal("1"), Decimal("50000"), "", "", now))
        for i in range(5):
            db.add_trade(TradeEntry(None, "acc2", "BTC-USD-PERP", "BUY",
                                    Decimal("1"), Decimal("50000"), "", "", now))

        acc1_count = db.get_trade_count(account_id="acc1")
        acc2_count = db.get_trade_count(account_id="acc2")

        assert acc1_count == 3
        assert acc2_count == 5

    def test_get_total_pnl(self, db):
        """Should calculate total PnL."""
        now = time.time()
        db.add_trade(TradeEntry(None, "acc1", "BTC-USD-PERP", "BUY",
                                Decimal("1"), Decimal("50000"), "", "", now,
                                pnl=Decimal("100")))
        db.add_trade(TradeEntry(None, "acc1", "BTC-USD-PERP", "SELL",
                                Decimal("1"), Decimal("50000"), "", "", now,
                                pnl=Decimal("-30")))

        total_pnl = db.get_total_pnl()

        assert total_pnl == Decimal("70")

    def test_add_rebalance(self, db):
        """Should add rebalance event."""
        rebalance_id = db.add_rebalance(
            market="BTC-USD-PERP",
            trigger_type="threshold",
            delta_before=Decimal("2000"),
            delta_after=Decimal("100"),
            orders_planned=4,
            orders_executed=4,
            success=True,
        )

        assert rebalance_id is not None
        assert rebalance_id > 0

    def test_get_rebalances(self, db):
        """Should retrieve rebalances."""
        for i in range(3):
            db.add_rebalance(
                market="BTC-USD-PERP",
                trigger_type="threshold",
                delta_before=Decimal("2000"),
                delta_after=Decimal("100"),
                orders_planned=4,
                orders_executed=4,
                success=True,
            )

        rebalances = db.get_rebalances()

        assert len(rebalances) == 3

    def test_get_rebalances_by_market(self, db):
        """Should filter rebalances by market."""
        db.add_rebalance("BTC-USD-PERP", "threshold", Decimal("2000"),
                         Decimal("100"), 4, 4, True)
        db.add_rebalance("ETH-USD-PERP", "manual", Decimal("1000"),
                         Decimal("50"), 2, 2, True)

        btc_rebalances = db.get_rebalances(market="BTC-USD-PERP")
        eth_rebalances = db.get_rebalances(market="ETH-USD-PERP")

        assert len(btc_rebalances) == 1
        assert len(eth_rebalances) == 1

    def test_get_statistics(self, db):
        """Should return statistics."""
        now = time.time()
        db.add_trade(TradeEntry(None, "acc1", "BTC-USD-PERP", "BUY",
                                Decimal("1"), Decimal("50000"), "", "", now,
                                pnl=Decimal("100")))
        db.add_rebalance("BTC-USD-PERP", "threshold", Decimal("2000"),
                         Decimal("100"), 4, 4, True)

        stats = db.get_statistics()

        assert stats["total_trades"] == 1
        assert stats["unique_accounts"] == 1
        assert stats["total_rebalances"] == 1
        assert "db_path" in stats

    def test_clear_all(self, db):
        """Should clear all data."""
        now = time.time()
        db.add_trade(TradeEntry(None, "acc1", "BTC-USD-PERP", "BUY",
                                Decimal("1"), Decimal("50000"), "", "", now))
        db.add_rebalance("BTC-USD-PERP", "threshold", Decimal("2000"),
                         Decimal("100"), 4, 4, True)

        db.clear_all()

        assert db.get_trade_count() == 0
        assert len(db.get_rebalances()) == 0

    def test_export_to_csv(self, db, tmp_path):
        """Should export trades to CSV."""
        now = time.time()
        for i in range(3):
            db.add_trade(TradeEntry(None, "acc1", "BTC-USD-PERP", "BUY",
                                    Decimal("1"), Decimal("50000"), "", "", now))

        csv_path = tmp_path / "export.csv"
        count = db.export_to_csv(str(csv_path))

        assert count == 3
        assert csv_path.exists()

        # Check CSV content
        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) == 4  # Header + 3 trades
