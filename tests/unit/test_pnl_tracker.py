"""Unit tests for PnL Tracker."""

import pytest
import time
from decimal import Decimal
from unittest.mock import MagicMock, patch

from src.core.pnl_tracker import (
    PnLType,
    TradeDirection,
    TradeRecord,
    PnLSnapshot,
    DailyPnL,
    PnLSummary,
    PnLTracker,
    MultiAccountPnLTracker,
    PnLReporter,
    get_pnl_tracker,
    reset_pnl_tracker,
)


class TestPnLType:
    """Tests for PnLType enum."""

    def test_pnl_type_values(self):
        """Should have expected PnL types."""
        assert PnLType.REALIZED.value == "realized"
        assert PnLType.UNREALIZED.value == "unrealized"
        assert PnLType.FUNDING.value == "funding"
        assert PnLType.FEES.value == "fees"
        assert PnLType.TOTAL.value == "total"


class TestTradeDirection:
    """Tests for TradeDirection enum."""

    def test_direction_values(self):
        """Should have expected directions."""
        assert TradeDirection.LONG.value == "long"
        assert TradeDirection.SHORT.value == "short"


class TestTradeRecord:
    """Tests for TradeRecord dataclass."""

    def test_create_trade_record(self):
        """Should create trade record."""
        trade = TradeRecord(
            trade_id="trade_1",
            market="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            entry_size=Decimal("1"),
            entry_time=time.time(),
        )

        assert trade.trade_id == "trade_1"
        assert trade.market == "BTC-USD-PERP"
        assert trade.is_closed is False

    def test_notional_value(self):
        """Should calculate notional value."""
        trade = TradeRecord(
            trade_id="trade_1",
            market="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            entry_size=Decimal("2"),
            entry_time=time.time(),
        )

        assert trade.notional_value == Decimal("100000")

    def test_holding_time(self):
        """Should calculate holding time."""
        trade = TradeRecord(
            trade_id="trade_1",
            market="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            entry_size=Decimal("1"),
            entry_time=time.time() - 3600,  # 1 hour ago
        )

        assert trade.holding_hours >= 0.99  # At least 1 hour

    def test_realized_pnl_long_profit(self):
        """Should calculate realized PnL for profitable long."""
        trade = TradeRecord(
            trade_id="trade_1",
            market="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            entry_size=Decimal("1"),
            entry_time=time.time(),
            exit_price=Decimal("55000"),
            exit_time=time.time(),
            is_closed=True,
        )

        # 55000 - 50000 = 5000 profit
        assert trade.realized_pnl == Decimal("5000")

    def test_realized_pnl_long_loss(self):
        """Should calculate realized PnL for losing long."""
        trade = TradeRecord(
            trade_id="trade_1",
            market="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            entry_size=Decimal("1"),
            entry_time=time.time(),
            exit_price=Decimal("48000"),
            exit_time=time.time(),
            is_closed=True,
        )

        # 48000 - 50000 = -2000 loss
        assert trade.realized_pnl == Decimal("-2000")

    def test_realized_pnl_short_profit(self):
        """Should calculate realized PnL for profitable short."""
        trade = TradeRecord(
            trade_id="trade_1",
            market="BTC-USD-PERP",
            direction=TradeDirection.SHORT,
            entry_price=Decimal("50000"),
            entry_size=Decimal("1"),
            entry_time=time.time(),
            exit_price=Decimal("48000"),
            exit_time=time.time(),
            is_closed=True,
        )

        # 50000 - 48000 = 2000 profit
        assert trade.realized_pnl == Decimal("2000")

    def test_realized_pnl_short_loss(self):
        """Should calculate realized PnL for losing short."""
        trade = TradeRecord(
            trade_id="trade_1",
            market="BTC-USD-PERP",
            direction=TradeDirection.SHORT,
            entry_price=Decimal("50000"),
            entry_size=Decimal("1"),
            entry_time=time.time(),
            exit_price=Decimal("52000"),
            exit_time=time.time(),
            is_closed=True,
        )

        # 50000 - 52000 = -2000 loss
        assert trade.realized_pnl == Decimal("-2000")

    def test_realized_pnl_includes_fees(self):
        """Should deduct fees from realized PnL."""
        trade = TradeRecord(
            trade_id="trade_1",
            market="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            entry_size=Decimal("1"),
            entry_time=time.time(),
            exit_price=Decimal("55000"),
            exit_time=time.time(),
            fees_paid=Decimal("50"),
            is_closed=True,
        )

        # 5000 profit - 50 fees = 4950
        assert trade.realized_pnl == Decimal("4950")

    def test_unrealized_pnl_open_long(self):
        """Should calculate unrealized PnL for open long."""
        trade = TradeRecord(
            trade_id="trade_1",
            market="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            entry_size=Decimal("1"),
            entry_time=time.time(),
        )

        unrealized = trade.calculate_unrealized_pnl(Decimal("52000"))

        assert unrealized == Decimal("2000")

    def test_unrealized_pnl_open_short(self):
        """Should calculate unrealized PnL for open short."""
        trade = TradeRecord(
            trade_id="trade_1",
            market="BTC-USD-PERP",
            direction=TradeDirection.SHORT,
            entry_price=Decimal("50000"),
            entry_size=Decimal("1"),
            entry_time=time.time(),
        )

        unrealized = trade.calculate_unrealized_pnl(Decimal("48000"))

        assert unrealized == Decimal("2000")

    def test_unrealized_pnl_closed_trade(self):
        """Should return zero for closed trade."""
        trade = TradeRecord(
            trade_id="trade_1",
            market="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            entry_size=Decimal("1"),
            entry_time=time.time(),
            is_closed=True,
        )

        unrealized = trade.calculate_unrealized_pnl(Decimal("55000"))

        assert unrealized == Decimal("0")

    def test_to_dict(self):
        """Should convert to dictionary."""
        trade = TradeRecord(
            trade_id="trade_1",
            market="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            entry_size=Decimal("1"),
            entry_time=time.time(),
        )

        d = trade.to_dict()

        assert d["trade_id"] == "trade_1"
        assert d["market"] == "BTC-USD-PERP"
        assert d["direction"] == "long"


class TestPnLSnapshot:
    """Tests for PnLSnapshot dataclass."""

    def test_create_snapshot(self):
        """Should create snapshot."""
        snapshot = PnLSnapshot(
            timestamp=time.time(),
            realized_pnl=Decimal("1000"),
            unrealized_pnl=Decimal("500"),
            fees=Decimal("50"),
            funding=Decimal("25"),
            equity=Decimal("11500"),
            open_positions=2,
        )

        assert snapshot.realized_pnl == Decimal("1000")
        assert snapshot.open_positions == 2

    def test_total_pnl(self):
        """Should calculate total PnL."""
        snapshot = PnLSnapshot(
            timestamp=time.time(),
            realized_pnl=Decimal("1000"),
            unrealized_pnl=Decimal("500"),
            fees=Decimal("50"),
            funding=Decimal("25"),
            equity=Decimal("11500"),
            open_positions=2,
        )

        assert snapshot.total_pnl == Decimal("1500")

    def test_net_pnl(self):
        """Should calculate net PnL after fees and funding."""
        snapshot = PnLSnapshot(
            timestamp=time.time(),
            realized_pnl=Decimal("1000"),
            unrealized_pnl=Decimal("500"),
            fees=Decimal("50"),
            funding=Decimal("25"),
            equity=Decimal("11500"),
            open_positions=2,
        )

        # 1500 - 50 - 25 = 1425
        assert snapshot.net_pnl == Decimal("1425")

    def test_to_dict(self):
        """Should convert to dictionary."""
        snapshot = PnLSnapshot(
            timestamp=time.time(),
            realized_pnl=Decimal("1000"),
            unrealized_pnl=Decimal("500"),
            fees=Decimal("50"),
            funding=Decimal("25"),
            equity=Decimal("11500"),
            open_positions=2,
        )

        d = snapshot.to_dict()

        assert "total_pnl" in d
        assert "net_pnl" in d


class TestDailyPnL:
    """Tests for DailyPnL dataclass."""

    def test_create_daily_pnl(self):
        """Should create daily PnL."""
        daily = DailyPnL(
            date="2025-01-01",
            starting_equity=Decimal("10000"),
            ending_equity=Decimal("10500"),
            realized_pnl=Decimal("500"),
            unrealized_pnl=Decimal("0"),
            fees=Decimal("25"),
            funding=Decimal("10"),
            trades_count=5,
            winners=3,
            losers=2,
        )

        assert daily.date == "2025-01-01"
        assert daily.trades_count == 5

    def test_return_pct(self):
        """Should calculate return percentage."""
        daily = DailyPnL(
            date="2025-01-01",
            starting_equity=Decimal("10000"),
            ending_equity=Decimal("10500"),
            realized_pnl=Decimal("500"),
            unrealized_pnl=Decimal("0"),
            fees=Decimal("0"),
            funding=Decimal("0"),
            trades_count=5,
            winners=3,
            losers=2,
        )

        assert daily.return_pct == 5.0

    def test_win_rate(self):
        """Should calculate win rate."""
        daily = DailyPnL(
            date="2025-01-01",
            starting_equity=Decimal("10000"),
            ending_equity=Decimal("10500"),
            realized_pnl=Decimal("500"),
            unrealized_pnl=Decimal("0"),
            fees=Decimal("0"),
            funding=Decimal("0"),
            trades_count=5,
            winners=3,
            losers=2,
        )

        assert daily.win_rate == 60.0

    def test_to_dict(self):
        """Should convert to dictionary."""
        daily = DailyPnL(
            date="2025-01-01",
            starting_equity=Decimal("10000"),
            ending_equity=Decimal("10500"),
            realized_pnl=Decimal("500"),
            unrealized_pnl=Decimal("0"),
            fees=Decimal("0"),
            funding=Decimal("0"),
            trades_count=5,
            winners=3,
            losers=2,
        )

        d = daily.to_dict()

        assert d["date"] == "2025-01-01"
        assert "return_pct" in d


class TestPnLSummary:
    """Tests for PnLSummary dataclass."""

    def test_create_summary(self):
        """Should create summary."""
        summary = PnLSummary(
            total_realized=Decimal("1000"),
            total_unrealized=Decimal("500"),
            total_fees=Decimal("50"),
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
        )

        assert summary.total_trades == 10

    def test_total_pnl(self):
        """Should calculate total PnL."""
        summary = PnLSummary(
            total_realized=Decimal("1000"),
            total_unrealized=Decimal("500"),
        )

        assert summary.total_pnl == Decimal("1500")

    def test_win_rate(self):
        """Should calculate win rate."""
        summary = PnLSummary(
            winning_trades=7,
            losing_trades=3,
        )

        assert summary.win_rate == 70.0

    def test_expectancy(self):
        """Should calculate expectancy."""
        summary = PnLSummary(
            total_realized=Decimal("1000"),
            total_unrealized=Decimal("0"),
            total_fees=Decimal("100"),
            total_funding=Decimal("50"),
            total_trades=10,
        )

        # (1000 - 100 - 50) / 10 = 85
        assert summary.expectancy == Decimal("85")

    def test_to_dict(self):
        """Should convert to dictionary."""
        summary = PnLSummary(
            total_realized=Decimal("1000"),
            total_trades=10,
        )

        d = summary.to_dict()

        assert "total_pnl" in d
        assert "win_rate" in d


class TestPnLTracker:
    """Tests for PnLTracker."""

    @pytest.fixture
    def tracker(self):
        """Create PnL tracker."""
        return PnLTracker(initial_equity=Decimal("10000"))

    def test_create_tracker(self, tracker):
        """Should create tracker with initial equity."""
        assert tracker.initial_equity == Decimal("10000")

    def test_open_trade(self, tracker):
        """Should open trade."""
        trade = tracker.open_trade(
            trade_id="trade_1",
            market="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            entry_size=Decimal("1"),
            fees=Decimal("25"),
        )

        assert trade.trade_id == "trade_1"
        assert len(tracker.get_open_trades()) == 1

    def test_close_trade(self, tracker):
        """Should close trade."""
        tracker.open_trade(
            trade_id="trade_1",
            market="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            entry_size=Decimal("1"),
        )

        trade = tracker.close_trade(
            trade_id="trade_1",
            exit_price=Decimal("55000"),
            fees=Decimal("25"),
        )

        assert trade.is_closed is True
        assert len(tracker.get_open_trades()) == 0
        assert len(tracker.get_closed_trades()) == 1

    def test_close_nonexistent_trade(self, tracker):
        """Should return None for nonexistent trade."""
        result = tracker.close_trade(
            trade_id="nonexistent",
            exit_price=Decimal("50000"),
        )

        assert result is None

    def test_add_funding(self, tracker):
        """Should add funding to trade."""
        tracker.open_trade(
            trade_id="trade_1",
            market="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            entry_size=Decimal("1"),
        )

        tracker.add_funding("trade_1", Decimal("5"))

        trade = tracker.get_open_trades()[0]
        assert trade.funding_paid == Decimal("5")

    def test_get_unrealized_pnl(self, tracker):
        """Should calculate unrealized PnL."""
        tracker.open_trade(
            trade_id="trade_1",
            market="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            entry_size=Decimal("1"),
        )

        prices = {"BTC-USD-PERP": Decimal("52000")}
        unrealized = tracker.get_unrealized_pnl(prices)

        assert unrealized == Decimal("2000")

    def test_get_realized_pnl(self, tracker):
        """Should calculate realized PnL."""
        tracker.open_trade(
            trade_id="trade_1",
            market="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            entry_size=Decimal("1"),
        )

        tracker.close_trade(
            trade_id="trade_1",
            exit_price=Decimal("55000"),
        )

        realized = tracker.get_realized_pnl()

        assert realized == Decimal("5000")

    def test_get_total_pnl(self, tracker):
        """Should calculate total PnL."""
        # Open trade
        tracker.open_trade(
            trade_id="trade_1",
            market="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            entry_size=Decimal("1"),
        )

        # Close trade with profit
        tracker.close_trade(
            trade_id="trade_1",
            exit_price=Decimal("55000"),
        )

        # Open another trade
        tracker.open_trade(
            trade_id="trade_2",
            market="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("55000"),
            entry_size=Decimal("1"),
        )

        prices = {"BTC-USD-PERP": Decimal("56000")}
        total = tracker.get_total_pnl(prices)

        # 5000 realized + 1000 unrealized
        assert total == Decimal("6000")

    def test_get_current_equity(self, tracker):
        """Should calculate current equity."""
        tracker.open_trade(
            trade_id="trade_1",
            market="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            entry_size=Decimal("1"),
        )

        tracker.close_trade(
            trade_id="trade_1",
            exit_price=Decimal("55000"),
        )

        prices = {}
        equity = tracker.get_current_equity(prices)

        # 10000 + 5000 = 15000
        assert equity == Decimal("15000")

    def test_get_drawdown(self, tracker):
        """Should calculate drawdown."""
        tracker.update_equity(Decimal("12000"))  # Peak
        tracker.update_equity(Decimal("11000"))  # Drop

        prices = {}
        tracker._peak_equity = Decimal("12000")

        # Mock get_current_equity
        with patch.object(tracker, 'get_current_equity', return_value=Decimal("10000")):
            dd, dd_pct = tracker.get_drawdown(prices)

            assert dd == Decimal("2000")
            assert dd_pct == pytest.approx(16.67, rel=0.01)

    def test_take_snapshot(self, tracker):
        """Should take PnL snapshot."""
        tracker.open_trade(
            trade_id="trade_1",
            market="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            entry_size=Decimal("1"),
            fees=Decimal("25"),
        )

        prices = {"BTC-USD-PERP": Decimal("51000")}
        snapshot = tracker.take_snapshot(prices)

        assert snapshot.unrealized_pnl == Decimal("975")  # 1000 - 25 fees
        assert snapshot.open_positions == 1

    def test_get_summary(self, tracker):
        """Should get PnL summary."""
        # Create winning trade
        tracker.open_trade(
            trade_id="trade_1",
            market="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            entry_size=Decimal("1"),
        )
        tracker.close_trade("trade_1", Decimal("55000"))

        # Create losing trade
        tracker.open_trade(
            trade_id="trade_2",
            market="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("55000"),
            entry_size=Decimal("1"),
        )
        tracker.close_trade("trade_2", Decimal("53000"))

        prices = {}
        summary = tracker.get_summary(prices)

        assert summary.total_trades == 2
        assert summary.winning_trades == 1
        assert summary.losing_trades == 1
        assert summary.win_rate == 50.0

    def test_get_trades_by_market(self, tracker):
        """Should filter trades by market."""
        tracker.open_trade("t1", "BTC-USD-PERP", TradeDirection.LONG, Decimal("50000"), Decimal("1"))
        tracker.open_trade("t2", "ETH-USD-PERP", TradeDirection.LONG, Decimal("3000"), Decimal("1"))
        tracker.close_trade("t1", Decimal("51000"))

        btc_trades = tracker.get_trades_by_market("BTC-USD-PERP")

        assert len(btc_trades) == 1
        assert btc_trades[0].market == "BTC-USD-PERP"

    def test_reset(self, tracker):
        """Should reset tracker."""
        tracker.open_trade("t1", "BTC-USD-PERP", TradeDirection.LONG, Decimal("50000"), Decimal("1"))
        tracker.close_trade("t1", Decimal("51000"))

        tracker.reset()

        assert len(tracker.get_open_trades()) == 0
        assert len(tracker.get_closed_trades()) == 0


class TestMultiAccountPnLTracker:
    """Tests for MultiAccountPnLTracker."""

    @pytest.fixture
    def tracker(self):
        """Create multi-account tracker."""
        return MultiAccountPnLTracker()

    def test_add_account(self, tracker):
        """Should add account tracker."""
        account_tracker = tracker.add_account("acc1", Decimal("10000"))

        assert account_tracker is not None
        assert tracker.get_tracker("acc1") is not None

    def test_get_combined_pnl(self, tracker):
        """Should combine PnL across accounts."""
        # Add accounts
        t1 = tracker.add_account("acc1", Decimal("10000"))
        t2 = tracker.add_account("acc2", Decimal("10000"))

        # Open trades
        t1.open_trade("t1", "BTC-USD-PERP", TradeDirection.LONG, Decimal("50000"), Decimal("1"))
        t2.open_trade("t2", "BTC-USD-PERP", TradeDirection.SHORT, Decimal("50000"), Decimal("1"))

        prices = {"BTC-USD-PERP": Decimal("52000")}
        combined = tracker.get_combined_pnl(prices)

        # Long: +2000, Short: -2000, Total: 0
        assert combined == Decimal("0")

    def test_get_combined_summary(self, tracker):
        """Should get combined summary."""
        t1 = tracker.add_account("acc1", Decimal("10000"))
        t2 = tracker.add_account("acc2", Decimal("10000"))

        t1.open_trade("t1", "BTC-USD-PERP", TradeDirection.LONG, Decimal("50000"), Decimal("1"))
        t1.close_trade("t1", Decimal("55000"))

        t2.open_trade("t2", "BTC-USD-PERP", TradeDirection.SHORT, Decimal("55000"), Decimal("1"))
        t2.close_trade("t2", Decimal("50000"))

        prices = {}
        summary = tracker.get_combined_summary(prices)

        assert summary["accounts"] == 2
        assert summary["total_trades"] == 2

    def test_get_delta_exposure(self, tracker):
        """Should calculate delta exposure."""
        t1 = tracker.add_account("acc1", Decimal("10000"))
        t2 = tracker.add_account("acc2", Decimal("10000"))

        # Long $50000 notional
        t1.open_trade("t1", "BTC-USD-PERP", TradeDirection.LONG, Decimal("50000"), Decimal("1"))

        # Short $50000 notional
        t2.open_trade("t2", "BTC-USD-PERP", TradeDirection.SHORT, Decimal("50000"), Decimal("1"))

        prices = {"BTC-USD-PERP": Decimal("50000")}
        exposure = tracker.get_delta_exposure(prices)

        # Long 50000 - Short 50000 = 0
        assert exposure.get("BTC-USD-PERP", Decimal("0")) == Decimal("0")


class TestPnLReporter:
    """Tests for PnLReporter."""

    @pytest.fixture
    def tracker(self):
        """Create PnL tracker."""
        return PnLTracker(initial_equity=Decimal("10000"))

    @pytest.fixture
    def reporter(self, tracker):
        """Create reporter."""
        return PnLReporter(tracker=tracker)

    def test_format_summary(self, reporter, tracker):
        """Should format summary as text."""
        tracker.open_trade("t1", "BTC-USD-PERP", TradeDirection.LONG, Decimal("50000"), Decimal("1"))
        tracker.close_trade("t1", Decimal("55000"))

        prices = {}
        text = reporter.format_summary(prices)

        assert "PnL Summary" in text
        assert "Total PnL" in text


class TestGlobalTracker:
    """Tests for global tracker functions."""

    def test_get_pnl_tracker(self):
        """Should get or create tracker."""
        reset_pnl_tracker()

        t1 = get_pnl_tracker()
        t2 = get_pnl_tracker()

        assert t1 is t2

    def test_reset_pnl_tracker(self):
        """Should reset tracker."""
        t1 = get_pnl_tracker()
        reset_pnl_tracker()
        t2 = get_pnl_tracker()

        assert t1 is not t2


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_size_trade(self):
        """Should handle zero size trade."""
        trade = TradeRecord(
            trade_id="t1",
            market="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            entry_size=Decimal("0"),
            entry_time=time.time(),
        )

        assert trade.notional_value == Decimal("0")

    def test_zero_price_trade(self):
        """Should handle zero price trade."""
        trade = TradeRecord(
            trade_id="t1",
            market="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("0"),
            entry_size=Decimal("1"),
            entry_time=time.time(),
        )

        assert trade.notional_value == Decimal("0")

    def test_empty_prices(self):
        """Should handle empty prices."""
        tracker = PnLTracker()
        tracker.open_trade("t1", "BTC-USD-PERP", TradeDirection.LONG, Decimal("50000"), Decimal("1"))

        prices = {}
        unrealized = tracker.get_unrealized_pnl(prices)

        assert unrealized == Decimal("0")

    def test_no_trades(self):
        """Should handle no trades."""
        tracker = PnLTracker()

        prices = {}
        summary = tracker.get_summary(prices)

        assert summary.total_trades == 0
        assert summary.win_rate == 0.0

    def test_all_winning_trades(self):
        """Should calculate 100% win rate."""
        tracker = PnLTracker()

        for i in range(5):
            tracker.open_trade(f"t{i}", "BTC-USD-PERP", TradeDirection.LONG, Decimal("50000"), Decimal("1"))
            tracker.close_trade(f"t{i}", Decimal("51000"))

        prices = {}
        summary = tracker.get_summary(prices)

        assert summary.win_rate == 100.0

    def test_all_losing_trades(self):
        """Should calculate 0% win rate."""
        tracker = PnLTracker()

        for i in range(5):
            tracker.open_trade(f"t{i}", "BTC-USD-PERP", TradeDirection.LONG, Decimal("50000"), Decimal("1"))
            tracker.close_trade(f"t{i}", Decimal("49000"))

        prices = {}
        summary = tracker.get_summary(prices)

        assert summary.win_rate == 0.0
