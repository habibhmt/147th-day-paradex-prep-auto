"""Tests for trade journal system."""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta

from src.analytics.trade_journal import (
    TradeDirection,
    TradeStatus,
    TradeOutcome,
    EmotionalState,
    SetupQuality,
    ExecutionQuality,
    TradeNote,
    TradeTag,
    TradeScreenshot,
    TradeEntry,
    DailyStats,
    WeeklyReview,
    TradeAnalyzer,
    TagManager,
    TradeJournal,
    get_trade_journal,
    set_trade_journal,
)


class TestEnums:
    """Test enum classes."""

    def test_trade_direction_values(self):
        """Test TradeDirection enum values."""
        assert TradeDirection.LONG.value == "long"
        assert TradeDirection.SHORT.value == "short"

    def test_trade_status_values(self):
        """Test TradeStatus enum values."""
        assert TradeStatus.OPEN.value == "open"
        assert TradeStatus.CLOSED.value == "closed"
        assert TradeStatus.PARTIAL.value == "partial"
        assert TradeStatus.CANCELLED.value == "cancelled"

    def test_trade_outcome_values(self):
        """Test TradeOutcome enum values."""
        assert TradeOutcome.WIN.value == "win"
        assert TradeOutcome.LOSS.value == "loss"
        assert TradeOutcome.BREAKEVEN.value == "breakeven"
        assert TradeOutcome.PENDING.value == "pending"

    def test_emotional_state_values(self):
        """Test EmotionalState enum values."""
        assert EmotionalState.CONFIDENT.value == "confident"
        assert EmotionalState.FEARFUL.value == "fearful"
        assert EmotionalState.GREEDY.value == "greedy"
        assert EmotionalState.NEUTRAL.value == "neutral"

    def test_setup_quality_values(self):
        """Test SetupQuality enum values."""
        assert SetupQuality.A_PLUS.value == "A+"
        assert SetupQuality.A.value == "A"
        assert SetupQuality.B.value == "B"

    def test_execution_quality_values(self):
        """Test ExecutionQuality enum values."""
        assert ExecutionQuality.PERFECT.value == "perfect"
        assert ExecutionQuality.GOOD.value == "good"
        assert ExecutionQuality.POOR.value == "poor"


class TestTradeNote:
    """Test TradeNote class."""

    def test_creation(self):
        """Test TradeNote creation."""
        note = TradeNote(
            id="note1",
            content="Entry at key support level",
            note_type="entry",
        )
        assert note.id == "note1"
        assert note.content == "Entry at key support level"
        assert note.note_type == "entry"

    def test_to_dict(self):
        """Test TradeNote to_dict."""
        note = TradeNote(id="note1", content="Test", note_type="observation")
        result = note.to_dict()
        assert result["id"] == "note1"
        assert result["note_type"] == "observation"


class TestTradeTag:
    """Test TradeTag class."""

    def test_creation(self):
        """Test TradeTag creation."""
        tag = TradeTag(name="breakout", color="#2196F3", description="Breakout trades")
        assert tag.name == "breakout"
        assert tag.color == "#2196F3"

    def test_to_dict(self):
        """Test TradeTag to_dict."""
        tag = TradeTag(name="scalp", color="#9C27B0")
        result = tag.to_dict()
        assert result["name"] == "scalp"
        assert result["color"] == "#9C27B0"


class TestTradeScreenshot:
    """Test TradeScreenshot class."""

    def test_creation(self):
        """Test TradeScreenshot creation."""
        ss = TradeScreenshot(
            id="ss1",
            path="/screenshots/trade1.png",
            caption="Entry point",
            timeframe="1h",
        )
        assert ss.id == "ss1"
        assert ss.path == "/screenshots/trade1.png"
        assert ss.timeframe == "1h"

    def test_to_dict(self):
        """Test TradeScreenshot to_dict."""
        ss = TradeScreenshot(id="ss1", path="/path/img.png")
        result = ss.to_dict()
        assert result["id"] == "ss1"
        assert result["path"] == "/path/img.png"


class TestTradeEntry:
    """Test TradeEntry class."""

    def test_creation(self):
        """Test TradeEntry creation."""
        trade = TradeEntry(
            id="trade1",
            symbol="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            status=TradeStatus.OPEN,
            entry_price=Decimal("50000"),
            entry_time=datetime.now(),
            size=Decimal("0.1"),
        )
        assert trade.id == "trade1"
        assert trade.symbol == "BTC-USD-PERP"
        assert trade.direction == TradeDirection.LONG

    def test_to_dict(self):
        """Test TradeEntry to_dict."""
        trade = TradeEntry(
            id="trade1",
            symbol="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            status=TradeStatus.CLOSED,
            entry_price=Decimal("50000"),
            entry_time=datetime.now(),
            size=Decimal("0.1"),
            exit_price=Decimal("51000"),
            exit_time=datetime.now(),
            pnl=Decimal("100"),
            outcome=TradeOutcome.WIN,
        )
        result = trade.to_dict()
        assert result["id"] == "trade1"
        assert result["direction"] == "long"
        assert result["outcome"] == "win"

    def test_duration(self):
        """Test trade duration calculation."""
        entry = datetime.now() - timedelta(hours=2)
        exit = datetime.now()
        trade = TradeEntry(
            id="trade1",
            symbol="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            status=TradeStatus.CLOSED,
            entry_price=Decimal("50000"),
            entry_time=entry,
            size=Decimal("0.1"),
            exit_time=exit,
        )
        assert trade.duration is not None
        assert trade.duration.total_seconds() > 7000  # ~2 hours

    def test_duration_open_trade(self):
        """Test duration for open trade."""
        trade = TradeEntry(
            id="trade1",
            symbol="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            status=TradeStatus.OPEN,
            entry_price=Decimal("50000"),
            entry_time=datetime.now(),
            size=Decimal("0.1"),
        )
        assert trade.duration is None

    def test_risk_reward_actual(self):
        """Test actual risk/reward calculation."""
        trade = TradeEntry(
            id="trade1",
            symbol="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            status=TradeStatus.CLOSED,
            entry_price=Decimal("50000"),
            entry_time=datetime.now(),
            size=Decimal("0.1"),
            stop_loss=Decimal("49000"),
            exit_price=Decimal("52000"),
        )
        rr = trade.risk_reward_actual
        assert rr is not None
        assert rr == Decimal("2")


class TestDailyStats:
    """Test DailyStats class."""

    def test_creation(self):
        """Test DailyStats creation."""
        stats = DailyStats(
            date=datetime.now(),
            total_trades=10,
            winning_trades=6,
            losing_trades=3,
            breakeven_trades=1,
            total_pnl=Decimal("500"),
            win_rate=Decimal("60"),
        )
        assert stats.total_trades == 10
        assert stats.win_rate == Decimal("60")

    def test_to_dict(self):
        """Test DailyStats to_dict."""
        stats = DailyStats(date=datetime.now(), total_trades=5)
        result = stats.to_dict()
        assert result["total_trades"] == 5


class TestWeeklyReview:
    """Test WeeklyReview class."""

    def test_creation(self):
        """Test WeeklyReview creation."""
        review = WeeklyReview(
            week_start=datetime.now(),
            week_end=datetime.now() + timedelta(days=6),
            total_trades=20,
            total_pnl=Decimal("1000"),
            lessons_learned=["Be patient", "Wait for confirmation"],
        )
        assert review.total_trades == 20
        assert len(review.lessons_learned) == 2

    def test_to_dict(self):
        """Test WeeklyReview to_dict."""
        review = WeeklyReview(
            week_start=datetime.now(),
            week_end=datetime.now() + timedelta(days=6),
        )
        result = review.to_dict()
        assert "week_start" in result
        assert "lessons_learned" in result


class TestTradeAnalyzer:
    """Test TradeAnalyzer class."""

    def _create_trades(self) -> list[TradeEntry]:
        """Create sample trades for testing."""
        trades = []
        base_time = datetime.now() - timedelta(days=7)

        # 3 winning trades
        for i in range(3):
            trade = TradeEntry(
                id=f"win{i}",
                symbol="BTC-USD-PERP",
                direction=TradeDirection.LONG,
                status=TradeStatus.CLOSED,
                entry_price=Decimal("50000"),
                entry_time=base_time + timedelta(hours=i * 2),
                exit_time=base_time + timedelta(hours=i * 2 + 1),
                size=Decimal("0.1"),
                exit_price=Decimal("51000"),
                pnl=Decimal("100"),
                fees=Decimal("5"),
                outcome=TradeOutcome.WIN,
            )
            trades.append(trade)

        # 2 losing trades
        for i in range(2):
            trade = TradeEntry(
                id=f"loss{i}",
                symbol="ETH-USD-PERP",
                direction=TradeDirection.SHORT,
                status=TradeStatus.CLOSED,
                entry_price=Decimal("3000"),
                entry_time=base_time + timedelta(hours=10 + i * 2),
                exit_time=base_time + timedelta(hours=11 + i * 2),
                size=Decimal("1"),
                exit_price=Decimal("3050"),
                pnl=Decimal("-50"),
                fees=Decimal("3"),
                outcome=TradeOutcome.LOSS,
            )
            trades.append(trade)

        return trades

    def test_calculate_stats(self):
        """Test calculating statistics."""
        analyzer = TradeAnalyzer()
        trades = self._create_trades()
        stats = analyzer.calculate_stats(trades)

        assert stats["total_trades"] == 5
        assert stats["winning_trades"] == 3
        assert stats["losing_trades"] == 2
        assert stats["win_rate"] == 60.0

    def test_empty_stats(self):
        """Test stats with no trades."""
        analyzer = TradeAnalyzer()
        stats = analyzer.calculate_stats([])
        assert stats["total_trades"] == 0
        assert stats["win_rate"] == 0

    def test_stats_by_symbol(self):
        """Test stats grouped by symbol."""
        analyzer = TradeAnalyzer()
        trades = self._create_trades()
        by_symbol = analyzer.stats_by_symbol(trades)

        assert "BTC-USD-PERP" in by_symbol
        assert "ETH-USD-PERP" in by_symbol
        assert by_symbol["BTC-USD-PERP"]["total_trades"] == 3
        assert by_symbol["ETH-USD-PERP"]["total_trades"] == 2

    def test_stats_by_strategy(self):
        """Test stats grouped by strategy."""
        analyzer = TradeAnalyzer()
        trades = self._create_trades()
        trades[0].strategy = "breakout"
        trades[1].strategy = "breakout"
        trades[2].strategy = "trend"

        by_strategy = analyzer.stats_by_strategy(trades)
        assert "breakout" in by_strategy
        assert "trend" in by_strategy

    def test_equity_curve(self):
        """Test equity curve calculation."""
        analyzer = TradeAnalyzer()
        trades = self._create_trades()
        curve = analyzer.equity_curve(trades)

        assert len(curve) > 1
        assert curve[0]["equity"] == 10000

    def test_drawdown_analysis(self):
        """Test drawdown analysis."""
        analyzer = TradeAnalyzer()
        trades = self._create_trades()
        dd = analyzer.drawdown_analysis(trades)

        assert "max_drawdown" in dd
        assert "max_drawdown_pct" in dd


class TestTagManager:
    """Test TagManager class."""

    def test_default_tags(self):
        """Test default tags are created."""
        manager = TagManager()
        tags = manager.get_all()
        assert len(tags) > 0
        assert any(t.name == "trend-following" for t in tags)

    def test_add_tag(self):
        """Test adding a tag."""
        manager = TagManager()
        tag = manager.add("my-tag", "#FF0000", "My custom tag")
        assert tag.name == "my-tag"
        assert tag.color == "#FF0000"

    def test_remove_tag(self):
        """Test removing a tag."""
        manager = TagManager()
        manager.add("temp-tag")
        assert manager.remove("temp-tag") is True
        assert manager.remove("nonexistent") is False

    def test_get_tag(self):
        """Test getting a tag."""
        manager = TagManager()
        manager.add("test-tag")
        tag = manager.get("test-tag")
        assert tag is not None
        assert tag.name == "test-tag"

    def test_search_tags(self):
        """Test searching tags."""
        manager = TagManager()
        results = manager.search("trend")
        assert len(results) > 0


class TestTradeJournal:
    """Test TradeJournal class."""

    def test_init(self):
        """Test initialization."""
        journal = TradeJournal()
        assert journal is not None
        assert journal.tag_manager is not None

    def test_add_trade(self):
        """Test adding a trade."""
        journal = TradeJournal()
        trade = journal.add_trade(
            symbol="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
            stop_loss=Decimal("49000"),
            take_profit=Decimal("52000"),
            strategy="breakout",
            setup_quality=SetupQuality.A,
        )
        assert trade.id is not None
        assert trade.symbol == "BTC-USD-PERP"
        assert trade.status == TradeStatus.OPEN

    def test_close_trade(self):
        """Test closing a trade."""
        journal = TradeJournal()
        trade = journal.add_trade(
            symbol="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
        )

        closed = journal.close_trade(
            trade_id=trade.id,
            exit_price=Decimal("51000"),
            fees=Decimal("5"),
            exit_reason="Take profit hit",
        )

        assert closed is not None
        assert closed.status == TradeStatus.CLOSED
        assert closed.pnl == Decimal("95")  # 100 - 5 fees
        assert closed.outcome == TradeOutcome.WIN

    def test_close_short_trade(self):
        """Test closing a short trade."""
        journal = TradeJournal()
        trade = journal.add_trade(
            symbol="BTC-USD-PERP",
            direction=TradeDirection.SHORT,
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
        )

        closed = journal.close_trade(
            trade_id=trade.id,
            exit_price=Decimal("49000"),
            fees=Decimal("5"),
        )

        assert closed is not None
        assert closed.pnl == Decimal("95")
        assert closed.outcome == TradeOutcome.WIN

    def test_close_losing_trade(self):
        """Test closing a losing trade."""
        journal = TradeJournal()
        trade = journal.add_trade(
            symbol="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
        )

        closed = journal.close_trade(
            trade_id=trade.id,
            exit_price=Decimal("49000"),
        )

        assert closed.outcome == TradeOutcome.LOSS
        assert closed.pnl < 0

    def test_update_trade(self):
        """Test updating a trade."""
        journal = TradeJournal()
        trade = journal.add_trade(
            symbol="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
        )

        updated = journal.update_trade(
            trade_id=trade.id,
            stop_loss=Decimal("48000"),
            take_profit=Decimal("55000"),
            strategy="swing",
        )

        assert updated is not None
        assert updated.stop_loss == Decimal("48000")
        assert updated.strategy == "swing"

    def test_add_note(self):
        """Test adding a note to trade."""
        journal = TradeJournal()
        trade = journal.add_trade(
            symbol="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
        )

        note = journal.add_note(
            trade_id=trade.id,
            content="Strong support at this level",
            note_type="entry",
        )

        assert note is not None
        assert len(trade.notes) == 1

    def test_add_screenshot(self):
        """Test adding a screenshot to trade."""
        journal = TradeJournal()
        trade = journal.add_trade(
            symbol="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
        )

        ss = journal.add_screenshot(
            trade_id=trade.id,
            path="/screenshots/entry.png",
            caption="Entry setup",
            timeframe="1h",
        )

        assert ss is not None
        assert len(trade.screenshots) == 1

    def test_add_remove_tag(self):
        """Test adding and removing tags."""
        journal = TradeJournal()
        trade = journal.add_trade(
            symbol="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
        )

        assert journal.add_tag(trade.id, "breakout") is True
        assert "breakout" in trade.tags

        assert journal.remove_tag(trade.id, "breakout") is True
        assert "breakout" not in trade.tags

    def test_get_trades_filters(self):
        """Test getting trades with filters."""
        journal = TradeJournal()

        # Add trades with different attributes
        journal.add_trade(
            symbol="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
            strategy="breakout",
        )
        journal.add_trade(
            symbol="ETH-USD-PERP",
            direction=TradeDirection.SHORT,
            entry_price=Decimal("3000"),
            size=Decimal("1"),
            strategy="trend",
        )

        btc_trades = journal.get_trades(symbol="BTC-USD-PERP")
        assert len(btc_trades) == 1

        long_trades = journal.get_trades(direction=TradeDirection.LONG)
        assert len(long_trades) == 1

        breakout_trades = journal.get_trades(strategy="breakout")
        assert len(breakout_trades) == 1

    def test_get_open_closed_trades(self):
        """Test getting open and closed trades."""
        journal = TradeJournal()

        trade1 = journal.add_trade(
            symbol="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
        )
        trade2 = journal.add_trade(
            symbol="ETH-USD-PERP",
            direction=TradeDirection.SHORT,
            entry_price=Decimal("3000"),
            size=Decimal("1"),
        )
        journal.close_trade(trade1.id, Decimal("51000"))

        assert len(journal.get_open_trades()) == 1
        assert len(journal.get_closed_trades()) == 1

    def test_get_winning_losing_trades(self):
        """Test getting winning and losing trades."""
        journal = TradeJournal()

        trade1 = journal.add_trade(
            symbol="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
        )
        trade2 = journal.add_trade(
            symbol="ETH-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("3000"),
            size=Decimal("1"),
        )

        journal.close_trade(trade1.id, Decimal("51000"))  # Win
        journal.close_trade(trade2.id, Decimal("2900"))   # Loss

        assert len(journal.get_winning_trades()) == 1
        assert len(journal.get_losing_trades()) == 1

    def test_delete_trade(self):
        """Test deleting a trade."""
        journal = TradeJournal()
        trade = journal.add_trade(
            symbol="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
        )

        assert journal.delete_trade(trade.id) is True
        assert journal.get_trade(trade.id) is None
        assert journal.delete_trade("nonexistent") is False

    def test_get_stats(self):
        """Test getting overall stats."""
        journal = TradeJournal()

        trade = journal.add_trade(
            symbol="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
        )
        journal.close_trade(trade.id, Decimal("51000"))

        stats = journal.get_stats()
        assert stats["total_trades"] == 1
        assert stats["winning_trades"] == 1

    def test_get_stats_by_symbol(self):
        """Test getting stats by symbol."""
        journal = TradeJournal()

        t1 = journal.add_trade(
            symbol="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
        )
        t2 = journal.add_trade(
            symbol="ETH-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("3000"),
            size=Decimal("1"),
        )
        journal.close_trade(t1.id, Decimal("51000"))
        journal.close_trade(t2.id, Decimal("3100"))

        by_symbol = journal.get_stats_by_symbol()
        assert "BTC-USD-PERP" in by_symbol
        assert "ETH-USD-PERP" in by_symbol

    def test_get_equity_curve(self):
        """Test getting equity curve."""
        journal = TradeJournal()

        t = journal.add_trade(
            symbol="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
        )
        journal.close_trade(t.id, Decimal("51000"))

        curve = journal.get_equity_curve()
        assert len(curve) > 0

    def test_get_drawdown_analysis(self):
        """Test getting drawdown analysis."""
        journal = TradeJournal()

        t = journal.add_trade(
            symbol="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
        )
        journal.close_trade(t.id, Decimal("51000"))

        dd = journal.get_drawdown_analysis()
        assert "max_drawdown" in dd

    def test_daily_stats(self):
        """Test daily stats."""
        journal = TradeJournal()

        t = journal.add_trade(
            symbol="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
            entry_time=datetime.now(),
        )
        journal.close_trade(t.id, Decimal("51000"))

        daily = journal.get_daily_stats(datetime.now())
        assert daily.total_trades >= 0

    def test_weekly_review(self):
        """Test weekly review."""
        journal = TradeJournal()
        week_start = datetime.now() - timedelta(days=datetime.now().weekday())

        review = journal.add_weekly_review(
            week_start=week_start,
            lessons_learned=["Be patient"],
            goals_next_week=["Follow the plan"],
            overall_rating=7,
        )

        assert review is not None
        assert review.overall_rating == 7

        retrieved = journal.get_weekly_review(week_start)
        assert retrieved is not None

    def test_add_callback(self):
        """Test adding callback."""
        journal = TradeJournal()
        events = []

        def callback(event, data):
            events.append(event)

        journal.add_callback(callback)
        journal.add_trade(
            symbol="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
        )

        assert "trade_added" in events

    def test_remove_callback(self):
        """Test removing callback."""
        journal = TradeJournal()

        def callback(event, data):
            pass

        journal.add_callback(callback)
        assert journal.remove_callback(callback) is True
        assert journal.remove_callback(callback) is False

    def test_export_json(self):
        """Test exporting to JSON."""
        journal = TradeJournal()
        journal.add_trade(
            symbol="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
        )

        json_str = journal.export_json()
        assert "trades" in json_str
        assert "BTC-USD-PERP" in json_str

    def test_get_summary(self):
        """Test getting summary."""
        journal = TradeJournal()
        t = journal.add_trade(
            symbol="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
        )
        journal.close_trade(t.id, Decimal("51000"))

        summary = journal.get_summary()
        assert summary["total_trades"] == 1
        assert summary["closed_trades"] == 1


class TestGlobalInstance:
    """Test global instance functions."""

    def test_get_trade_journal(self):
        """Test getting global instance."""
        journal = get_trade_journal()
        assert journal is not None
        assert isinstance(journal, TradeJournal)

    def test_set_trade_journal(self):
        """Test setting global instance."""
        custom = TradeJournal()
        set_trade_journal(custom)
        assert get_trade_journal() is custom


class TestEdgeCases:
    """Test edge cases."""

    def test_close_nonexistent_trade(self):
        """Test closing nonexistent trade."""
        journal = TradeJournal()
        result = journal.close_trade("fake-id", Decimal("50000"))
        assert result is None

    def test_close_already_closed_trade(self):
        """Test closing already closed trade."""
        journal = TradeJournal()
        trade = journal.add_trade(
            symbol="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
        )
        journal.close_trade(trade.id, Decimal("51000"))
        result = journal.close_trade(trade.id, Decimal("52000"))
        assert result is None

    def test_update_nonexistent_trade(self):
        """Test updating nonexistent trade."""
        journal = TradeJournal()
        result = journal.update_trade("fake-id", stop_loss=Decimal("49000"))
        assert result is None

    def test_add_note_nonexistent_trade(self):
        """Test adding note to nonexistent trade."""
        journal = TradeJournal()
        result = journal.add_note("fake-id", "Test note")
        assert result is None

    def test_breakeven_trade(self):
        """Test breakeven trade outcome."""
        journal = TradeJournal()
        trade = journal.add_trade(
            symbol="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
        )
        journal.close_trade(trade.id, Decimal("50000"))  # Same price
        assert trade.outcome == TradeOutcome.BREAKEVEN

    def test_filter_by_tags(self):
        """Test filtering by tags."""
        journal = TradeJournal()

        t1 = journal.add_trade(
            symbol="BTC-USD-PERP",
            direction=TradeDirection.LONG,
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
            tags=["breakout", "swing"],
        )
        t2 = journal.add_trade(
            symbol="ETH-USD-PERP",
            direction=TradeDirection.SHORT,
            entry_price=Decimal("3000"),
            size=Decimal("1"),
            tags=["scalp"],
        )

        breakout_trades = journal.get_trades(tags=["breakout"])
        assert len(breakout_trades) == 1
        assert breakout_trades[0].id == t1.id
