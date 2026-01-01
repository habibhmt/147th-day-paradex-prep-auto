"""Unit tests for XP optimization modules."""

import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch
import time

from src.xp.volume_tracker import VolumeTracker, TradeRecord, VolumeStats
from src.xp.optimizer import XPOptimizer, XPRecommendation
from src.core.position_manager import PositionManager, Position


class TestTradeRecord:
    """Tests for TradeRecord dataclass."""

    def test_create_trade_record(self):
        """Should create trade record correctly."""
        record = TradeRecord(
            account_id="acc1",
            market="BTC-USD-PERP",
            size=Decimal("1.5"),
            price=Decimal("50000"),
            side="BUY",
        )

        assert record.account_id == "acc1"
        assert record.market == "BTC-USD-PERP"
        assert record.size == Decimal("1.5")
        assert record.price == Decimal("50000")
        assert record.side == "BUY"

    def test_volume_property(self):
        """Should calculate volume correctly."""
        record = TradeRecord(
            account_id="acc1",
            market="BTC-USD-PERP",
            size=Decimal("2"),
            price=Decimal("50000"),
            side="BUY",
        )

        assert record.volume == Decimal("100000")


class TestVolumeStats:
    """Tests for VolumeStats dataclass."""

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        stats = VolumeStats(
            account_id="acc1",
            total_volume=Decimal("100000"),
            volume_24h=Decimal("50000"),
            volume_7d=Decimal("80000"),
            trade_count=10,
            trade_count_24h=5,
            avg_trade_size=Decimal("10000"),
        )

        d = stats.to_dict()

        assert d["account_id"] == "acc1"
        assert d["total_volume"] == "100000"
        assert d["volume_24h"] == "50000"
        assert d["trade_count"] == 10


class TestVolumeTracker:
    """Tests for VolumeTracker."""

    @pytest.fixture
    def tracker(self):
        """Create fresh volume tracker."""
        return VolumeTracker()

    def test_record_trade(self, tracker):
        """Should record trade correctly."""
        tracker.record_trade(
            account_id="acc1",
            market="BTC-USD-PERP",
            size=Decimal("1"),
            price=Decimal("50000"),
            side="BUY",
        )

        stats = tracker.get_account_stats("acc1")
        assert stats.trade_count == 1
        assert stats.total_volume == Decimal("50000")

    def test_record_multiple_trades(self, tracker):
        """Should accumulate volume correctly."""
        tracker.record_trade(
            account_id="acc1",
            market="BTC-USD-PERP",
            size=Decimal("1"),
            price=Decimal("50000"),
            side="BUY",
        )
        tracker.record_trade(
            account_id="acc1",
            market="BTC-USD-PERP",
            size=Decimal("2"),
            price=Decimal("50000"),
            side="SELL",
        )

        stats = tracker.get_account_stats("acc1")
        assert stats.trade_count == 2
        assert stats.total_volume == Decimal("150000")

    def test_get_account_stats_unknown_account(self, tracker):
        """Should return empty stats for unknown account."""
        stats = tracker.get_account_stats("unknown")

        assert stats.account_id == "unknown"
        assert stats.trade_count == 0
        assert stats.total_volume == Decimal("0")

    def test_get_all_stats(self, tracker):
        """Should return stats for all accounts."""
        tracker.record_trade("acc1", "BTC-USD-PERP", Decimal("1"), Decimal("50000"), "BUY")
        tracker.record_trade("acc2", "ETH-USD-PERP", Decimal("10"), Decimal("3000"), "SELL")

        all_stats = tracker.get_all_stats()

        assert len(all_stats) == 2
        account_ids = [s.account_id for s in all_stats]
        assert "acc1" in account_ids
        assert "acc2" in account_ids

    def test_get_total_volume_24h(self, tracker):
        """Should calculate total 24h volume."""
        tracker.record_trade("acc1", "BTC-USD-PERP", Decimal("1"), Decimal("50000"), "BUY")
        tracker.record_trade("acc2", "BTC-USD-PERP", Decimal("1"), Decimal("50000"), "SELL")

        total = tracker.get_total_volume_24h()
        assert total == Decimal("100000")

    def test_get_total_volume_7d(self, tracker):
        """Should calculate total 7d volume."""
        tracker.record_trade("acc1", "BTC-USD-PERP", Decimal("1"), Decimal("50000"), "BUY")
        tracker.record_trade("acc2", "BTC-USD-PERP", Decimal("2"), Decimal("50000"), "SELL")

        total = tracker.get_total_volume_7d()
        assert total == Decimal("150000")

    def test_estimate_xp_share(self, tracker):
        """Should estimate XP share based on volume proportion."""
        tracker.record_trade("acc1", "BTC-USD-PERP", Decimal("1"), Decimal("50000"), "BUY")
        tracker.record_trade("acc2", "BTC-USD-PERP", Decimal("3"), Decimal("50000"), "SELL")
        # acc1: 50000, acc2: 150000, total: 200000
        # acc1 share: 25%, acc2 share: 75%

        weekly_pool = 4_000_000
        xp1 = tracker.estimate_xp_share("acc1", weekly_pool)
        xp2 = tracker.estimate_xp_share("acc2", weekly_pool)

        assert xp1 == 1_000_000  # 25%
        assert xp2 == 3_000_000  # 75%

    def test_estimate_xp_share_unknown_account(self, tracker):
        """Should return 0 for unknown account."""
        xp = tracker.estimate_xp_share("unknown")
        assert xp == 0.0

    def test_get_volume_by_market(self, tracker):
        """Should break down volume by market."""
        tracker.record_trade("acc1", "BTC-USD-PERP", Decimal("1"), Decimal("50000"), "BUY")
        tracker.record_trade("acc1", "ETH-USD-PERP", Decimal("10"), Decimal("3000"), "SELL")
        tracker.record_trade("acc1", "BTC-USD-PERP", Decimal("1"), Decimal("50000"), "BUY")

        by_market = tracker.get_volume_by_market("acc1")

        assert by_market["BTC-USD-PERP"] == Decimal("100000")
        assert by_market["ETH-USD-PERP"] == Decimal("30000")

    def test_get_recent_trades(self, tracker):
        """Should return recent trades."""
        for i in range(5):
            tracker.record_trade("acc1", "BTC-USD-PERP", Decimal("1"), Decimal("50000"), "BUY")

        recent = tracker.get_recent_trades("acc1", limit=3)

        assert len(recent) == 3

    def test_summary(self, tracker):
        """Should return summary dictionary."""
        tracker.record_trade("acc1", "BTC-USD-PERP", Decimal("1"), Decimal("50000"), "BUY")

        summary = tracker.summary()

        assert summary["total_trades"] == 1
        assert summary["accounts_tracked"] == 1
        assert "total_volume_24h" in summary


class TestXPOptimizer:
    """Tests for XPOptimizer."""

    @pytest.fixture
    def position_manager(self):
        """Create position manager."""
        return PositionManager()

    @pytest.fixture
    def volume_tracker(self):
        """Create volume tracker."""
        return VolumeTracker()

    @pytest.fixture
    def optimizer(self, position_manager, volume_tracker):
        """Create XP optimizer."""
        return XPOptimizer(
            position_manager=position_manager,
            volume_tracker=volume_tracker,
            min_position_duration=24.0,
            optimal_position_duration=48.0,
            target_daily_volume=100000.0,
        )

    def test_should_close_position_no_position(self, optimizer):
        """Should return False when no position exists."""
        should_close, reason = optimizer.should_close_position("acc1", "BTC-USD-PERP")

        assert should_close is False
        assert "No position" in reason

    def test_should_close_position_below_min_duration(self, optimizer, position_manager):
        """Should not close position below min duration."""
        # Create position opened recently (10 hours ago)
        pos = Position(
            account_id="acc1",
            market="BTC-USD-PERP",
            side="LONG",
            size=Decimal("1000"),
            entry_price=Decimal("50000"),
            opened_at=time.time() - (10 * 3600),  # 10 hours ago
        )
        position_manager.update_position(pos)

        should_close, reason = optimizer.should_close_position("acc1", "BTC-USD-PERP")

        assert should_close is False
        assert "Hold for XP" in reason

    def test_should_close_position_optimal_duration(self, optimizer, position_manager):
        """Should close position at optimal duration."""
        # Create position opened 50 hours ago (> 48h optimal)
        pos = Position(
            account_id="acc1",
            market="BTC-USD-PERP",
            side="LONG",
            size=Decimal("1000"),
            entry_price=Decimal("50000"),
            opened_at=time.time() - (50 * 3600),  # 50 hours ago
        )
        position_manager.update_position(pos)

        should_close, reason = optimizer.should_close_position("acc1", "BTC-USD-PERP")

        assert should_close is True
        assert "Optimal duration" in reason

    def test_get_positions_to_close(self, optimizer, position_manager):
        """Should list positions ready for closing."""
        # Position ready (60 hours)
        pos1 = Position(
            account_id="acc1",
            market="BTC-USD-PERP",
            side="LONG",
            size=Decimal("1000"),
            entry_price=Decimal("50000"),
            opened_at=time.time() - (60 * 3600),
        )
        # Position not ready (10 hours)
        pos2 = Position(
            account_id="acc2",
            market="BTC-USD-PERP",
            side="SHORT",
            size=Decimal("1000"),
            entry_price=Decimal("50000"),
            opened_at=time.time() - (10 * 3600),
        )
        position_manager.update_position(pos1)
        position_manager.update_position(pos2)

        to_close = optimizer.get_positions_to_close("BTC-USD-PERP")

        assert len(to_close) == 1
        assert to_close[0]["account_id"] == "acc1"

    def test_get_volume_progress_below_target(self, optimizer, volume_tracker):
        """Should show progress below target."""
        volume_tracker.record_trade("acc1", "BTC-USD-PERP", Decimal("1"), Decimal("25000"), "BUY")

        progress = optimizer.get_volume_progress()

        assert progress["progress_pct"] == 25.0
        assert progress["on_track"] is False
        assert Decimal(progress["remaining"]) == Decimal("75000")

    def test_get_volume_progress_on_target(self, optimizer, volume_tracker):
        """Should show on track when target met."""
        volume_tracker.record_trade("acc1", "BTC-USD-PERP", Decimal("2"), Decimal("50000"), "BUY")

        progress = optimizer.get_volume_progress()

        assert progress["progress_pct"] == 100
        assert progress["on_track"] is True

    def test_get_recommendations_low_volume(self, optimizer, volume_tracker):
        """Should recommend increasing volume when low."""
        # Small volume
        volume_tracker.record_trade("acc1", "BTC-USD-PERP", Decimal("0.1"), Decimal("50000"), "BUY")

        recommendations = optimizer.get_recommendations()

        volume_recs = [r for r in recommendations if r.action == "increase_volume"]
        assert len(volume_recs) == 1
        assert volume_recs[0].priority == "high"

    def test_get_recommendations_positions_to_rotate(self, optimizer, position_manager, volume_tracker):
        """Should recommend rotating old positions."""
        # Ensure volume target is met
        volume_tracker.record_trade("acc1", "BTC-USD-PERP", Decimal("2"), Decimal("50000"), "BUY")

        # Old position
        pos = Position(
            account_id="acc1",
            market="BTC-USD-PERP",
            side="LONG",
            size=Decimal("1000"),
            entry_price=Decimal("50000"),
            opened_at=time.time() - (60 * 3600),
        )
        position_manager.update_position(pos)

        recommendations = optimizer.get_recommendations()

        rotate_recs = [r for r in recommendations if r.action == "rotate_positions"]
        assert len(rotate_recs) == 1
        assert rotate_recs[0].priority == "medium"

    def test_calculate_optimal_trade_frequency(self, optimizer):
        """Should calculate optimal trade frequency."""
        freq = optimizer.calculate_optimal_trade_frequency()

        assert freq["trades_per_day"] == 0.5  # 24h / 48h
        assert freq["trade_interval_hours"] == 48.0
        assert "volume_per_trade" in freq

    def test_estimate_weekly_xp_single_account(self, optimizer, volume_tracker):
        """Should estimate XP for single account."""
        volume_tracker.record_trade("acc1", "BTC-USD-PERP", Decimal("2"), Decimal("50000"), "BUY")

        estimate = optimizer.estimate_weekly_xp("acc1")

        assert estimate["account_id"] == "acc1"
        assert estimate["estimated_xp"] == 4_000_000  # 100% share
        assert estimate["weekly_pool"] == 4_000_000

    def test_estimate_weekly_xp_all_accounts(self, optimizer, volume_tracker):
        """Should estimate XP for all accounts."""
        volume_tracker.record_trade("acc1", "BTC-USD-PERP", Decimal("1"), Decimal("50000"), "BUY")
        volume_tracker.record_trade("acc2", "BTC-USD-PERP", Decimal("1"), Decimal("50000"), "SELL")

        estimate = optimizer.estimate_weekly_xp()

        assert estimate["total_estimated_xp"] == 4_000_000
        assert len(estimate["accounts"]) == 2

    def test_get_distribution_status(self, optimizer):
        """Should return distribution status."""
        status = optimizer.get_distribution_status()

        assert "next_distribution" in status
        assert "hours_remaining" in status
        assert status["weekly_pool"] == 4_000_000

    def test_summary(self, optimizer, volume_tracker):
        """Should return complete summary."""
        volume_tracker.record_trade("acc1", "BTC-USD-PERP", Decimal("2"), Decimal("50000"), "BUY")

        summary = optimizer.summary()

        assert "volume_progress" in summary
        assert "distribution_status" in summary
        assert "recommendations_count" in summary
        assert "estimated_xp" in summary
        assert "trade_frequency" in summary
