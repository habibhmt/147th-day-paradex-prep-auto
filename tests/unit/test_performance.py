"""Unit tests for Performance Analytics."""

import pytest
import time
from decimal import Decimal
from datetime import datetime, timedelta

from src.analytics.performance import (
    TradeMetrics,
    DeltaMetrics,
    XPMetrics,
    PerformanceAnalyzer,
)


class TestTradeMetrics:
    """Tests for TradeMetrics dataclass."""

    def test_default_values(self):
        """Should have correct default values."""
        metrics = TradeMetrics()

        assert metrics.total_trades == 0
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 0
        assert metrics.total_pnl == Decimal("0")
        assert metrics.total_volume == Decimal("0")
        assert metrics.total_fees == Decimal("0")

    def test_win_rate_zero_trades(self):
        """Should return 0% win rate with no trades."""
        metrics = TradeMetrics()
        assert metrics.win_rate == 0.0

    def test_win_rate_calculation(self):
        """Should calculate win rate correctly."""
        metrics = TradeMetrics(
            total_trades=10,
            winning_trades=7,
            losing_trades=3,
        )

        assert metrics.win_rate == 70.0

    def test_win_rate_all_winners(self):
        """Should handle 100% win rate."""
        metrics = TradeMetrics(
            total_trades=5,
            winning_trades=5,
            losing_trades=0,
        )

        assert metrics.win_rate == 100.0

    def test_profit_factor_no_loss(self):
        """Should return infinity when no losses."""
        metrics = TradeMetrics(
            largest_win=Decimal("1000"),
            largest_loss=Decimal("0"),
        )

        assert metrics.profit_factor == float("inf")

    def test_profit_factor_no_trades(self):
        """Should return 0 when no wins or losses."""
        metrics = TradeMetrics()
        assert metrics.profit_factor == 0.0

    def test_profit_factor_calculation(self):
        """Should calculate profit factor correctly."""
        metrics = TradeMetrics(
            largest_win=Decimal("1000"),
            largest_loss=Decimal("-500"),
        )

        assert metrics.profit_factor == 2.0

    def test_net_pnl(self):
        """Should calculate net PnL after fees."""
        metrics = TradeMetrics(
            total_pnl=Decimal("1000"),
            total_fees=Decimal("50"),
        )

        assert metrics.net_pnl == Decimal("950")

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        metrics = TradeMetrics(
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            total_pnl=Decimal("500"),
            total_volume=Decimal("100000"),
            total_fees=Decimal("25"),
            avg_trade_size=Decimal("10000"),
            largest_win=Decimal("200"),
            largest_loss=Decimal("-100"),
        )

        d = metrics.to_dict()

        assert d["total_trades"] == 10
        assert d["winning_trades"] == 6
        assert d["losing_trades"] == 4
        assert d["win_rate"] == 60.0
        assert d["total_pnl"] == "500"
        assert d["net_pnl"] == "475"
        assert d["total_volume"] == "100000"
        assert d["profit_factor"] == 2.0


class TestDeltaMetrics:
    """Tests for DeltaMetrics dataclass."""

    def test_default_values(self):
        """Should have correct default values."""
        metrics = DeltaMetrics()

        assert metrics.checks_count == 0
        assert metrics.neutral_count == 0
        assert metrics.rebalance_count == 0
        assert metrics.avg_delta_pct == 0.0
        assert metrics.max_delta_pct == 0.0
        assert metrics.time_in_neutral_pct == 0.0

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        metrics = DeltaMetrics(
            checks_count=100,
            neutral_count=85,
            rebalance_count=3,
            avg_delta_pct=2.5,
            max_delta_pct=8.3,
            time_in_neutral_pct=85.0,
        )

        d = metrics.to_dict()

        assert d["checks_count"] == 100
        assert d["neutral_count"] == 85
        assert d["rebalance_count"] == 3
        assert d["avg_delta_pct"] == 2.5
        assert d["max_delta_pct"] == 8.3
        assert d["time_in_neutral_pct"] == 85.0


class TestXPMetrics:
    """Tests for XPMetrics dataclass."""

    def test_default_values(self):
        """Should have correct default values."""
        metrics = XPMetrics()

        assert metrics.estimated_xp == 0.0
        assert metrics.volume_contribution == Decimal("0")
        assert metrics.position_hours == 0.0
        assert metrics.efficiency_score == 0.0

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        metrics = XPMetrics(
            estimated_xp=15000.0,
            volume_contribution=Decimal("500000"),
            position_hours=48.5,
            efficiency_score=0.03,
        )

        d = metrics.to_dict()

        assert d["estimated_xp"] == 15000.0
        assert d["volume_contribution"] == "500000"
        assert d["position_hours"] == 48.5
        assert d["efficiency_score"] == 0.03


class TestPerformanceAnalyzer:
    """Tests for PerformanceAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create performance analyzer."""
        return PerformanceAnalyzer()

    def test_initial_state(self, analyzer):
        """Should start with empty data."""
        summary = analyzer.get_summary()

        assert summary["accounts_tracked"] == 0
        assert summary["trade_performance"]["total_trades"] == 0

    def test_record_trade(self, analyzer):
        """Should record trade correctly."""
        analyzer.record_trade(
            account_id="acc1",
            pnl=Decimal("100"),
            volume=Decimal("10000"),
            fee=Decimal("5"),
        )

        metrics = analyzer.get_trade_metrics("acc1")

        assert metrics.total_trades == 1
        assert metrics.total_pnl == Decimal("100")
        assert metrics.total_volume == Decimal("10000")
        assert metrics.total_fees == Decimal("5")

    def test_record_winning_trade(self, analyzer):
        """Should track winning trades."""
        analyzer.record_trade("acc1", Decimal("500"), Decimal("10000"))

        metrics = analyzer.get_trade_metrics("acc1")

        assert metrics.winning_trades == 1
        assert metrics.largest_win == Decimal("500")

    def test_record_losing_trade(self, analyzer):
        """Should track losing trades."""
        analyzer.record_trade("acc1", Decimal("-200"), Decimal("10000"))

        metrics = analyzer.get_trade_metrics("acc1")

        assert metrics.losing_trades == 1
        assert metrics.largest_loss == Decimal("-200")

    def test_record_multiple_trades(self, analyzer):
        """Should aggregate multiple trades."""
        analyzer.record_trade("acc1", Decimal("100"), Decimal("5000"))
        analyzer.record_trade("acc1", Decimal("-50"), Decimal("3000"))
        analyzer.record_trade("acc1", Decimal("200"), Decimal("7000"))

        metrics = analyzer.get_trade_metrics("acc1")

        assert metrics.total_trades == 3
        assert metrics.winning_trades == 2
        assert metrics.losing_trades == 1
        assert metrics.total_pnl == Decimal("250")
        assert metrics.total_volume == Decimal("15000")
        assert metrics.largest_win == Decimal("200")
        assert metrics.largest_loss == Decimal("-50")

    def test_avg_trade_size(self, analyzer):
        """Should calculate average trade size."""
        analyzer.record_trade("acc1", Decimal("100"), Decimal("6000"))
        analyzer.record_trade("acc1", Decimal("100"), Decimal("4000"))

        metrics = analyzer.get_trade_metrics("acc1")

        assert metrics.avg_trade_size == Decimal("5000")

    def test_record_trades_multiple_accounts(self, analyzer):
        """Should track separate accounts."""
        analyzer.record_trade("acc1", Decimal("100"), Decimal("10000"))
        analyzer.record_trade("acc2", Decimal("200"), Decimal("20000"))

        acc1_metrics = analyzer.get_trade_metrics("acc1")
        acc2_metrics = analyzer.get_trade_metrics("acc2")

        assert acc1_metrics.total_pnl == Decimal("100")
        assert acc2_metrics.total_pnl == Decimal("200")

    def test_get_trade_metrics_aggregate(self, analyzer):
        """Should aggregate all accounts when no account specified."""
        analyzer.record_trade("acc1", Decimal("100"), Decimal("10000"))
        analyzer.record_trade("acc2", Decimal("200"), Decimal("20000"))
        analyzer.record_trade("acc1", Decimal("-50"), Decimal("5000"))

        total = analyzer.get_trade_metrics()

        assert total.total_trades == 3
        assert total.total_pnl == Decimal("250")
        assert total.total_volume == Decimal("35000")

    def test_get_trade_metrics_nonexistent_account(self, analyzer):
        """Should return empty metrics for unknown account."""
        metrics = analyzer.get_trade_metrics("unknown")

        assert metrics.total_trades == 0
        assert metrics.total_pnl == Decimal("0")

    def test_record_delta_check(self, analyzer):
        """Should record delta checks."""
        analyzer.record_delta_check(2.5, True)
        analyzer.record_delta_check(6.0, False)
        analyzer.record_delta_check(1.0, True)

        delta_metrics = analyzer.get_delta_metrics()

        assert delta_metrics.checks_count == 3
        assert delta_metrics.neutral_count == 2

    def test_get_delta_metrics_empty(self, analyzer):
        """Should return empty delta metrics with no checks."""
        metrics = analyzer.get_delta_metrics()

        assert metrics.checks_count == 0
        assert metrics.neutral_count == 0

    def test_get_delta_metrics_calculation(self, analyzer):
        """Should calculate delta metrics correctly."""
        analyzer.record_delta_check(2.0, True)
        analyzer.record_delta_check(4.0, True)
        analyzer.record_delta_check(8.0, False)
        analyzer.record_delta_check(2.0, True)

        metrics = analyzer.get_delta_metrics()

        assert metrics.checks_count == 4
        assert metrics.neutral_count == 3
        assert metrics.avg_delta_pct == 4.0  # (2+4+8+2)/4
        assert metrics.max_delta_pct == 8.0
        assert metrics.time_in_neutral_pct == 75.0

    def test_get_xp_metrics_empty(self, analyzer):
        """Should return empty XP metrics with no trades."""
        metrics = analyzer.get_xp_metrics()

        assert metrics.estimated_xp == 0.0
        assert metrics.volume_contribution == Decimal("0")

    def test_get_xp_metrics_with_volume(self, analyzer):
        """Should estimate XP based on volume."""
        analyzer.record_trade("acc1", Decimal("0"), Decimal("1000000"))

        metrics = analyzer.get_xp_metrics()

        assert metrics.volume_contribution == Decimal("1000000")
        assert metrics.estimated_xp > 0

    def test_get_hourly_volume(self, analyzer):
        """Should track hourly volume."""
        analyzer.record_trade("acc1", Decimal("0"), Decimal("50000"))
        analyzer.record_trade("acc1", Decimal("0"), Decimal("30000"))

        hourly = analyzer.get_hourly_volume(hours=1)

        assert len(hourly) == 1
        # Current hour should have volume
        current_hour = datetime.now().strftime("%Y-%m-%d-%H")
        assert hourly[current_hour] == Decimal("80000")

    def test_get_hourly_volume_empty_hours(self, analyzer):
        """Should return zeros for hours without trades."""
        hourly = analyzer.get_hourly_volume(hours=24)

        assert len(hourly) == 24
        for hour_key, volume in hourly.items():
            assert volume == Decimal("0")

    def test_get_account_comparison(self, analyzer):
        """Should compare accounts by PnL."""
        analyzer.record_trade("acc1", Decimal("100"), Decimal("10000"))
        analyzer.record_trade("acc2", Decimal("500"), Decimal("20000"))
        analyzer.record_trade("acc3", Decimal("-50"), Decimal("5000"))

        comparison = analyzer.get_account_comparison()

        assert len(comparison) == 3
        # Should be sorted by PnL descending
        assert comparison[0]["account_id"] == "acc2"
        assert comparison[1]["account_id"] == "acc1"
        assert comparison[2]["account_id"] == "acc3"

    def test_get_summary(self, analyzer):
        """Should return complete summary."""
        analyzer.record_trade("acc1", Decimal("100"), Decimal("10000"))
        analyzer.record_delta_check(3.0, True)

        summary = analyzer.get_summary()

        assert "trade_performance" in summary
        assert "delta_neutrality" in summary
        assert "xp_metrics" in summary
        assert "accounts_tracked" in summary
        assert "timestamp" in summary
        assert summary["accounts_tracked"] == 1

    def test_reset(self, analyzer):
        """Should reset all data."""
        analyzer.record_trade("acc1", Decimal("100"), Decimal("10000"))
        analyzer.record_delta_check(5.0, False)

        analyzer.reset()

        assert analyzer.get_trade_metrics().total_trades == 0
        assert analyzer.get_delta_metrics().checks_count == 0
        assert len(analyzer._hourly_volume) == 0
        assert len(analyzer._account_metrics) == 0

    def test_get_streaks_empty(self, analyzer):
        """Should return zeros with no trades."""
        streaks = analyzer.get_streaks()

        assert streaks["current_streak"] == 0
        assert streaks["max_win_streak"] == 0
        assert streaks["max_loss_streak"] == 0

    def test_get_streaks_win_streak(self, analyzer):
        """Should track winning streaks."""
        analyzer.record_trade("acc1", Decimal("100"), Decimal("1000"))
        analyzer.record_trade("acc1", Decimal("50"), Decimal("1000"))
        analyzer.record_trade("acc1", Decimal("200"), Decimal("1000"))

        streaks = analyzer.get_streaks()

        assert streaks["current_streak"] == 3
        assert streaks["current_streak_type"] == "win"
        assert streaks["max_win_streak"] == 3

    def test_get_streaks_loss_streak(self, analyzer):
        """Should track losing streaks."""
        analyzer.record_trade("acc1", Decimal("-100"), Decimal("1000"))
        analyzer.record_trade("acc1", Decimal("-50"), Decimal("1000"))

        streaks = analyzer.get_streaks()

        assert streaks["current_streak"] == 2
        assert streaks["current_streak_type"] == "loss"
        assert streaks["max_loss_streak"] == 2

    def test_get_streaks_mixed(self, analyzer):
        """Should track max streaks in mixed sequences."""
        # Win, Win, Win, Loss, Loss, Win
        analyzer.record_trade("acc1", Decimal("100"), Decimal("1000"))
        analyzer.record_trade("acc1", Decimal("100"), Decimal("1000"))
        analyzer.record_trade("acc1", Decimal("100"), Decimal("1000"))
        analyzer.record_trade("acc1", Decimal("-50"), Decimal("1000"))
        analyzer.record_trade("acc1", Decimal("-50"), Decimal("1000"))
        analyzer.record_trade("acc1", Decimal("100"), Decimal("1000"))

        streaks = analyzer.get_streaks()

        assert streaks["max_win_streak"] == 3
        assert streaks["max_loss_streak"] == 2
        assert streaks["current_streak"] == 1
        assert streaks["current_streak_type"] == "win"

    def test_get_drawdown_empty(self, analyzer):
        """Should return zeros with no trades."""
        dd = analyzer.get_drawdown()

        assert dd["max_drawdown"] == "0"
        assert dd["current_drawdown"] == "0"

    def test_get_drawdown_no_drawdown(self, analyzer):
        """Should handle all profitable trades."""
        analyzer.record_trade("acc1", Decimal("100"), Decimal("1000"))
        analyzer.record_trade("acc1", Decimal("200"), Decimal("1000"))

        dd = analyzer.get_drawdown()

        assert dd["max_drawdown"] == "0"
        assert dd["current_drawdown"] == "0"
        assert dd["peak_equity"] == "300"
        assert dd["current_equity"] == "300"

    def test_get_drawdown_calculation(self, analyzer):
        """Should calculate drawdown correctly."""
        analyzer.record_trade("acc1", Decimal("500"), Decimal("1000"))  # Equity: 500
        analyzer.record_trade("acc1", Decimal("300"), Decimal("1000"))  # Equity: 800 (peak)
        analyzer.record_trade("acc1", Decimal("-200"), Decimal("1000"))  # Equity: 600 (dd: 200)
        analyzer.record_trade("acc1", Decimal("-100"), Decimal("1000"))  # Equity: 500 (dd: 300)
        analyzer.record_trade("acc1", Decimal("100"), Decimal("1000"))  # Equity: 600

        dd = analyzer.get_drawdown()

        assert dd["max_drawdown"] == "300"
        assert dd["current_drawdown"] == "200"  # 800 - 600
        assert dd["peak_equity"] == "800"
        assert dd["current_equity"] == "600"

    def test_largest_win_updates(self, analyzer):
        """Should update largest win correctly."""
        analyzer.record_trade("acc1", Decimal("100"), Decimal("1000"))
        analyzer.record_trade("acc1", Decimal("500"), Decimal("1000"))
        analyzer.record_trade("acc1", Decimal("200"), Decimal("1000"))

        metrics = analyzer.get_trade_metrics("acc1")

        assert metrics.largest_win == Decimal("500")

    def test_largest_loss_updates(self, analyzer):
        """Should update largest loss correctly."""
        analyzer.record_trade("acc1", Decimal("-100"), Decimal("1000"))
        analyzer.record_trade("acc1", Decimal("-500"), Decimal("1000"))
        analyzer.record_trade("acc1", Decimal("-200"), Decimal("1000"))

        metrics = analyzer.get_trade_metrics("acc1")

        assert metrics.largest_loss == Decimal("-500")

    def test_zero_pnl_trade(self, analyzer):
        """Should handle zero PnL trades."""
        analyzer.record_trade("acc1", Decimal("0"), Decimal("1000"))

        metrics = analyzer.get_trade_metrics("acc1")

        assert metrics.total_trades == 1
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 0

    def test_zero_pnl_breaks_streak(self, analyzer):
        """Should break streak on zero PnL trade."""
        analyzer.record_trade("acc1", Decimal("100"), Decimal("1000"))
        analyzer.record_trade("acc1", Decimal("100"), Decimal("1000"))
        analyzer.record_trade("acc1", Decimal("0"), Decimal("1000"))

        streaks = analyzer.get_streaks()

        assert streaks["current_streak"] == 0
        assert streaks["current_streak_type"] == "none"
        assert streaks["max_win_streak"] == 2
