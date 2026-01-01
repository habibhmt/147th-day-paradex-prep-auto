"""Unit tests for Trade Analytics."""

import pytest
import time
from decimal import Decimal
from unittest.mock import MagicMock

from src.analytics.trade_analytics import (
    TimeFrame,
    TradeOutcome,
    TradeData,
    TradeStatistics,
    StreakAnalysis,
    TimeAnalysis,
    MarketAnalysis,
    TradeAnalyzer,
    RealTimeAnalytics,
    get_trade_analyzer,
    reset_trade_analyzer,
)


class TestTimeFrame:
    """Tests for TimeFrame enum."""

    def test_timeframe_values(self):
        """Should have expected timeframe values."""
        assert TimeFrame.HOURLY.value == "1h"
        assert TimeFrame.DAILY.value == "1d"
        assert TimeFrame.WEEKLY.value == "1w"
        assert TimeFrame.MONTHLY.value == "1m"
        assert TimeFrame.ALL_TIME.value == "all"


class TestTradeOutcome:
    """Tests for TradeOutcome enum."""

    def test_outcome_values(self):
        """Should have expected outcome values."""
        assert TradeOutcome.WIN.value == "win"
        assert TradeOutcome.LOSS.value == "loss"
        assert TradeOutcome.BREAKEVEN.value == "breakeven"


class TestTradeData:
    """Tests for TradeData dataclass."""

    @pytest.fixture
    def winning_trade(self):
        """Create winning trade."""
        return TradeData(
            trade_id="t1",
            market="BTC-USD-PERP",
            direction="long",
            entry_price=Decimal("50000"),
            exit_price=Decimal("55000"),
            size=Decimal("1"),
            entry_time=time.time() - 3600,
            exit_time=time.time(),
            pnl=Decimal("5000"),
            fees=Decimal("50"),
        )

    @pytest.fixture
    def losing_trade(self):
        """Create losing trade."""
        return TradeData(
            trade_id="t2",
            market="BTC-USD-PERP",
            direction="long",
            entry_price=Decimal("50000"),
            exit_price=Decimal("48000"),
            size=Decimal("1"),
            entry_time=time.time() - 7200,
            exit_time=time.time() - 3600,
            pnl=Decimal("-2000"),
            fees=Decimal("50"),
        )

    def test_create_trade(self, winning_trade):
        """Should create trade data."""
        assert winning_trade.trade_id == "t1"
        assert winning_trade.market == "BTC-USD-PERP"

    def test_net_pnl(self, winning_trade):
        """Should calculate net PnL."""
        # 5000 - 50 fees = 4950
        assert winning_trade.net_pnl == Decimal("4950")

    def test_net_pnl_with_funding(self):
        """Should include funding in net PnL."""
        trade = TradeData(
            trade_id="t1",
            market="BTC-USD-PERP",
            direction="long",
            entry_price=Decimal("50000"),
            exit_price=Decimal("55000"),
            size=Decimal("1"),
            entry_time=time.time() - 3600,
            exit_time=time.time(),
            pnl=Decimal("5000"),
            fees=Decimal("50"),
            funding=Decimal("100"),
        )

        # 5000 - 50 - 100 = 4850
        assert trade.net_pnl == Decimal("4850")

    def test_pnl_pct(self, winning_trade):
        """Should calculate PnL percentage."""
        # 5000 / 50000 * 100 = 10%
        assert winning_trade.pnl_pct == 10.0

    def test_holding_time(self, winning_trade):
        """Should calculate holding time."""
        # About 1 hour
        assert winning_trade.holding_time >= 0.9
        assert winning_trade.holding_time <= 1.1

    def test_outcome_win(self, winning_trade):
        """Should detect winning trade."""
        assert winning_trade.outcome == TradeOutcome.WIN

    def test_outcome_loss(self, losing_trade):
        """Should detect losing trade."""
        assert losing_trade.outcome == TradeOutcome.LOSS

    def test_outcome_breakeven(self):
        """Should detect breakeven trade."""
        trade = TradeData(
            trade_id="t1",
            market="BTC-USD-PERP",
            direction="long",
            entry_price=Decimal("50000"),
            exit_price=Decimal("50050"),
            size=Decimal("1"),
            entry_time=time.time() - 3600,
            exit_time=time.time(),
            pnl=Decimal("50"),
            fees=Decimal("50"),
        )

        assert trade.outcome == TradeOutcome.BREAKEVEN

    def test_to_dict(self, winning_trade):
        """Should convert to dictionary."""
        d = winning_trade.to_dict()

        assert d["trade_id"] == "t1"
        assert d["outcome"] == "win"


class TestTradeStatistics:
    """Tests for TradeStatistics dataclass."""

    def test_create_statistics(self):
        """Should create statistics."""
        stats = TradeStatistics(
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            total_pnl=Decimal("1000"),
        )

        assert stats.total_trades == 10

    def test_win_rate(self):
        """Should calculate win rate."""
        stats = TradeStatistics(
            winning_trades=6,
            losing_trades=4,
        )

        assert stats.win_rate == 60.0

    def test_win_rate_no_trades(self):
        """Should handle no trades."""
        stats = TradeStatistics()

        assert stats.win_rate == 0.0

    def test_profit_factor(self):
        """Should calculate profit factor."""
        stats = TradeStatistics(
            gross_profit=Decimal("2000"),
            gross_loss=Decimal("-1000"),
        )

        assert stats.profit_factor == 2.0

    def test_profit_factor_no_loss(self):
        """Should handle no losses."""
        stats = TradeStatistics(
            gross_profit=Decimal("2000"),
            gross_loss=Decimal("0"),
        )

        assert stats.profit_factor == float('inf')

    def test_payoff_ratio(self):
        """Should calculate payoff ratio."""
        stats = TradeStatistics(
            avg_win=Decimal("200"),
            avg_loss=Decimal("-100"),
        )

        assert stats.payoff_ratio == 2.0

    def test_expectancy(self):
        """Should calculate expectancy."""
        stats = TradeStatistics(
            total_trades=10,
            net_pnl=Decimal("1000"),
        )

        assert stats.expectancy == Decimal("100")

    def test_to_dict(self):
        """Should convert to dictionary."""
        stats = TradeStatistics(total_trades=5)

        d = stats.to_dict()

        assert "total_trades" in d
        assert "win_rate" in d


class TestStreakAnalysis:
    """Tests for StreakAnalysis dataclass."""

    def test_create_analysis(self):
        """Should create streak analysis."""
        analysis = StreakAnalysis(
            current_streak=3,
            current_streak_type=TradeOutcome.WIN,
            longest_win_streak=5,
            longest_loss_streak=2,
        )

        assert analysis.current_streak == 3
        assert analysis.longest_win_streak == 5

    def test_to_dict(self):
        """Should convert to dictionary."""
        analysis = StreakAnalysis(
            current_streak=2,
            current_streak_type=TradeOutcome.LOSS,
        )

        d = analysis.to_dict()

        assert d["current_streak"] == 2
        assert d["current_streak_type"] == "loss"


class TestTradeAnalyzer:
    """Tests for TradeAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer."""
        return TradeAnalyzer()

    @pytest.fixture
    def sample_trades(self):
        """Create sample trades."""
        base_time = time.time()
        return [
            TradeData("t1", "BTC-USD-PERP", "long", Decimal("50000"), Decimal("55000"),
                     Decimal("1"), base_time - 10000, base_time - 9000,
                     Decimal("5000"), Decimal("50")),
            TradeData("t2", "BTC-USD-PERP", "long", Decimal("55000"), Decimal("53000"),
                     Decimal("1"), base_time - 8000, base_time - 7000,
                     Decimal("-2000"), Decimal("50")),
            TradeData("t3", "ETH-USD-PERP", "short", Decimal("3000"), Decimal("2800"),
                     Decimal("10"), base_time - 6000, base_time - 5000,
                     Decimal("2000"), Decimal("30")),
            TradeData("t4", "BTC-USD-PERP", "long", Decimal("53000"), Decimal("56000"),
                     Decimal("1"), base_time - 4000, base_time - 3000,
                     Decimal("3000"), Decimal("50")),
            TradeData("t5", "BTC-USD-PERP", "short", Decimal("56000"), Decimal("54000"),
                     Decimal("1"), base_time - 2000, base_time - 1000,
                     Decimal("2000"), Decimal("50")),
        ]

    def test_add_trade(self, analyzer):
        """Should add trade."""
        trade = TradeData("t1", "BTC-USD-PERP", "long", Decimal("50000"),
                         Decimal("51000"), Decimal("1"), time.time() - 100,
                         time.time(), Decimal("1000"), Decimal("10"))

        analyzer.add_trade(trade)

        assert len(analyzer._trades) == 1

    def test_add_trades(self, analyzer, sample_trades):
        """Should add multiple trades."""
        analyzer.add_trades(sample_trades)

        assert len(analyzer._trades) == 5

    def test_clear_trades(self, analyzer, sample_trades):
        """Should clear trades."""
        analyzer.add_trades(sample_trades)
        analyzer.clear_trades()

        assert len(analyzer._trades) == 0

    def test_get_trades_filtered_by_market(self, analyzer, sample_trades):
        """Should filter by market."""
        analyzer.add_trades(sample_trades)

        btc_trades = analyzer.get_trades(market="BTC-USD-PERP")

        assert len(btc_trades) == 4  # 4 BTC trades

    def test_calculate_statistics(self, analyzer, sample_trades):
        """Should calculate statistics."""
        analyzer.add_trades(sample_trades)

        stats = analyzer.calculate_statistics()

        assert stats.total_trades == 5
        assert stats.winning_trades == 4
        assert stats.losing_trades == 1

    def test_statistics_win_rate(self, analyzer, sample_trades):
        """Should calculate correct win rate."""
        analyzer.add_trades(sample_trades)

        stats = analyzer.calculate_statistics()

        assert stats.win_rate == 80.0  # 4/5 wins

    def test_statistics_profit_factor(self, analyzer, sample_trades):
        """Should calculate profit factor."""
        analyzer.add_trades(sample_trades)

        stats = analyzer.calculate_statistics()

        assert stats.profit_factor > 0

    def test_analyze_streaks(self, analyzer, sample_trades):
        """Should analyze streaks."""
        analyzer.add_trades(sample_trades)

        streaks = analyzer.analyze_streaks()

        assert streaks.longest_win_streak > 0
        assert len(streaks.win_streaks) > 0

    def test_analyze_by_time(self, analyzer, sample_trades):
        """Should analyze by time."""
        analyzer.add_trades(sample_trades)

        time_analysis = analyzer.analyze_by_time()

        assert time_analysis.best_hour is not None
        assert time_analysis.best_day is not None

    def test_analyze_by_market(self, analyzer, sample_trades):
        """Should analyze by market."""
        analyzer.add_trades(sample_trades)

        market_analysis = analyzer.analyze_by_market()

        assert "BTC-USD-PERP" in market_analysis.market_stats
        assert market_analysis.most_traded == "BTC-USD-PERP"

    def test_calculate_sharpe_ratio(self, analyzer, sample_trades):
        """Should calculate Sharpe ratio."""
        analyzer.add_trades(sample_trades)

        sharpe = analyzer.calculate_sharpe_ratio()

        assert isinstance(sharpe, float)

    def test_calculate_sortino_ratio(self, analyzer, sample_trades):
        """Should calculate Sortino ratio."""
        analyzer.add_trades(sample_trades)

        sortino = analyzer.calculate_sortino_ratio()

        assert isinstance(sortino, float)

    def test_calculate_calmar_ratio(self, analyzer, sample_trades):
        """Should calculate Calmar ratio."""
        analyzer.add_trades(sample_trades)

        calmar = analyzer.calculate_calmar_ratio()

        assert isinstance(calmar, float)

    def test_get_full_report(self, analyzer, sample_trades):
        """Should get full report."""
        analyzer.add_trades(sample_trades)

        report = analyzer.get_full_report()

        assert "statistics" in report
        assert "streaks" in report
        assert "time_analysis" in report
        assert "market_analysis" in report
        assert "sharpe_ratio" in report

    def test_empty_analyzer(self, analyzer):
        """Should handle empty analyzer."""
        stats = analyzer.calculate_statistics()

        assert stats.total_trades == 0
        assert stats.win_rate == 0.0


class TestRealTimeAnalytics:
    """Tests for RealTimeAnalytics."""

    @pytest.fixture
    def analytics(self):
        """Create real-time analytics."""
        return RealTimeAnalytics()

    def test_on_trade_complete(self, analytics):
        """Should update stats on trade complete."""
        trade = TradeData("t1", "BTC-USD-PERP", "long", Decimal("50000"),
                         Decimal("51000"), Decimal("1"), time.time() - 100,
                         time.time(), Decimal("1000"), Decimal("10"))

        stats = analytics.on_trade_complete(trade)

        assert stats.total_trades == 1

    def test_callback_called(self, analytics):
        """Should call callbacks."""
        callback = MagicMock()
        analytics.add_callback(callback)

        trade = TradeData("t1", "BTC-USD-PERP", "long", Decimal("50000"),
                         Decimal("51000"), Decimal("1"), time.time() - 100,
                         time.time(), Decimal("1000"), Decimal("10"))

        analytics.on_trade_complete(trade)

        callback.assert_called_once()

    def test_window_size(self, analytics):
        """Should respect window size."""
        analytics._window_size = 5

        for i in range(10):
            trade = TradeData(f"t{i}", "BTC-USD-PERP", "long", Decimal("50000"),
                             Decimal("51000"), Decimal("1"), time.time() - 100,
                             time.time(), Decimal("100"), Decimal("1"))
            analytics.on_trade_complete(trade)

        assert len(analytics._trades) == 5

    def test_get_current_stats(self, analytics):
        """Should get current stats."""
        trade = TradeData("t1", "BTC-USD-PERP", "long", Decimal("50000"),
                         Decimal("51000"), Decimal("1"), time.time() - 100,
                         time.time(), Decimal("1000"), Decimal("10"))

        analytics.on_trade_complete(trade)

        stats = analytics.get_current_stats()

        assert stats.total_trades == 1


class TestGlobalAnalyzer:
    """Tests for global analyzer functions."""

    def test_get_trade_analyzer(self):
        """Should get or create analyzer."""
        reset_trade_analyzer()

        a1 = get_trade_analyzer()
        a2 = get_trade_analyzer()

        assert a1 is a2

    def test_reset_trade_analyzer(self):
        """Should reset analyzer."""
        a1 = get_trade_analyzer()
        reset_trade_analyzer()
        a2 = get_trade_analyzer()

        assert a1 is not a2


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_size_trade(self):
        """Should handle zero size."""
        trade = TradeData("t1", "BTC-USD-PERP", "long", Decimal("50000"),
                         Decimal("51000"), Decimal("0"), time.time() - 100,
                         time.time(), Decimal("0"), Decimal("0"))

        assert trade.pnl_pct == 0.0

    def test_all_losing_trades(self):
        """Should handle all losses."""
        analyzer = TradeAnalyzer()

        for i in range(5):
            trade = TradeData(f"t{i}", "BTC-USD-PERP", "long", Decimal("50000"),
                             Decimal("49000"), Decimal("1"), time.time() - 100,
                             time.time(), Decimal("-1000"), Decimal("10"))
            analyzer.add_trade(trade)

        stats = analyzer.calculate_statistics()

        assert stats.win_rate == 0.0
        assert stats.winning_trades == 0

    def test_all_winning_trades(self):
        """Should handle all wins."""
        analyzer = TradeAnalyzer()

        for i in range(5):
            trade = TradeData(f"t{i}", "BTC-USD-PERP", "long", Decimal("50000"),
                             Decimal("51000"), Decimal("1"), time.time() - 100,
                             time.time(), Decimal("1000"), Decimal("10"))
            analyzer.add_trade(trade)

        stats = analyzer.calculate_statistics()

        assert stats.win_rate == 100.0

    def test_single_trade(self):
        """Should handle single trade."""
        analyzer = TradeAnalyzer()

        trade = TradeData("t1", "BTC-USD-PERP", "long", Decimal("50000"),
                         Decimal("51000"), Decimal("1"), time.time() - 100,
                         time.time(), Decimal("1000"), Decimal("10"))
        analyzer.add_trade(trade)

        stats = analyzer.calculate_statistics()
        sharpe = analyzer.calculate_sharpe_ratio()

        assert stats.total_trades == 1
        assert sharpe == 0.0  # Can't calculate with 1 trade
