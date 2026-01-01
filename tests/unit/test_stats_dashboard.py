"""Unit tests for Stats Dashboard."""

import pytest
import time
from unittest.mock import MagicMock, patch

from src.cli.stats_dashboard import (
    StatsDashboard,
    StatsDashboardConfig,
    MetricValue,
    DashboardTheme,
)


class TestDashboardTheme:
    """Tests for DashboardTheme enum."""

    def test_theme_values(self):
        """Should have expected theme values."""
        assert DashboardTheme.DARK.value == "dark"
        assert DashboardTheme.LIGHT.value == "light"
        assert DashboardTheme.HACKER.value == "hacker"


class TestMetricValue:
    """Tests for MetricValue dataclass."""

    def test_create_metric(self):
        """Should create metric correctly."""
        metric = MetricValue(
            name="Test Metric",
            current=100.0,
            previous=90.0,
            unit="$",
        )

        assert metric.name == "Test Metric"
        assert metric.current == 100.0
        assert metric.previous == 90.0
        assert metric.unit == "$"

    def test_change_calculation(self):
        """Should calculate change correctly."""
        metric = MetricValue(name="Test", current=100.0, previous=80.0)

        assert metric.change == 20.0

    def test_change_pct_calculation(self):
        """Should calculate percentage change correctly."""
        metric = MetricValue(name="Test", current=120.0, previous=100.0)

        assert metric.change_pct == 20.0

    def test_change_pct_zero_previous(self):
        """Should handle zero previous value."""
        metric = MetricValue(name="Test", current=100.0, previous=0.0)

        assert metric.change_pct == 0.0

    def test_trend_up(self):
        """Should show up trend."""
        metric = MetricValue(name="Test", current=100.0, previous=80.0)

        assert metric.trend == "↑"
        assert metric.trend_color == "green"

    def test_trend_down(self):
        """Should show down trend."""
        metric = MetricValue(name="Test", current=80.0, previous=100.0)

        assert metric.trend == "↓"
        assert metric.trend_color == "red"

    def test_trend_unchanged(self):
        """Should show unchanged trend."""
        metric = MetricValue(name="Test", current=100.0, previous=100.0)

        assert metric.trend == "→"
        assert metric.trend_color == "yellow"

    def test_formatted_value(self):
        """Should format value correctly."""
        metric = MetricValue(
            name="Test",
            current=1234.567,
            unit="$",
            format_spec=".2f",
        )

        assert metric.formatted_value() == "1234.57$"

    def test_formatted_value_integer(self):
        """Should format integer values."""
        metric = MetricValue(
            name="Test",
            current=1000.0,
            format_spec=",.0f",
        )

        assert metric.formatted_value() == "1,000"

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        metric = MetricValue(
            name="Test",
            current=100.0,
            previous=80.0,
            unit="%",
        )

        d = metric.to_dict()

        assert d["name"] == "Test"
        assert d["current"] == 100.0
        assert d["previous"] == 80.0
        assert d["change"] == 20.0
        assert d["change_pct"] == 25.0
        assert d["unit"] == "%"


class TestStatsDashboardConfig:
    """Tests for StatsDashboardConfig dataclass."""

    def test_default_config(self):
        """Should have correct defaults."""
        config = StatsDashboardConfig()

        assert config.refresh_rate == 1.0
        assert config.theme == DashboardTheme.DARK
        assert config.show_charts is True
        assert config.max_history == 60
        assert config.compact_mode is False

    def test_custom_config(self):
        """Should accept custom values."""
        config = StatsDashboardConfig(
            refresh_rate=0.5,
            theme=DashboardTheme.LIGHT,
            compact_mode=True,
        )

        assert config.refresh_rate == 0.5
        assert config.theme == DashboardTheme.LIGHT
        assert config.compact_mode is True


class TestStatsDashboard:
    """Tests for StatsDashboard."""

    @pytest.fixture
    def dashboard(self):
        """Create dashboard instance."""
        return StatsDashboard()

    @pytest.fixture
    def compact_dashboard(self):
        """Create compact dashboard instance."""
        config = StatsDashboardConfig(compact_mode=True)
        return StatsDashboard(config=config)

    def test_initial_state(self, dashboard):
        """Should start with correct initial state."""
        assert dashboard._running is False
        assert len(dashboard._metrics) > 0
        assert len(dashboard._pnl_history) == 0
        assert len(dashboard._volume_history) == 0

    def test_default_metrics_initialized(self, dashboard):
        """Should initialize default metrics."""
        assert "pnl" in dashboard._metrics
        assert "volume_24h" in dashboard._metrics
        assert "win_rate" in dashboard._metrics
        assert "delta_pct" in dashboard._metrics
        assert "estimated_xp" in dashboard._metrics

    def test_update_metric(self, dashboard):
        """Should update metric value."""
        dashboard.update_metric("pnl", 1000.0)

        metric = dashboard._metrics["pnl"]
        assert metric.current == 1000.0

    def test_update_metric_preserves_previous(self, dashboard):
        """Should preserve previous value."""
        dashboard.update_metric("pnl", 500.0)
        dashboard.update_metric("pnl", 1000.0)

        metric = dashboard._metrics["pnl"]
        assert metric.current == 1000.0
        assert metric.previous == 500.0

    def test_update_metrics_batch(self, dashboard):
        """Should update multiple metrics at once."""
        dashboard.update_metrics({
            "pnl": 1000.0,
            "volume_24h": 50000.0,
            "win_rate": 65.0,
        })

        assert dashboard._metrics["pnl"].current == 1000.0
        assert dashboard._metrics["volume_24h"].current == 50000.0
        assert dashboard._metrics["win_rate"].current == 65.0

    def test_update_nonexistent_metric(self, dashboard):
        """Should ignore nonexistent metrics."""
        dashboard.update_metric("nonexistent", 100.0)
        # Should not raise

    def test_record_pnl(self, dashboard):
        """Should record PnL history."""
        dashboard.record_pnl(100.0)
        dashboard.record_pnl(150.0)
        dashboard.record_pnl(120.0)

        assert len(dashboard._pnl_history) == 3
        assert dashboard._pnl_history == [100.0, 150.0, 120.0]

    def test_record_pnl_limit(self, dashboard):
        """Should limit PnL history."""
        dashboard.config.max_history = 5

        for i in range(10):
            dashboard.record_pnl(float(i))

        assert len(dashboard._pnl_history) == 5
        assert dashboard._pnl_history[0] == 5.0  # Oldest kept

    def test_record_volume(self, dashboard):
        """Should record volume history."""
        dashboard.record_volume(10000.0)
        dashboard.record_volume(15000.0)

        assert len(dashboard._volume_history) == 2

    def test_record_volume_limit(self, dashboard):
        """Should limit volume history."""
        dashboard.config.max_history = 3

        for i in range(7):
            dashboard.record_volume(float(i * 1000))

        assert len(dashboard._volume_history) == 3

    def test_create_layout(self, dashboard):
        """Should create valid layout."""
        layout = dashboard.create_layout()

        assert layout is not None
        # Layout has children with these names
        assert layout["header"] is not None
        assert layout["body"] is not None
        assert layout["footer"] is not None

    def test_create_compact_layout(self, compact_dashboard):
        """Should create compact layout."""
        layout = compact_dashboard.create_layout()

        assert layout is not None
        # Compact mode has different structure

    def test_render_header(self, dashboard):
        """Should render header panel."""
        panel = dashboard.render_header()

        assert panel is not None
        assert panel.title is None or "Delta" not in str(panel.title)

    def test_render_footer(self, dashboard):
        """Should render footer panel."""
        panel = dashboard.render_footer()

        assert panel is not None

    def test_render_key_metrics(self, dashboard):
        """Should render key metrics panel."""
        dashboard.update_metrics({
            "pnl": 1500.0,
            "volume_24h": 100000.0,
        })

        panel = dashboard.render_key_metrics()

        assert panel is not None
        assert panel.title == "Key Metrics"

    def test_render_performance(self, dashboard):
        """Should render performance panel."""
        dashboard.record_pnl(100.0)
        dashboard.record_pnl(150.0)

        panel = dashboard.render_performance()

        assert panel is not None
        assert panel.title == "Performance"

    def test_render_trades_table_empty(self, dashboard):
        """Should render empty trades table."""
        panel = dashboard.render_trades_table()

        assert panel is not None
        assert "Recent Trades" in panel.title

    def test_render_trades_table_with_data(self, dashboard):
        """Should render trades table with data."""
        trades = [
            {
                "time": "10:30:00",
                "market": "BTC-USD-PERP",
                "side": "BUY",
                "size": 1.5,
                "pnl": 50.0,
            },
            {
                "time": "10:31:00",
                "market": "ETH-USD-PERP",
                "side": "SELL",
                "size": 10.0,
                "pnl": -25.0,
            },
        ]

        panel = dashboard.render_trades_table(trades)

        assert panel is not None
        assert "2" in panel.title  # Trade count

    def test_render_risk_panel(self, dashboard):
        """Should render risk panel."""
        dashboard.update_metrics({
            "delta_pct": 3.0,
            "drawdown": 2.0,
        })

        panel = dashboard.render_risk_panel()

        assert panel is not None
        assert panel.title == "Risk Monitor"

    def test_create_sparkline(self, dashboard):
        """Should create sparkline from data."""
        data = [1, 2, 3, 4, 5]
        sparkline = dashboard._create_sparkline(data, width=5)

        assert len(sparkline) == 5
        # First char should be lowest, last should be highest
        assert sparkline[0] == "▁"
        assert sparkline[-1] == "█"

    def test_create_sparkline_empty(self, dashboard):
        """Should handle empty data."""
        sparkline = dashboard._create_sparkline([], width=10)

        assert len(sparkline) == 10
        assert sparkline == "─" * 10

    def test_create_sparkline_single_value(self, dashboard):
        """Should handle single value."""
        sparkline = dashboard._create_sparkline([5], width=5)

        # Single value should produce some output
        assert len(sparkline) > 0

    def test_create_progress_bar(self, dashboard):
        """Should create progress bar."""
        bar = dashboard._create_progress_bar(50, width=10)

        assert "[" in bar
        assert "]" in bar
        assert "█" in bar
        assert "░" in bar

    def test_create_progress_bar_empty(self, dashboard):
        """Should create empty progress bar."""
        bar = dashboard._create_progress_bar(0, width=10)

        assert "█" not in bar.replace("[", "").replace("]", "")

    def test_create_progress_bar_full(self, dashboard):
        """Should create full progress bar."""
        bar = dashboard._create_progress_bar(100, width=10)

        assert "░" not in bar.replace("[", "").replace("]", "")

    def test_stop(self, dashboard):
        """Should stop dashboard."""
        dashboard._running = True
        dashboard.stop()

        assert dashboard._running is False

    def test_get_metrics(self, dashboard):
        """Should return all metrics."""
        dashboard.update_metric("pnl", 1000.0)

        metrics = dashboard.get_metrics()

        assert "pnl" in metrics
        assert metrics["pnl"]["current"] == 1000.0

    def test_get_status(self, dashboard):
        """Should return dashboard status."""
        dashboard._running = True

        status = dashboard.get_status()

        assert "running" in status
        assert "uptime_seconds" in status
        assert "refresh_rate" in status
        assert "theme" in status
        assert status["running"] is True

    def test_set_data_provider(self, dashboard):
        """Should set data provider."""
        mock_provider = MagicMock()

        dashboard.set_data_provider(mock_provider)

        assert dashboard._data_provider == mock_provider

    def test_metric_trend_colors_in_render(self, dashboard):
        """Should use correct colors for trends."""
        # Set up a positive trend
        dashboard.update_metric("pnl", 100.0)
        dashboard.update_metric("pnl", 150.0)

        # Get the metric
        metric = dashboard._metrics["pnl"]
        assert metric.trend_color == "green"

    def test_dashboard_uptime_tracking(self, dashboard):
        """Should track uptime."""
        # Wait a tiny bit
        time.sleep(0.01)

        status = dashboard.get_status()

        assert status["uptime_seconds"] > 0

    def test_metrics_all_have_units(self, dashboard):
        """Should have units for all metrics."""
        # Check that key metrics have proper units
        assert dashboard._metrics["pnl"].unit == "$"
        assert dashboard._metrics["win_rate"].unit == "%"
        assert dashboard._metrics["delta_pct"].unit == "%"
