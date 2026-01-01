"""Unit tests for DeltaCalculator."""

import pytest
from decimal import Decimal

from src.core.position_manager import PositionManager, Position
from src.core.delta_calculator import DeltaCalculator, DeltaReport


class TestDeltaCalculator:
    """Tests for DeltaCalculator class."""

    @pytest.fixture
    def position_manager(self):
        """Create a fresh PositionManager."""
        return PositionManager()

    @pytest.fixture
    def calculator(self, position_manager):
        """Create a DeltaCalculator with default threshold."""
        return DeltaCalculator(position_manager, neutrality_threshold=5.0)

    def test_empty_positions_is_neutral(self, calculator):
        """No positions should be considered neutral."""
        report = calculator.calculate_delta("BTC-USD-PERP")

        assert report.is_neutral is True
        assert report.net_delta == Decimal("0")
        assert report.delta_percentage == 0.0

    def test_perfectly_balanced(self, calculator, position_manager):
        """Equal long and short should be neutral."""
        # Add long position
        position_manager.update_position(Position(
            account_id="acc1",
            market="BTC-USD-PERP",
            side="LONG",
            size=Decimal("1000"),
            entry_price=Decimal("50000"),
        ))

        # Add equal short position
        position_manager.update_position(Position(
            account_id="acc2",
            market="BTC-USD-PERP",
            side="SHORT",
            size=Decimal("1000"),
            entry_price=Decimal("50000"),
        ))

        report = calculator.calculate_delta("BTC-USD-PERP")

        assert report.is_neutral is True
        assert report.net_delta == Decimal("0")
        assert report.delta_percentage == 0.0
        assert report.total_long == Decimal("1000")
        assert report.total_short == Decimal("1000")

    def test_imbalanced_long_heavy(self, calculator, position_manager):
        """More longs than shorts should not be neutral."""
        position_manager.update_position(Position(
            account_id="acc1",
            market="BTC-USD-PERP",
            side="LONG",
            size=Decimal("1500"),
            entry_price=Decimal("50000"),
        ))

        position_manager.update_position(Position(
            account_id="acc2",
            market="BTC-USD-PERP",
            side="SHORT",
            size=Decimal("1000"),
            entry_price=Decimal("50000"),
        ))

        report = calculator.calculate_delta("BTC-USD-PERP")

        assert report.is_neutral is False
        assert report.net_delta == Decimal("500")  # 1500 - 1000
        assert report.delta_percentage == 20.0  # 500 / 2500 * 100

    def test_imbalanced_short_heavy(self, calculator, position_manager):
        """More shorts than longs should not be neutral."""
        position_manager.update_position(Position(
            account_id="acc1",
            market="BTC-USD-PERP",
            side="LONG",
            size=Decimal("1000"),
            entry_price=Decimal("50000"),
        ))

        position_manager.update_position(Position(
            account_id="acc2",
            market="BTC-USD-PERP",
            side="SHORT",
            size=Decimal("1500"),
            entry_price=Decimal("50000"),
        ))

        report = calculator.calculate_delta("BTC-USD-PERP")

        assert report.is_neutral is False
        assert report.net_delta == Decimal("-500")  # 1000 - 1500

    def test_within_threshold_is_neutral(self, calculator, position_manager):
        """Small imbalance within threshold should be neutral."""
        # 4% imbalance (within 5% threshold)
        position_manager.update_position(Position(
            account_id="acc1",
            market="BTC-USD-PERP",
            side="LONG",
            size=Decimal("520"),
            entry_price=Decimal("50000"),
        ))

        position_manager.update_position(Position(
            account_id="acc2",
            market="BTC-USD-PERP",
            side="SHORT",
            size=Decimal("480"),
            entry_price=Decimal("50000"),
        ))

        report = calculator.calculate_delta("BTC-USD-PERP")

        # Delta = 40, Gross = 1000, Pct = 4%
        assert report.is_neutral is True
        assert report.delta_percentage == 4.0

    def test_needs_rebalancing(self, calculator, position_manager):
        """Should detect when rebalancing is needed."""
        position_manager.update_position(Position(
            account_id="acc1",
            market="BTC-USD-PERP",
            side="LONG",
            size=Decimal("2000"),
            entry_price=Decimal("50000"),
        ))

        position_manager.update_position(Position(
            account_id="acc2",
            market="BTC-USD-PERP",
            side="SHORT",
            size=Decimal("1000"),
            entry_price=Decimal("50000"),
        ))

        assert calculator.needs_rebalancing("BTC-USD-PERP") is True

    def test_rebalance_size_calculation(self, calculator, position_manager):
        """Should calculate correct rebalance size."""
        position_manager.update_position(Position(
            account_id="acc1",
            market="BTC-USD-PERP",
            side="LONG",
            size=Decimal("1200"),
            entry_price=Decimal("50000"),
        ))

        position_manager.update_position(Position(
            account_id="acc2",
            market="BTC-USD-PERP",
            side="SHORT",
            size=Decimal("1000"),
            entry_price=Decimal("50000"),
        ))

        # Imbalance is 200, so each side needs to move 100
        rebalance_size = calculator.calculate_rebalance_size("BTC-USD-PERP")
        assert rebalance_size == Decimal("100")

    def test_accounts_tracking(self, calculator, position_manager):
        """Should track which accounts are long vs short."""
        position_manager.update_position(Position(
            account_id="acc1",
            market="BTC-USD-PERP",
            side="LONG",
            size=Decimal("1000"),
            entry_price=Decimal("50000"),
        ))

        position_manager.update_position(Position(
            account_id="acc2",
            market="BTC-USD-PERP",
            side="LONG",
            size=Decimal("500"),
            entry_price=Decimal("50000"),
        ))

        position_manager.update_position(Position(
            account_id="acc3",
            market="BTC-USD-PERP",
            side="SHORT",
            size=Decimal("1500"),
            entry_price=Decimal("50000"),
        ))

        report = calculator.calculate_delta("BTC-USD-PERP")

        assert "acc1" in report.accounts_long
        assert "acc2" in report.accounts_long
        assert "acc3" in report.accounts_short
        assert len(report.accounts_long) == 2
        assert len(report.accounts_short) == 1

    def test_multiple_markets(self, calculator, position_manager):
        """Should handle multiple markets independently."""
        # BTC positions
        position_manager.update_position(Position(
            account_id="acc1",
            market="BTC-USD-PERP",
            side="LONG",
            size=Decimal("1000"),
            entry_price=Decimal("50000"),
        ))

        # ETH positions
        position_manager.update_position(Position(
            account_id="acc1",
            market="ETH-USD-PERP",
            side="SHORT",
            size=Decimal("500"),
            entry_price=Decimal("3000"),
        ))

        btc_report = calculator.calculate_delta("BTC-USD-PERP")
        eth_report = calculator.calculate_delta("ETH-USD-PERP")

        assert btc_report.market == "BTC-USD-PERP"
        assert eth_report.market == "ETH-USD-PERP"
        assert btc_report.total_long == Decimal("1000")
        assert eth_report.total_short == Decimal("500")

    def test_threshold_change(self, calculator, position_manager):
        """Should respect threshold changes."""
        position_manager.update_position(Position(
            account_id="acc1",
            market="BTC-USD-PERP",
            side="LONG",
            size=Decimal("550"),
            entry_price=Decimal("50000"),
        ))

        position_manager.update_position(Position(
            account_id="acc2",
            market="BTC-USD-PERP",
            side="SHORT",
            size=Decimal("450"),
            entry_price=Decimal("50000"),
        ))

        # 10% delta with 5% threshold = not neutral
        assert calculator.calculate_delta("BTC-USD-PERP").is_neutral is False

        # Change threshold to 15%
        calculator.set_threshold(15.0)

        # Now it should be neutral
        assert calculator.calculate_delta("BTC-USD-PERP").is_neutral is True


class TestDeltaReport:
    """Tests for DeltaReport dataclass."""

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        report = DeltaReport(
            market="BTC-USD-PERP",
            total_long=Decimal("1000"),
            total_short=Decimal("1000"),
            net_delta=Decimal("0"),
            delta_percentage=0.0,
            is_neutral=True,
            accounts_long=["acc1"],
            accounts_short=["acc2"],
        )

        d = report.to_dict()

        assert d["market"] == "BTC-USD-PERP"
        assert d["total_long"] == "1000"
        assert d["is_neutral"] is True
