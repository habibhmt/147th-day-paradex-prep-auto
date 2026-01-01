"""Unit tests for Position Manager."""

import pytest
from decimal import Decimal
import time

from src.core.position_manager import PositionManager, Position


class TestPosition:
    """Tests for Position dataclass."""

    def test_create_position(self):
        """Should create position correctly."""
        pos = Position(
            account_id="acc1",
            market="BTC-USD-PERP",
            side="LONG",
            size=Decimal("1000"),
            entry_price=Decimal("50000"),
        )

        assert pos.account_id == "acc1"
        assert pos.market == "BTC-USD-PERP"
        assert pos.side == "LONG"
        assert pos.size == Decimal("1000")
        assert pos.entry_price == Decimal("50000")

    def test_notional_value(self):
        """Should calculate notional value correctly."""
        pos = Position(
            account_id="acc1",
            market="BTC-USD-PERP",
            side="LONG",
            size=Decimal("2"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("51000"),
        )

        assert pos.notional_value == Decimal("102000")  # 2 * 51000

    def test_signed_size_long(self):
        """Long position should have positive signed size."""
        pos = Position(
            account_id="acc1",
            market="BTC-USD-PERP",
            side="LONG",
            size=Decimal("1000"),
            entry_price=Decimal("50000"),
        )

        assert pos.signed_size == Decimal("1000")

    def test_signed_size_short(self):
        """Short position should have negative signed size."""
        pos = Position(
            account_id="acc1",
            market="BTC-USD-PERP",
            side="SHORT",
            size=Decimal("1000"),
            entry_price=Decimal("50000"),
        )

        assert pos.signed_size == Decimal("-1000")

    def test_duration_hours(self):
        """Should calculate duration correctly."""
        pos = Position(
            account_id="acc1",
            market="BTC-USD-PERP",
            side="LONG",
            size=Decimal("1000"),
            entry_price=Decimal("50000"),
            opened_at=time.time() - (10 * 3600),  # 10 hours ago
        )

        assert 9.9 < pos.duration_hours < 10.1

    def test_duration_hours_not_set(self):
        """Duration should be 0 when opened_at not set."""
        pos = Position(
            account_id="acc1",
            market="BTC-USD-PERP",
            side="LONG",
            size=Decimal("1000"),
            entry_price=Decimal("50000"),
            opened_at=0,
        )

        assert pos.duration_hours == 0.0

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        pos = Position(
            account_id="acc1",
            market="BTC-USD-PERP",
            side="LONG",
            size=Decimal("1000"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("51000"),
            leverage=Decimal("10"),
        )

        d = pos.to_dict()

        assert d["account_id"] == "acc1"
        assert d["market"] == "BTC-USD-PERP"
        assert d["side"] == "LONG"
        assert d["size"] == "1000"
        assert d["entry_price"] == "50000"
        assert d["leverage"] == "10"


class TestPositionManager:
    """Tests for PositionManager."""

    @pytest.fixture
    def pm(self):
        """Create fresh position manager."""
        return PositionManager()

    def test_update_position(self, pm):
        """Should add new position."""
        pos = Position(
            account_id="acc1",
            market="BTC-USD-PERP",
            side="LONG",
            size=Decimal("1000"),
            entry_price=Decimal("50000"),
        )

        pm.update_position(pos)

        result = pm.get_position("acc1", "BTC-USD-PERP")
        assert result is not None
        assert result.size == Decimal("1000")

    def test_update_position_overwrites(self, pm):
        """Should overwrite existing position."""
        pos1 = Position(
            account_id="acc1",
            market="BTC-USD-PERP",
            side="LONG",
            size=Decimal("1000"),
            entry_price=Decimal("50000"),
        )
        pos2 = Position(
            account_id="acc1",
            market="BTC-USD-PERP",
            side="LONG",
            size=Decimal("2000"),
            entry_price=Decimal("51000"),
        )

        pm.update_position(pos1)
        pm.update_position(pos2)

        result = pm.get_position("acc1", "BTC-USD-PERP")
        assert result.size == Decimal("2000")

    def test_update_position_zero_size_removes(self, pm):
        """Zero size should remove position."""
        pos = Position(
            account_id="acc1",
            market="BTC-USD-PERP",
            side="LONG",
            size=Decimal("1000"),
            entry_price=Decimal("50000"),
        )
        pm.update_position(pos)

        # Close position
        close_pos = Position(
            account_id="acc1",
            market="BTC-USD-PERP",
            side="LONG",
            size=Decimal("0"),
            entry_price=Decimal("50000"),
        )
        pm.update_position(close_pos)

        assert pm.get_position("acc1", "BTC-USD-PERP") is None

    def test_get_position_nonexistent(self, pm):
        """Should return None for nonexistent position."""
        result = pm.get_position("acc1", "BTC-USD-PERP")
        assert result is None

    def test_get_account_positions(self, pm):
        """Should return all positions for account."""
        pos1 = Position(
            account_id="acc1",
            market="BTC-USD-PERP",
            side="LONG",
            size=Decimal("1000"),
            entry_price=Decimal("50000"),
        )
        pos2 = Position(
            account_id="acc1",
            market="ETH-USD-PERP",
            side="SHORT",
            size=Decimal("500"),
            entry_price=Decimal("3000"),
        )
        pm.update_position(pos1)
        pm.update_position(pos2)

        positions = pm.get_account_positions("acc1")

        assert len(positions) == 2
        markets = [p.market for p in positions]
        assert "BTC-USD-PERP" in markets
        assert "ETH-USD-PERP" in markets

    def test_get_all_positions(self, pm):
        """Should return all positions."""
        pos1 = Position("acc1", "BTC-USD-PERP", "LONG", Decimal("1000"), Decimal("50000"))
        pos2 = Position("acc2", "BTC-USD-PERP", "SHORT", Decimal("1000"), Decimal("50000"))
        pos3 = Position("acc1", "ETH-USD-PERP", "LONG", Decimal("500"), Decimal("3000"))
        pm.update_position(pos1)
        pm.update_position(pos2)
        pm.update_position(pos3)

        all_positions = pm.get_all_positions()
        assert len(all_positions) == 3

    def test_get_all_positions_filtered(self, pm):
        """Should filter by market."""
        pos1 = Position("acc1", "BTC-USD-PERP", "LONG", Decimal("1000"), Decimal("50000"))
        pos2 = Position("acc2", "ETH-USD-PERP", "LONG", Decimal("500"), Decimal("3000"))
        pm.update_position(pos1)
        pm.update_position(pos2)

        btc_positions = pm.get_all_positions("BTC-USD-PERP")

        assert len(btc_positions) == 1
        assert btc_positions[0].market == "BTC-USD-PERP"

    def test_get_market_positions(self, pm):
        """Should return dict of account positions for market."""
        pos1 = Position("acc1", "BTC-USD-PERP", "LONG", Decimal("1000"), Decimal("50000"))
        pos2 = Position("acc2", "BTC-USD-PERP", "SHORT", Decimal("1000"), Decimal("50000"))
        pm.update_position(pos1)
        pm.update_position(pos2)

        market_pos = pm.get_market_positions("BTC-USD-PERP")

        assert "acc1" in market_pos
        assert "acc2" in market_pos
        assert market_pos["acc1"].side == "LONG"
        assert market_pos["acc2"].side == "SHORT"

    def test_get_net_exposure(self, pm):
        """Should calculate net exposure correctly."""
        pos1 = Position("acc1", "BTC-USD-PERP", "LONG", Decimal("1000"), Decimal("50000"))
        pos2 = Position("acc2", "BTC-USD-PERP", "SHORT", Decimal("600"), Decimal("50000"))
        pm.update_position(pos1)
        pm.update_position(pos2)

        net = pm.get_net_exposure("BTC-USD-PERP")
        assert net == Decimal("400")  # 1000 - 600

    def test_get_gross_exposure(self, pm):
        """Should calculate gross exposure correctly."""
        pos1 = Position("acc1", "BTC-USD-PERP", "LONG", Decimal("1000"), Decimal("50000"))
        pos2 = Position("acc2", "BTC-USD-PERP", "SHORT", Decimal("600"), Decimal("50000"))
        pm.update_position(pos1)
        pm.update_position(pos2)

        gross = pm.get_gross_exposure("BTC-USD-PERP")
        assert gross == Decimal("1600")  # 1000 + 600

    def test_get_long_exposure(self, pm):
        """Should calculate total long exposure."""
        pos1 = Position("acc1", "BTC-USD-PERP", "LONG", Decimal("1000"), Decimal("50000"))
        pos2 = Position("acc2", "BTC-USD-PERP", "LONG", Decimal("500"), Decimal("50000"))
        pos3 = Position("acc3", "BTC-USD-PERP", "SHORT", Decimal("600"), Decimal("50000"))
        pm.update_position(pos1)
        pm.update_position(pos2)
        pm.update_position(pos3)

        long_exp = pm.get_long_exposure("BTC-USD-PERP")
        assert long_exp == Decimal("1500")

    def test_get_short_exposure(self, pm):
        """Should calculate total short exposure."""
        pos1 = Position("acc1", "BTC-USD-PERP", "LONG", Decimal("1000"), Decimal("50000"))
        pos2 = Position("acc2", "BTC-USD-PERP", "SHORT", Decimal("500"), Decimal("50000"))
        pos3 = Position("acc3", "BTC-USD-PERP", "SHORT", Decimal("600"), Decimal("50000"))
        pm.update_position(pos1)
        pm.update_position(pos2)
        pm.update_position(pos3)

        short_exp = pm.get_short_exposure("BTC-USD-PERP")
        assert short_exp == Decimal("1100")

    def test_get_accounts_by_side(self, pm):
        """Should return accounts with positions on specified side."""
        pos1 = Position("acc1", "BTC-USD-PERP", "LONG", Decimal("1000"), Decimal("50000"))
        pos2 = Position("acc2", "BTC-USD-PERP", "LONG", Decimal("500"), Decimal("50000"))
        pos3 = Position("acc3", "BTC-USD-PERP", "SHORT", Decimal("600"), Decimal("50000"))
        pm.update_position(pos1)
        pm.update_position(pos2)
        pm.update_position(pos3)

        long_accounts = pm.get_accounts_by_side("BTC-USD-PERP", "LONG")
        short_accounts = pm.get_accounts_by_side("BTC-USD-PERP", "SHORT")

        assert set(long_accounts) == {"acc1", "acc2"}
        assert short_accounts == ["acc3"]

    def test_get_total_pnl(self, pm):
        """Should calculate total PnL."""
        pos1 = Position(
            "acc1", "BTC-USD-PERP", "LONG", Decimal("1000"), Decimal("50000"),
            unrealized_pnl=Decimal("100"), realized_pnl=Decimal("50"),
        )
        pos2 = Position(
            "acc2", "BTC-USD-PERP", "SHORT", Decimal("500"), Decimal("50000"),
            unrealized_pnl=Decimal("-30"), realized_pnl=Decimal("20"),
        )
        pm.update_position(pos1)
        pm.update_position(pos2)

        pnl = pm.get_total_pnl()

        assert pnl["unrealized"] == Decimal("70")  # 100 - 30
        assert pnl["realized"] == Decimal("70")  # 50 + 20
        assert pnl["total"] == Decimal("140")

    def test_get_markets(self, pm):
        """Should return list of all markets."""
        pos1 = Position("acc1", "BTC-USD-PERP", "LONG", Decimal("1000"), Decimal("50000"))
        pos2 = Position("acc2", "ETH-USD-PERP", "LONG", Decimal("500"), Decimal("3000"))
        pos3 = Position("acc3", "BTC-USD-PERP", "SHORT", Decimal("600"), Decimal("50000"))
        pm.update_position(pos1)
        pm.update_position(pos2)
        pm.update_position(pos3)

        markets = pm.get_markets()

        assert set(markets) == {"BTC-USD-PERP", "ETH-USD-PERP"}

    def test_has_position(self, pm):
        """Should check if position exists."""
        pos = Position("acc1", "BTC-USD-PERP", "LONG", Decimal("1000"), Decimal("50000"))
        pm.update_position(pos)

        assert pm.has_position("acc1", "BTC-USD-PERP") is True
        assert pm.has_position("acc1", "ETH-USD-PERP") is False
        assert pm.has_position("acc2", "BTC-USD-PERP") is False

    def test_clear_account(self, pm):
        """Should clear all positions for account."""
        pos1 = Position("acc1", "BTC-USD-PERP", "LONG", Decimal("1000"), Decimal("50000"))
        pos2 = Position("acc1", "ETH-USD-PERP", "LONG", Decimal("500"), Decimal("3000"))
        pos3 = Position("acc2", "BTC-USD-PERP", "SHORT", Decimal("600"), Decimal("50000"))
        pm.update_position(pos1)
        pm.update_position(pos2)
        pm.update_position(pos3)

        pm.clear_account("acc1")

        assert pm.get_account_positions("acc1") == []
        assert len(pm.get_account_positions("acc2")) == 1

    def test_clear_all(self, pm):
        """Should clear all positions."""
        pos1 = Position("acc1", "BTC-USD-PERP", "LONG", Decimal("1000"), Decimal("50000"))
        pos2 = Position("acc2", "BTC-USD-PERP", "SHORT", Decimal("600"), Decimal("50000"))
        pm.update_position(pos1)
        pm.update_position(pos2)

        pm.clear_all()

        assert pm.total_positions == 0

    def test_total_positions(self, pm):
        """Should count total positions."""
        pos1 = Position("acc1", "BTC-USD-PERP", "LONG", Decimal("1000"), Decimal("50000"))
        pos2 = Position("acc1", "ETH-USD-PERP", "LONG", Decimal("500"), Decimal("3000"))
        pos3 = Position("acc2", "BTC-USD-PERP", "SHORT", Decimal("600"), Decimal("50000"))
        pm.update_position(pos1)
        pm.update_position(pos2)
        pm.update_position(pos3)

        assert pm.total_positions == 3

    def test_accounts_with_positions(self, pm):
        """Should list accounts with open positions."""
        pos1 = Position("acc1", "BTC-USD-PERP", "LONG", Decimal("1000"), Decimal("50000"))
        pos2 = Position("acc2", "BTC-USD-PERP", "SHORT", Decimal("600"), Decimal("50000"))
        pm.update_position(pos1)
        pm.update_position(pos2)

        accounts = pm.accounts_with_positions

        assert set(accounts) == {"acc1", "acc2"}

    def test_update_from_api(self, pm):
        """Should update positions from API response."""
        api_positions = [
            {
                "market": "BTC-USD-PERP",
                "side": "LONG",
                "size": "1.5",
                "avg_entry_price": "50000",
                "mark_price": "51000",
                "unrealized_pnl": "1500",
                "leverage": "10",
            },
            {
                "market": "ETH-USD-PERP",
                "side": "SHORT",
                "size": "10",
                "avg_entry_price": "3000",
                "mark_price": "2950",
                "unrealized_pnl": "500",
                "leverage": "5",
            },
        ]

        pm.update_from_api("acc1", api_positions)

        positions = pm.get_account_positions("acc1")
        assert len(positions) == 2

        btc_pos = pm.get_position("acc1", "BTC-USD-PERP")
        assert btc_pos.side == "LONG"
        assert btc_pos.size == Decimal("1.5")
        assert btc_pos.entry_price == Decimal("50000")

    def test_summary(self, pm):
        """Should return summary dictionary."""
        pos1 = Position(
            "acc1", "BTC-USD-PERP", "LONG", Decimal("1000"), Decimal("50000"),
            unrealized_pnl=Decimal("100"),
        )
        pos2 = Position(
            "acc2", "BTC-USD-PERP", "SHORT", Decimal("1000"), Decimal("50000"),
            unrealized_pnl=Decimal("-50"),
        )
        pm.update_position(pos1)
        pm.update_position(pos2)

        summary = pm.summary()

        assert summary["total_positions"] == 2
        assert summary["accounts_with_positions"] == 2
        assert "BTC-USD-PERP" in summary["markets"]
        assert summary["markets"]["BTC-USD-PERP"]["net_exposure"] == "0"
        assert summary["pnl"]["unrealized"] == "50"
