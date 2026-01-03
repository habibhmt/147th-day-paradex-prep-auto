"""
Tests for Market Making Strategy Module
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from src.strategies.market_making import (
    MarketMakingType,
    QuoteStatus,
    InventoryState,
    SpreadState,
    Quote,
    QuotePair,
    MarketMakingConfig,
    InventoryMetrics,
    MarketMakingMetrics,
    MarketData,
    SpreadCalculator,
    InventoryManager,
    QuoteManager,
    MarketMakingStrategy,
    MultiLevelMarketMaker,
    GridMarketMaker,
)


class TestMarketMakingType:
    """Tests for MarketMakingType enum."""

    def test_all_types_defined(self):
        """Test all market making types are defined."""
        assert MarketMakingType.BASIC.value == "basic"
        assert MarketMakingType.INVENTORY.value == "inventory"
        assert MarketMakingType.AVELLANEDA_STOIKOV.value == "avellaneda_stoikov"
        assert MarketMakingType.ADAPTIVE.value == "adaptive"
        assert MarketMakingType.INFORMATION.value == "information"


class TestQuoteStatus:
    """Tests for QuoteStatus enum."""

    def test_all_statuses_defined(self):
        """Test all quote statuses are defined."""
        assert QuoteStatus.ACTIVE.value == "active"
        assert QuoteStatus.FILLED.value == "filled"
        assert QuoteStatus.CANCELLED.value == "cancelled"
        assert QuoteStatus.EXPIRED.value == "expired"
        assert QuoteStatus.PARTIAL.value == "partial"


class TestInventoryState:
    """Tests for InventoryState enum."""

    def test_all_states_defined(self):
        """Test all inventory states are defined."""
        assert InventoryState.LONG.value == "long"
        assert InventoryState.SHORT.value == "short"
        assert InventoryState.NEUTRAL.value == "neutral"
        assert InventoryState.EXTREME_LONG.value == "extreme_long"
        assert InventoryState.EXTREME_SHORT.value == "extreme_short"


class TestSpreadState:
    """Tests for SpreadState enum."""

    def test_all_states_defined(self):
        """Test all spread states are defined."""
        assert SpreadState.TIGHT.value == "tight"
        assert SpreadState.NORMAL.value == "normal"
        assert SpreadState.WIDE.value == "wide"
        assert SpreadState.VERY_WIDE.value == "very_wide"


class TestQuote:
    """Tests for Quote dataclass."""

    def test_creation(self):
        """Test quote creation."""
        quote = Quote(
            quote_id="q_1",
            symbol="ETH-USD-PERP",
            side="bid",
            price=Decimal("2000.00"),
            size=Decimal("1.0"),
            timestamp=datetime.now(),
        )
        assert quote.quote_id == "q_1"
        assert quote.side == "bid"
        assert quote.status == QuoteStatus.ACTIVE

    def test_remaining_size(self):
        """Test remaining size calculation."""
        quote = Quote(
            quote_id="q_1",
            symbol="ETH-USD-PERP",
            side="ask",
            price=Decimal("2000.00"),
            size=Decimal("1.0"),
            timestamp=datetime.now(),
            filled_size=Decimal("0.3"),
        )
        assert quote.remaining_size == Decimal("0.7")

    def test_to_dict(self):
        """Test quote to_dict method."""
        quote = Quote(
            quote_id="q_1",
            symbol="ETH-USD-PERP",
            side="bid",
            price=Decimal("2000.00"),
            size=Decimal("1.0"),
            timestamp=datetime.now(),
        )
        result = quote.to_dict()
        assert result["quote_id"] == "q_1"
        assert result["side"] == "bid"
        assert result["price"] == "2000.00"


class TestQuotePair:
    """Tests for QuotePair dataclass."""

    def test_creation(self):
        """Test quote pair creation."""
        now = datetime.now()
        bid = Quote(
            quote_id="q_bid",
            symbol="ETH-USD-PERP",
            side="bid",
            price=Decimal("1999.00"),
            size=Decimal("1.0"),
            timestamp=now,
        )
        ask = Quote(
            quote_id="q_ask",
            symbol="ETH-USD-PERP",
            side="ask",
            price=Decimal("2001.00"),
            size=Decimal("1.0"),
            timestamp=now,
        )
        pair = QuotePair(
            bid=bid,
            ask=ask,
            spread=Decimal("2.00"),
            mid_price=Decimal("2000.00"),
            timestamp=now,
        )
        assert pair.spread == Decimal("2.00")
        assert pair.mid_price == Decimal("2000.00")

    def test_to_dict(self):
        """Test quote pair to_dict method."""
        now = datetime.now()
        bid = Quote(
            quote_id="q_bid",
            symbol="ETH-USD-PERP",
            side="bid",
            price=Decimal("1999.00"),
            size=Decimal("1.0"),
            timestamp=now,
        )
        ask = Quote(
            quote_id="q_ask",
            symbol="ETH-USD-PERP",
            side="ask",
            price=Decimal("2001.00"),
            size=Decimal("1.0"),
            timestamp=now,
        )
        pair = QuotePair(
            bid=bid,
            ask=ask,
            spread=Decimal("2.00"),
            mid_price=Decimal("2000.00"),
            timestamp=now,
        )
        result = pair.to_dict()
        assert "bid" in result
        assert "ask" in result
        assert result["spread"] == "2.00"


class TestMarketMakingConfig:
    """Tests for MarketMakingConfig dataclass."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = MarketMakingConfig(
            strategy_type=MarketMakingType.INVENTORY,
            base_spread=Decimal("0.002"),
            quote_size=Decimal("0.5"),
        )
        assert config.strategy_type == MarketMakingType.INVENTORY
        assert config.base_spread == Decimal("0.002")

    def test_invalid_base_spread(self):
        """Test invalid base spread."""
        with pytest.raises(ValueError):
            MarketMakingConfig(base_spread=Decimal("0"))

    def test_invalid_min_spread(self):
        """Test invalid min spread."""
        with pytest.raises(ValueError):
            MarketMakingConfig(min_spread=Decimal("-0.001"))

    def test_min_spread_exceeds_base(self):
        """Test min spread exceeding base."""
        with pytest.raises(ValueError):
            MarketMakingConfig(
                base_spread=Decimal("0.001"),
                min_spread=Decimal("0.002"),
            )

    def test_max_spread_less_than_base(self):
        """Test max spread less than base."""
        with pytest.raises(ValueError):
            MarketMakingConfig(
                base_spread=Decimal("0.01"),
                max_spread=Decimal("0.005"),
            )

    def test_invalid_quote_size(self):
        """Test invalid quote size."""
        with pytest.raises(ValueError):
            MarketMakingConfig(quote_size=Decimal("0"))

    def test_invalid_max_position(self):
        """Test invalid max position."""
        with pytest.raises(ValueError):
            MarketMakingConfig(max_position=Decimal("-1"))

    def test_to_dict(self):
        """Test config to_dict method."""
        config = MarketMakingConfig()
        result = config.to_dict()
        assert "strategy_type" in result
        assert "base_spread" in result
        assert "inventory_skew" in result


class TestInventoryMetrics:
    """Tests for InventoryMetrics dataclass."""

    def test_creation(self):
        """Test metrics creation."""
        metrics = InventoryMetrics()
        assert metrics.position == Decimal("0")
        assert metrics.realized_pnl == Decimal("0")

    def test_net_volume(self):
        """Test net volume calculation."""
        metrics = InventoryMetrics(
            buy_volume=Decimal("10"),
            sell_volume=Decimal("7"),
        )
        assert metrics.net_volume == Decimal("3")

    def test_total_pnl(self):
        """Test total PnL calculation."""
        metrics = InventoryMetrics(
            unrealized_pnl=Decimal("50"),
            realized_pnl=Decimal("100"),
        )
        assert metrics.total_pnl == Decimal("150")

    def test_to_dict(self):
        """Test metrics to_dict method."""
        metrics = InventoryMetrics(position=Decimal("5"))
        result = metrics.to_dict()
        assert result["position"] == "5"


class TestMarketMakingMetrics:
    """Tests for MarketMakingMetrics dataclass."""

    def test_creation(self):
        """Test metrics creation."""
        metrics = MarketMakingMetrics()
        assert metrics.total_quotes == 0
        assert metrics.fill_rate == 0.0

    def test_to_dict(self):
        """Test metrics to_dict method."""
        metrics = MarketMakingMetrics(total_quotes=100, filled_quotes=70)
        result = metrics.to_dict()
        assert result["total_quotes"] == 100
        assert result["filled_quotes"] == 70


class TestMarketData:
    """Tests for MarketData dataclass."""

    def test_creation(self):
        """Test market data creation."""
        market = MarketData(
            timestamp=datetime.now(),
            bid=Decimal("1999.00"),
            ask=Decimal("2001.00"),
            last_price=Decimal("2000.00"),
        )
        assert market.bid == Decimal("1999.00")

    def test_mid_price(self):
        """Test mid price calculation."""
        market = MarketData(
            timestamp=datetime.now(),
            bid=Decimal("1999.00"),
            ask=Decimal("2001.00"),
            last_price=Decimal("2000.00"),
        )
        assert market.mid_price == Decimal("2000.00")

    def test_spread(self):
        """Test spread calculation."""
        market = MarketData(
            timestamp=datetime.now(),
            bid=Decimal("1999.00"),
            ask=Decimal("2001.00"),
            last_price=Decimal("2000.00"),
        )
        assert market.spread == Decimal("2.00")

    def test_spread_pct(self):
        """Test spread percentage calculation."""
        market = MarketData(
            timestamp=datetime.now(),
            bid=Decimal("1999.00"),
            ask=Decimal("2001.00"),
            last_price=Decimal("2000.00"),
        )
        assert market.spread_pct == pytest.approx(0.001, rel=0.01)


class TestSpreadCalculator:
    """Tests for SpreadCalculator class."""

    def test_calculate_base_spread(self):
        """Test base spread calculation."""
        config = MarketMakingConfig(base_spread=Decimal("0.002"))
        calc = SpreadCalculator(config)
        market = MarketData(
            timestamp=datetime.now(),
            bid=Decimal("1999.00"),
            ask=Decimal("2001.00"),
            last_price=Decimal("2000.00"),
        )
        spread = calc.calculate_base_spread(market)
        assert spread == Decimal("0.002")

    def test_calculate_volatility_spread(self):
        """Test volatility spread calculation."""
        config = MarketMakingConfig(
            base_spread=Decimal("0.001"),
            volatility_multiplier=2.0,
        )
        calc = SpreadCalculator(config)
        market = MarketData(
            timestamp=datetime.now(),
            bid=Decimal("1999.00"),
            ask=Decimal("2001.00"),
            last_price=Decimal("2000.00"),
            volatility=0.02,
        )
        spread = calc.calculate_volatility_spread(market)
        assert spread > config.base_spread

    def test_calculate_inventory_spread_long(self):
        """Test inventory spread with long position."""
        config = MarketMakingConfig(
            base_spread=Decimal("0.001"),
            max_position=Decimal("10.0"),
            inventory_skew=0.5,
        )
        calc = SpreadCalculator(config)
        market = MarketData(
            timestamp=datetime.now(),
            bid=Decimal("1999.00"),
            ask=Decimal("2001.00"),
            last_price=Decimal("2000.00"),
        )
        inventory = InventoryMetrics(position=Decimal("5"))
        bid_spread, ask_spread = calc.calculate_inventory_spread(market, inventory)
        # With long position, bid spread should be wider
        assert bid_spread >= ask_spread

    def test_calculate_inventory_spread_short(self):
        """Test inventory spread with short position."""
        config = MarketMakingConfig(
            base_spread=Decimal("0.001"),
            max_position=Decimal("10.0"),
            inventory_skew=0.5,
        )
        calc = SpreadCalculator(config)
        market = MarketData(
            timestamp=datetime.now(),
            bid=Decimal("1999.00"),
            ask=Decimal("2001.00"),
            last_price=Decimal("2000.00"),
        )
        inventory = InventoryMetrics(position=Decimal("-5"))
        bid_spread, ask_spread = calc.calculate_inventory_spread(market, inventory)
        # With short position, ask spread should be wider
        assert ask_spread >= bid_spread

    def test_calculate_avellaneda_stoikov(self):
        """Test Avellaneda-Stoikov spread calculation."""
        config = MarketMakingConfig(
            gamma=0.1,
            kappa=1.5,
        )
        calc = SpreadCalculator(config)
        market = MarketData(
            timestamp=datetime.now(),
            bid=Decimal("1999.00"),
            ask=Decimal("2001.00"),
            last_price=Decimal("2000.00"),
            volatility=0.02,
        )
        inventory = InventoryMetrics(position=Decimal("0"))
        bid_spread, ask_spread = calc.calculate_avellaneda_stoikov(market, inventory, 1.0)
        assert bid_spread > 0
        assert ask_spread > 0


class TestInventoryManager:
    """Tests for InventoryManager class."""

    def test_creation(self):
        """Test inventory manager creation."""
        manager = InventoryManager(Decimal("10.0"))
        assert manager.max_position == Decimal("10.0")
        assert manager.metrics.position == Decimal("0")

    def test_record_buy(self):
        """Test recording a buy fill."""
        manager = InventoryManager(Decimal("10.0"))
        manager.record_fill("bid", Decimal("2000.00"), Decimal("1.0"))
        assert manager.metrics.position == Decimal("1.0")
        assert manager.metrics.avg_entry_price == Decimal("2000.00")
        assert manager.metrics.total_buys == 1
        assert manager.metrics.buy_volume == Decimal("1.0")

    def test_record_sell(self):
        """Test recording a sell fill."""
        manager = InventoryManager(Decimal("10.0"))
        manager.record_fill("bid", Decimal("2000.00"), Decimal("1.0"))
        manager.record_fill("ask", Decimal("2010.00"), Decimal("1.0"))
        assert manager.metrics.position == Decimal("0")
        assert manager.metrics.total_sells == 1
        assert manager.metrics.sell_volume == Decimal("1.0")
        assert manager.metrics.realized_pnl == Decimal("10.00")

    def test_update_unrealized_pnl(self):
        """Test unrealized PnL update."""
        manager = InventoryManager(Decimal("10.0"))
        manager.record_fill("bid", Decimal("2000.00"), Decimal("1.0"))
        manager.update_unrealized_pnl(Decimal("2050.00"))
        assert manager.metrics.unrealized_pnl == Decimal("50.00")

    def test_get_state_neutral(self):
        """Test neutral inventory state."""
        manager = InventoryManager(Decimal("10.0"))
        assert manager.get_state() == InventoryState.NEUTRAL

    def test_get_state_long(self):
        """Test long inventory state."""
        manager = InventoryManager(Decimal("10.0"))
        manager.metrics.position = Decimal("5.0")
        assert manager.get_state() == InventoryState.LONG

    def test_get_state_extreme_long(self):
        """Test extreme long inventory state."""
        manager = InventoryManager(Decimal("10.0"))
        manager.metrics.position = Decimal("9.0")
        assert manager.get_state() == InventoryState.EXTREME_LONG

    def test_get_state_short(self):
        """Test short inventory state."""
        manager = InventoryManager(Decimal("10.0"))
        manager.metrics.position = Decimal("-5.0")
        assert manager.get_state() == InventoryState.SHORT

    def test_get_state_extreme_short(self):
        """Test extreme short inventory state."""
        manager = InventoryManager(Decimal("10.0"))
        manager.metrics.position = Decimal("-9.0")
        assert manager.get_state() == InventoryState.EXTREME_SHORT

    def test_can_buy(self):
        """Test can buy check."""
        manager = InventoryManager(Decimal("10.0"))
        assert manager.can_buy(Decimal("5.0"))
        manager.metrics.position = Decimal("8.0")
        assert not manager.can_buy(Decimal("5.0"))

    def test_can_sell(self):
        """Test can sell check."""
        manager = InventoryManager(Decimal("10.0"))
        assert manager.can_sell(Decimal("5.0"))
        manager.metrics.position = Decimal("-8.0")
        assert not manager.can_sell(Decimal("5.0"))

    def test_get_suggested_size_neutral(self):
        """Test suggested size in neutral state."""
        manager = InventoryManager(Decimal("10.0"))
        bid_size = manager.get_suggested_size("bid")
        ask_size = manager.get_suggested_size("ask")
        assert bid_size == Decimal("1.0")  # 10% of max
        assert ask_size == Decimal("1.0")

    def test_get_suggested_size_long(self):
        """Test suggested size when long."""
        manager = InventoryManager(Decimal("10.0"))
        manager.metrics.position = Decimal("5.0")
        bid_size = manager.get_suggested_size("bid")
        ask_size = manager.get_suggested_size("ask")
        # Should sell more, buy less when long
        assert ask_size > bid_size

    def test_get_suggested_size_extreme_long(self):
        """Test suggested size when extreme long."""
        manager = InventoryManager(Decimal("10.0"))
        manager.metrics.position = Decimal("9.0")
        bid_size = manager.get_suggested_size("bid")
        assert bid_size == Decimal("0")  # Should not buy more


class TestQuoteManager:
    """Tests for QuoteManager class."""

    def test_creation(self):
        """Test quote manager creation."""
        manager = QuoteManager("ETH-USD-PERP")
        assert manager.symbol == "ETH-USD-PERP"
        assert len(manager.active_quotes) == 0

    def test_create_quote(self):
        """Test quote creation."""
        manager = QuoteManager("ETH-USD-PERP")
        quote = manager.create_quote("bid", Decimal("2000.00"), Decimal("1.0"))
        assert quote.side == "bid"
        assert quote.price == Decimal("2000.00")
        assert quote.quote_id in manager.active_quotes

    def test_create_quote_with_expiry(self):
        """Test quote creation with expiry."""
        manager = QuoteManager("ETH-USD-PERP")
        quote = manager.create_quote("ask", Decimal("2000.00"), Decimal("1.0"), 30)
        assert quote.expiry is not None

    def test_fill_quote_full(self):
        """Test full quote fill."""
        manager = QuoteManager("ETH-USD-PERP")
        quote = manager.create_quote("bid", Decimal("2000.00"), Decimal("1.0"))
        filled = manager.fill_quote(quote.quote_id, Decimal("1.0"))
        assert filled.status == QuoteStatus.FILLED
        assert quote.quote_id not in manager.active_quotes

    def test_fill_quote_partial(self):
        """Test partial quote fill."""
        manager = QuoteManager("ETH-USD-PERP")
        quote = manager.create_quote("bid", Decimal("2000.00"), Decimal("1.0"))
        filled = manager.fill_quote(quote.quote_id, Decimal("0.5"))
        assert filled.status == QuoteStatus.PARTIAL
        assert quote.quote_id in manager.active_quotes
        assert filled.remaining_size == Decimal("0.5")

    def test_fill_quote_not_found(self):
        """Test filling non-existent quote."""
        manager = QuoteManager("ETH-USD-PERP")
        result = manager.fill_quote("invalid", Decimal("1.0"))
        assert result is None

    def test_cancel_quote(self):
        """Test quote cancellation."""
        manager = QuoteManager("ETH-USD-PERP")
        quote = manager.create_quote("bid", Decimal("2000.00"), Decimal("1.0"))
        cancelled = manager.cancel_quote(quote.quote_id)
        assert cancelled.status == QuoteStatus.CANCELLED
        assert quote.quote_id not in manager.active_quotes

    def test_cancel_quote_not_found(self):
        """Test cancelling non-existent quote."""
        manager = QuoteManager("ETH-USD-PERP")
        result = manager.cancel_quote("invalid")
        assert result is None

    def test_cancel_all(self):
        """Test cancelling all quotes."""
        manager = QuoteManager("ETH-USD-PERP")
        manager.create_quote("bid", Decimal("1999.00"), Decimal("1.0"))
        manager.create_quote("ask", Decimal("2001.00"), Decimal("1.0"))
        cancelled = manager.cancel_all()
        assert len(cancelled) == 2
        assert len(manager.active_quotes) == 0

    def test_cancel_all_by_side(self):
        """Test cancelling quotes by side."""
        manager = QuoteManager("ETH-USD-PERP")
        manager.create_quote("bid", Decimal("1999.00"), Decimal("1.0"))
        manager.create_quote("ask", Decimal("2001.00"), Decimal("1.0"))
        cancelled = manager.cancel_all("bid")
        assert len(cancelled) == 1
        assert len(manager.active_quotes) == 1

    def test_get_active_quotes(self):
        """Test getting active quotes."""
        manager = QuoteManager("ETH-USD-PERP")
        manager.create_quote("bid", Decimal("1999.00"), Decimal("1.0"))
        manager.create_quote("ask", Decimal("2001.00"), Decimal("1.0"))
        all_quotes = manager.get_active_quotes()
        assert len(all_quotes) == 2
        bids = manager.get_active_quotes("bid")
        assert len(bids) == 1

    def test_get_best_bid(self):
        """Test getting best bid."""
        manager = QuoteManager("ETH-USD-PERP")
        manager.create_quote("bid", Decimal("1998.00"), Decimal("1.0"))
        manager.create_quote("bid", Decimal("1999.00"), Decimal("1.0"))
        best = manager.get_best_bid()
        assert best.price == Decimal("1999.00")

    def test_get_best_ask(self):
        """Test getting best ask."""
        manager = QuoteManager("ETH-USD-PERP")
        manager.create_quote("ask", Decimal("2001.00"), Decimal("1.0"))
        manager.create_quote("ask", Decimal("2002.00"), Decimal("1.0"))
        best = manager.get_best_ask()
        assert best.price == Decimal("2001.00")

    def test_get_best_bid_empty(self):
        """Test getting best bid when empty."""
        manager = QuoteManager("ETH-USD-PERP")
        assert manager.get_best_bid() is None

    def test_get_best_ask_empty(self):
        """Test getting best ask when empty."""
        manager = QuoteManager("ETH-USD-PERP")
        assert manager.get_best_ask() is None


class TestMarketMakingStrategy:
    """Tests for MarketMakingStrategy class."""

    def test_creation(self):
        """Test strategy creation."""
        strategy = MarketMakingStrategy("ETH-USD-PERP")
        assert strategy.symbol == "ETH-USD-PERP"
        assert strategy.config.strategy_type == MarketMakingType.INVENTORY

    def test_creation_with_config(self):
        """Test strategy creation with config."""
        config = MarketMakingConfig(strategy_type=MarketMakingType.BASIC)
        strategy = MarketMakingStrategy("ETH-USD-PERP", config)
        assert strategy.config.strategy_type == MarketMakingType.BASIC

    def test_on_market_data_generates_quotes(self):
        """Test market data generates quotes."""
        config = MarketMakingConfig(quote_refresh_seconds=0)
        strategy = MarketMakingStrategy("ETH-USD-PERP", config)
        market = MarketData(
            timestamp=datetime.now(),
            bid=Decimal("1999.00"),
            ask=Decimal("2001.00"),
            last_price=Decimal("2000.00"),
        )
        result = strategy.on_market_data(market)
        assert result is not None
        assert result.bid is not None
        assert result.ask is not None

    def test_on_market_data_respects_refresh(self):
        """Test market data respects refresh interval."""
        config = MarketMakingConfig(quote_refresh_seconds=60)
        strategy = MarketMakingStrategy("ETH-USD-PERP", config)
        market = MarketData(
            timestamp=datetime.now(),
            bid=Decimal("1999.00"),
            ask=Decimal("2001.00"),
            last_price=Decimal("2000.00"),
        )
        # First call should generate
        result1 = strategy.on_market_data(market)
        assert result1 is not None
        # Second call should not (refresh interval not passed)
        result2 = strategy.on_market_data(market)
        assert result2 is None

    def test_on_fill(self):
        """Test fill processing."""
        config = MarketMakingConfig(quote_refresh_seconds=0)
        strategy = MarketMakingStrategy("ETH-USD-PERP", config)
        market = MarketData(
            timestamp=datetime.now(),
            bid=Decimal("1999.00"),
            ask=Decimal("2001.00"),
            last_price=Decimal("2000.00"),
        )
        quotes = strategy.on_market_data(market)
        result = strategy.on_fill(
            quotes.bid.quote_id,
            Decimal("1999.00"),
            Decimal("0.1"),
        )
        assert "quote_id" in result
        assert strategy.inventory_manager.metrics.position == Decimal("0.1")

    def test_on_fill_not_found(self):
        """Test fill with invalid quote."""
        strategy = MarketMakingStrategy("ETH-USD-PERP")
        result = strategy.on_fill("invalid", Decimal("2000.00"), Decimal("1.0"))
        assert "error" in result

    def test_get_quotes(self):
        """Test getting current quotes."""
        config = MarketMakingConfig(quote_refresh_seconds=0)
        strategy = MarketMakingStrategy("ETH-USD-PERP", config)
        market = MarketData(
            timestamp=datetime.now(),
            bid=Decimal("1999.00"),
            ask=Decimal("2001.00"),
            last_price=Decimal("2000.00"),
        )
        strategy.on_market_data(market)
        quotes = strategy.get_quotes()
        assert quotes["best_bid"] is not None
        assert quotes["best_ask"] is not None
        assert quotes["active_bids"] > 0
        assert quotes["active_asks"] > 0

    def test_get_inventory(self):
        """Test getting inventory."""
        strategy = MarketMakingStrategy("ETH-USD-PERP")
        inventory = strategy.get_inventory()
        assert "metrics" in inventory
        assert "state" in inventory
        assert "can_buy" in inventory
        assert "can_sell" in inventory

    def test_get_status(self):
        """Test getting status."""
        strategy = MarketMakingStrategy("ETH-USD-PERP")
        status = strategy.get_status()
        assert status["symbol"] == "ETH-USD-PERP"
        assert "quotes" in status
        assert "inventory" in status
        assert "metrics" in status
        assert "config" in status

    def test_cancel_all_quotes(self):
        """Test cancelling all quotes."""
        config = MarketMakingConfig(quote_refresh_seconds=0)
        strategy = MarketMakingStrategy("ETH-USD-PERP", config)
        market = MarketData(
            timestamp=datetime.now(),
            bid=Decimal("1999.00"),
            ask=Decimal("2001.00"),
            last_price=Decimal("2000.00"),
        )
        strategy.on_market_data(market)
        cancelled = strategy.cancel_all_quotes()
        assert cancelled == 2
        assert strategy.metrics.cancelled_quotes == 2


class TestMarketMakingStrategyTypes:
    """Tests for different market making strategy types."""

    def test_basic_strategy(self):
        """Test basic market making strategy."""
        config = MarketMakingConfig(
            strategy_type=MarketMakingType.BASIC,
            quote_refresh_seconds=0,
        )
        strategy = MarketMakingStrategy("ETH-USD-PERP", config)
        market = MarketData(
            timestamp=datetime.now(),
            bid=Decimal("1999.00"),
            ask=Decimal("2001.00"),
            last_price=Decimal("2000.00"),
        )
        quotes = strategy.on_market_data(market)
        assert quotes is not None

    def test_adaptive_strategy(self):
        """Test adaptive market making strategy."""
        config = MarketMakingConfig(
            strategy_type=MarketMakingType.ADAPTIVE,
            quote_refresh_seconds=0,
        )
        strategy = MarketMakingStrategy("ETH-USD-PERP", config)
        market = MarketData(
            timestamp=datetime.now(),
            bid=Decimal("1999.00"),
            ask=Decimal("2001.00"),
            last_price=Decimal("2000.00"),
            volatility=0.02,
        )
        quotes = strategy.on_market_data(market)
        assert quotes is not None

    def test_avellaneda_stoikov_strategy(self):
        """Test Avellaneda-Stoikov market making strategy."""
        config = MarketMakingConfig(
            strategy_type=MarketMakingType.AVELLANEDA_STOIKOV,
            quote_refresh_seconds=0,
        )
        strategy = MarketMakingStrategy("ETH-USD-PERP", config)
        market = MarketData(
            timestamp=datetime.now(),
            bid=Decimal("1999.00"),
            ask=Decimal("2001.00"),
            last_price=Decimal("2000.00"),
            volatility=0.02,
        )
        quotes = strategy.on_market_data(market)
        assert quotes is not None

    def test_information_strategy(self):
        """Test information-based market making strategy."""
        config = MarketMakingConfig(
            strategy_type=MarketMakingType.INFORMATION,
            quote_refresh_seconds=0,
        )
        strategy = MarketMakingStrategy("ETH-USD-PERP", config)
        market = MarketData(
            timestamp=datetime.now(),
            bid=Decimal("1999.00"),
            ask=Decimal("2001.00"),
            last_price=Decimal("2000.00"),
        )
        quotes = strategy.on_market_data(market)
        assert quotes is not None


class TestMultiLevelMarketMaker:
    """Tests for MultiLevelMarketMaker class."""

    def test_creation(self):
        """Test multi-level market maker creation."""
        mm = MultiLevelMarketMaker("ETH-USD-PERP")
        assert mm.symbol == "ETH-USD-PERP"
        assert mm.levels > 0

    def test_generate_ladder(self):
        """Test generating quote ladder."""
        config = MarketMakingConfig(max_quotes_per_side=3)
        mm = MultiLevelMarketMaker("ETH-USD-PERP", config)
        market = MarketData(
            timestamp=datetime.now(),
            bid=Decimal("1999.00"),
            ask=Decimal("2001.00"),
            last_price=Decimal("2000.00"),
        )
        ladder = mm.generate_ladder(market)
        assert len(ladder) == 3
        # Each level should have wider spread
        assert ladder[0].spread < ladder[1].spread
        assert ladder[1].spread < ladder[2].spread

    def test_get_status(self):
        """Test getting multi-level status."""
        mm = MultiLevelMarketMaker("ETH-USD-PERP")
        status = mm.get_status()
        assert "levels" in status
        assert "total_active_quotes" in status


class TestGridMarketMaker:
    """Tests for GridMarketMaker class."""

    def test_creation(self):
        """Test grid market maker creation."""
        mm = GridMarketMaker("ETH-USD-PERP", grid_size=5)
        assert mm.symbol == "ETH-USD-PERP"
        assert mm.grid_size == 5

    def test_initialize_grid(self):
        """Test initializing grid."""
        mm = GridMarketMaker("ETH-USD-PERP", grid_size=3)
        quotes = mm.initialize_grid(Decimal("2000.00"))
        assert len(quotes) == 6  # 3 bids + 3 asks
        bids = mm.quote_manager.get_active_quotes("bid")
        asks = mm.quote_manager.get_active_quotes("ask")
        assert len(bids) == 3
        assert len(asks) == 3

    def test_on_fill_creates_opposite(self):
        """Test fill creates opposite order."""
        mm = GridMarketMaker("ETH-USD-PERP", grid_size=3)
        quotes = mm.initialize_grid(Decimal("2000.00"))
        bid = mm.quote_manager.get_best_bid()
        new_quote = mm.on_fill(bid.quote_id, bid.price, bid.size)
        assert new_quote is not None
        assert new_quote.side == "ask"

    def test_on_fill_not_found(self):
        """Test fill with invalid quote."""
        mm = GridMarketMaker("ETH-USD-PERP")
        result = mm.on_fill("invalid", Decimal("2000.00"), Decimal("1.0"))
        assert result is None

    def test_get_grid_status(self):
        """Test getting grid status."""
        mm = GridMarketMaker("ETH-USD-PERP", grid_size=5)
        mm.initialize_grid(Decimal("2000.00"))
        status = mm.get_grid_status()
        assert status["grid_size"] == 5
        assert status["active_bids"] == 5
        assert status["active_asks"] == 5


class TestMarketMakingIntegration:
    """Integration tests for market making."""

    def test_full_market_making_cycle(self):
        """Test full market making cycle."""
        config = MarketMakingConfig(
            strategy_type=MarketMakingType.INVENTORY,
            quote_refresh_seconds=0,
            base_spread=Decimal("0.001"),
        )
        strategy = MarketMakingStrategy("ETH-USD-PERP", config)
        # Generate initial quotes
        market = MarketData(
            timestamp=datetime.now(),
            bid=Decimal("1999.00"),
            ask=Decimal("2001.00"),
            last_price=Decimal("2000.00"),
        )
        quotes = strategy.on_market_data(market)
        assert quotes is not None
        # Fill bid
        strategy.on_fill(quotes.bid.quote_id, quotes.bid.price, Decimal("0.1"))
        assert strategy.inventory_manager.metrics.position == Decimal("0.1")
        # Fill ask
        strategy.on_fill(quotes.ask.quote_id, quotes.ask.price, Decimal("0.1"))
        assert strategy.inventory_manager.metrics.position == Decimal("0")
        # Should have realized profit from spread
        assert strategy.inventory_manager.metrics.realized_pnl > 0

    def test_inventory_skew_effect(self):
        """Test inventory skew affects spreads."""
        config = MarketMakingConfig(
            strategy_type=MarketMakingType.INVENTORY,
            quote_refresh_seconds=0,
            inventory_skew=0.5,
            max_position=Decimal("10.0"),
        )
        strategy = MarketMakingStrategy("ETH-USD-PERP", config)
        market = MarketData(
            timestamp=datetime.now(),
            bid=Decimal("1999.00"),
            ask=Decimal("2001.00"),
            last_price=Decimal("2000.00"),
        )
        # Get quotes with neutral inventory
        quotes_neutral = strategy.on_market_data(market)
        neutral_spread = quotes_neutral.ask.price - quotes_neutral.bid.price
        # Build long inventory
        for _ in range(5):
            strategy.inventory_manager.record_fill("bid", Decimal("2000.00"), Decimal("1.0"))
        strategy.last_quote_time = None  # Reset to force quote refresh
        # Get quotes with long inventory
        quotes_long = strategy.on_market_data(market)
        long_bid_price = quotes_long.bid.price
        # Bid should be lower (wider bid spread) when long
        assert long_bid_price <= quotes_neutral.bid.price

    def test_multiple_fills(self):
        """Test multiple fills and PnL tracking."""
        config = MarketMakingConfig(quote_refresh_seconds=0)
        strategy = MarketMakingStrategy("ETH-USD-PERP", config)
        market = MarketData(
            timestamp=datetime.now(),
            bid=Decimal("1999.00"),
            ask=Decimal("2001.00"),
            last_price=Decimal("2000.00"),
        )
        # Execute multiple trades
        for i in range(5):
            strategy.last_quote_time = None
            quotes = strategy.on_market_data(market)
            strategy.on_fill(quotes.bid.quote_id, quotes.bid.price, Decimal("0.1"))
            strategy.on_fill(quotes.ask.quote_id, quotes.ask.price, Decimal("0.1"))
        assert strategy.metrics.filled_quotes == 10
        assert strategy.inventory_manager.metrics.total_buys == 5
        assert strategy.inventory_manager.metrics.total_sells == 5
