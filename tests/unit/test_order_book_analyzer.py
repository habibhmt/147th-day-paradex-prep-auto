"""Unit tests for Order Book Analyzer."""

import pytest
import time
from decimal import Decimal
from unittest.mock import MagicMock

from src.analytics.order_book_analyzer import (
    OrderBookSide,
    LiquidityLevel,
    MarketCondition,
    OrderLevel,
    SpreadMetrics,
    DepthMetrics,
    ImbalanceMetrics,
    PriceImpact,
    LiquidityMetrics,
    SupportResistance,
    VWAPResult,
    OrderBookSnapshot,
    OrderBookAnalyzer,
    RealTimeOrderBook,
    MultiMarketOrderBook,
    get_order_book_analyzer,
    reset_order_book_analyzer,
)


class TestOrderBookSide:
    """Tests for OrderBookSide enum."""

    def test_side_values(self):
        """Should have expected side values."""
        assert OrderBookSide.BID.value == "bid"
        assert OrderBookSide.ASK.value == "ask"


class TestLiquidityLevel:
    """Tests for LiquidityLevel enum."""

    def test_level_values(self):
        """Should have expected level values."""
        assert LiquidityLevel.VERY_LOW.value == "very_low"
        assert LiquidityLevel.LOW.value == "low"
        assert LiquidityLevel.MEDIUM.value == "medium"
        assert LiquidityLevel.HIGH.value == "high"
        assert LiquidityLevel.VERY_HIGH.value == "very_high"


class TestMarketCondition:
    """Tests for MarketCondition enum."""

    def test_condition_values(self):
        """Should have expected condition values."""
        assert MarketCondition.BALANCED.value == "balanced"
        assert MarketCondition.BID_HEAVY.value == "bid_heavy"
        assert MarketCondition.ASK_HEAVY.value == "ask_heavy"
        assert MarketCondition.THIN.value == "thin"
        assert MarketCondition.THICK.value == "thick"


class TestOrderLevel:
    """Tests for OrderLevel dataclass."""

    def test_create_order_level(self):
        """Should create order level."""
        level = OrderLevel(
            price=Decimal("50000"),
            size=Decimal("10"),
            side=OrderBookSide.BID,
        )

        assert level.price == Decimal("50000")
        assert level.size == Decimal("10")

    def test_notional_calculation(self):
        """Should calculate notional value."""
        level = OrderLevel(
            price=Decimal("50000"),
            size=Decimal("2"),
            side=OrderBookSide.BID,
        )

        assert level.notional == Decimal("100000")

    def test_to_dict(self):
        """Should convert to dictionary."""
        level = OrderLevel(
            price=Decimal("50000"),
            size=Decimal("10"),
            side=OrderBookSide.ASK,
        )

        d = level.to_dict()

        assert d["price"] == "50000"
        assert d["side"] == "ask"


class TestSpreadMetrics:
    """Tests for SpreadMetrics dataclass."""

    def test_create_spread_metrics(self):
        """Should create spread metrics."""
        metrics = SpreadMetrics(
            bid_price=Decimal("49990"),
            ask_price=Decimal("50010"),
            spread=Decimal("20"),
            spread_pct=0.04,
            spread_bps=4.0,
            mid_price=Decimal("50000"),
        )

        assert metrics.spread == Decimal("20")
        assert metrics.spread_bps == 4.0

    def test_to_dict(self):
        """Should convert to dictionary."""
        metrics = SpreadMetrics(
            bid_price=Decimal("100"),
            ask_price=Decimal("101"),
        )

        d = metrics.to_dict()

        assert "spread_pct" in d
        assert "mid_price" in d


class TestDepthMetrics:
    """Tests for DepthMetrics dataclass."""

    def test_create_depth_metrics(self):
        """Should create depth metrics."""
        metrics = DepthMetrics(
            bid_depth=Decimal("100"),
            ask_depth=Decimal("80"),
            total_depth=Decimal("180"),
            depth_ratio=1.25,
            depth_imbalance=0.11,
        )

        assert metrics.total_depth == Decimal("180")
        assert metrics.depth_ratio == 1.25

    def test_to_dict(self):
        """Should convert to dictionary."""
        metrics = DepthMetrics(bid_levels=10, ask_levels=10)

        d = metrics.to_dict()

        assert d["bid_levels"] == 10


class TestImbalanceMetrics:
    """Tests for ImbalanceMetrics dataclass."""

    def test_create_imbalance_metrics(self):
        """Should create imbalance metrics."""
        metrics = ImbalanceMetrics(
            volume_imbalance=0.25,
            pressure_side=OrderBookSide.BID,
            pressure_strength=0.25,
        )

        assert metrics.volume_imbalance == 0.25
        assert metrics.pressure_side == OrderBookSide.BID

    def test_to_dict(self):
        """Should convert to dictionary."""
        metrics = ImbalanceMetrics(
            top_level_imbalance=0.1,
            pressure_side=OrderBookSide.ASK,
        )

        d = metrics.to_dict()

        assert d["pressure_side"] == "ask"


class TestPriceImpact:
    """Tests for PriceImpact dataclass."""

    def test_create_price_impact(self):
        """Should create price impact."""
        impact = PriceImpact(
            size=Decimal("10"),
            side=OrderBookSide.BID,
            average_price=Decimal("50010"),
            impact_pct=0.02,
            fully_filled=True,
        )

        assert impact.impact_pct == 0.02
        assert impact.fully_filled is True

    def test_to_dict(self):
        """Should convert to dictionary."""
        impact = PriceImpact(
            size=Decimal("5"),
            side=OrderBookSide.ASK,
            levels_consumed=3,
        )

        d = impact.to_dict()

        assert d["levels_consumed"] == 3


class TestLiquidityMetrics:
    """Tests for LiquidityMetrics dataclass."""

    def test_create_liquidity_metrics(self):
        """Should create liquidity metrics."""
        metrics = LiquidityMetrics(
            liquidity_level=LiquidityLevel.HIGH,
            liquidity_score=80.0,
        )

        assert metrics.liquidity_level == LiquidityLevel.HIGH

    def test_to_dict(self):
        """Should convert to dictionary."""
        metrics = LiquidityMetrics(
            liquidity_level=LiquidityLevel.MEDIUM,
            liquidity_score=60.0,
        )

        d = metrics.to_dict()

        assert d["liquidity_level"] == "medium"


class TestSupportResistance:
    """Tests for SupportResistance dataclass."""

    def test_create_support_resistance(self):
        """Should create support/resistance."""
        sr = SupportResistance(
            support_levels=[Decimal("49000"), Decimal("48000")],
            resistance_levels=[Decimal("51000"), Decimal("52000")],
            strongest_support=Decimal("49000"),
            strongest_resistance=Decimal("51000"),
        )

        assert len(sr.support_levels) == 2
        assert sr.strongest_support == Decimal("49000")

    def test_to_dict(self):
        """Should convert to dictionary."""
        sr = SupportResistance(
            support_levels=[Decimal("100")],
            resistance_levels=[Decimal("110")],
        )

        d = sr.to_dict()

        assert "support_levels" in d
        assert d["support_levels"] == ["100"]


class TestVWAPResult:
    """Tests for VWAPResult dataclass."""

    def test_create_vwap_result(self):
        """Should create VWAP result."""
        result = VWAPResult(
            vwap=Decimal("50050"),
            total_volume=Decimal("100"),
            deviation_from_mid=0.1,
        )

        assert result.vwap == Decimal("50050")

    def test_to_dict(self):
        """Should convert to dictionary."""
        result = VWAPResult(
            vwap=Decimal("100"),
            total_volume=Decimal("50"),
        )

        d = result.to_dict()

        assert d["vwap"] == "100"


class TestOrderBookSnapshot:
    """Tests for OrderBookSnapshot dataclass."""

    def test_create_snapshot(self):
        """Should create snapshot."""
        snapshot = OrderBookSnapshot(
            market="BTC-USD-PERP",
            bids=[OrderLevel(Decimal("50000"), Decimal("10"), OrderBookSide.BID)],
            asks=[OrderLevel(Decimal("50010"), Decimal("10"), OrderBookSide.ASK)],
        )

        assert snapshot.market == "BTC-USD-PERP"
        assert len(snapshot.bids) == 1

    def test_best_bid(self):
        """Should get best bid."""
        snapshot = OrderBookSnapshot(
            bids=[
                OrderLevel(Decimal("50000"), Decimal("10"), OrderBookSide.BID),
                OrderLevel(Decimal("49990"), Decimal("5"), OrderBookSide.BID),
            ],
        )

        assert snapshot.best_bid.price == Decimal("50000")

    def test_best_ask(self):
        """Should get best ask."""
        snapshot = OrderBookSnapshot(
            asks=[
                OrderLevel(Decimal("50010"), Decimal("10"), OrderBookSide.ASK),
                OrderLevel(Decimal("50020"), Decimal("5"), OrderBookSide.ASK),
            ],
        )

        assert snapshot.best_ask.price == Decimal("50010")

    def test_to_dict(self):
        """Should convert to dictionary."""
        snapshot = OrderBookSnapshot(market="ETH-USD-PERP")

        d = snapshot.to_dict()

        assert d["market"] == "ETH-USD-PERP"


class TestOrderBookAnalyzer:
    """Tests for OrderBookAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer."""
        return OrderBookAnalyzer()

    @pytest.fixture
    def sample_book(self):
        """Create sample order book."""
        bids = [
            (Decimal("50000"), Decimal("10")),
            (Decimal("49990"), Decimal("15")),
            (Decimal("49980"), Decimal("20")),
            (Decimal("49970"), Decimal("25")),
            (Decimal("49960"), Decimal("30")),
        ]
        asks = [
            (Decimal("50010"), Decimal("12")),
            (Decimal("50020"), Decimal("18")),
            (Decimal("50030"), Decimal("22")),
            (Decimal("50040"), Decimal("28")),
            (Decimal("50050"), Decimal("35")),
        ]
        return bids, asks

    def test_analyze(self, analyzer, sample_book):
        """Should analyze order book."""
        bids, asks = sample_book

        snapshot = analyzer.analyze(bids, asks, "BTC-USD-PERP")

        assert snapshot.market == "BTC-USD-PERP"
        assert len(snapshot.bids) == 5
        assert len(snapshot.asks) == 5
        assert snapshot.spread is not None
        assert snapshot.depth is not None

    def test_calculate_spread(self, analyzer, sample_book):
        """Should calculate spread metrics."""
        bids, asks = sample_book
        bid_levels = [OrderLevel(p, s, OrderBookSide.BID) for p, s in bids]
        ask_levels = [OrderLevel(p, s, OrderBookSide.ASK) for p, s in asks]

        spread = analyzer.calculate_spread(bid_levels, ask_levels)

        assert spread.bid_price == Decimal("50000")
        assert spread.ask_price == Decimal("50010")
        assert spread.spread == Decimal("10")
        assert spread.mid_price == Decimal("50005")

    def test_calculate_spread_empty(self, analyzer):
        """Should handle empty order book."""
        spread = analyzer.calculate_spread([], [])

        assert spread.spread == Decimal("0")

    def test_calculate_depth(self, analyzer, sample_book):
        """Should calculate depth metrics."""
        bids, asks = sample_book
        bid_levels = [OrderLevel(p, s, OrderBookSide.BID) for p, s in bids]
        ask_levels = [OrderLevel(p, s, OrderBookSide.ASK) for p, s in asks]

        depth = analyzer.calculate_depth(bid_levels, ask_levels)

        assert depth.bid_depth == Decimal("100")  # 10+15+20+25+30
        assert depth.ask_depth == Decimal("115")  # 12+18+22+28+35
        assert depth.bid_levels == 5
        assert depth.ask_levels == 5

    def test_calculate_imbalance(self, analyzer, sample_book):
        """Should calculate imbalance metrics."""
        bids, asks = sample_book
        bid_levels = [OrderLevel(p, s, OrderBookSide.BID) for p, s in bids]
        ask_levels = [OrderLevel(p, s, OrderBookSide.ASK) for p, s in asks]

        imbalance = analyzer.calculate_imbalance(bid_levels, ask_levels)

        # bid_depth=100, ask_depth=115
        # imbalance = (100-115)/(100+115) = -15/215 = -0.069
        assert imbalance.volume_imbalance < 0

    def test_calculate_imbalance_with_pressure(self, analyzer):
        """Should detect pressure side."""
        bids = [OrderLevel(Decimal("100"), Decimal("100"), OrderBookSide.BID)]
        asks = [OrderLevel(Decimal("101"), Decimal("10"), OrderBookSide.ASK)]

        imbalance = analyzer.calculate_imbalance(bids, asks)

        assert imbalance.pressure_side == OrderBookSide.BID
        assert imbalance.pressure_strength > 0

    def test_calculate_liquidity(self, analyzer, sample_book):
        """Should calculate liquidity metrics."""
        bids, asks = sample_book
        snapshot = analyzer.analyze(bids, asks)

        liquidity = analyzer.calculate_liquidity(snapshot)

        assert liquidity.liquidity_level is not None
        assert liquidity.liquidity_score > 0

    def test_estimate_price_impact_buy(self, analyzer, sample_book):
        """Should estimate price impact for buy."""
        bids, asks = sample_book
        analyzer.analyze(bids, asks)

        impact = analyzer.estimate_price_impact(
            size=Decimal("20"),
            side=OrderBookSide.BID,
        )

        # Buying 20: 12 at 50010, 8 at 50020
        assert impact.fully_filled is True
        assert impact.levels_consumed == 2

    def test_estimate_price_impact_sell(self, analyzer, sample_book):
        """Should estimate price impact for sell."""
        bids, asks = sample_book
        analyzer.analyze(bids, asks)

        impact = analyzer.estimate_price_impact(
            size=Decimal("30"),
            side=OrderBookSide.ASK,
        )

        # Selling 30: 10 at 50000, 15 at 49990, 5 at 49980
        assert impact.fully_filled is True
        assert impact.levels_consumed == 3

    def test_estimate_price_impact_partial_fill(self, analyzer):
        """Should handle partial fills."""
        bids = [(Decimal("100"), Decimal("5"))]
        asks = [(Decimal("101"), Decimal("5"))]
        analyzer.analyze(bids, asks)

        impact = analyzer.estimate_price_impact(
            size=Decimal("10"),
            side=OrderBookSide.BID,
        )

        assert impact.fully_filled is False
        assert impact.remaining_size == Decimal("5")

    def test_calculate_vwap(self, analyzer, sample_book):
        """Should calculate VWAP."""
        bids, asks = sample_book
        ask_levels = [OrderLevel(p, s, OrderBookSide.ASK) for p, s in asks]

        vwap = analyzer.calculate_vwap(ask_levels)

        assert vwap.vwap > Decimal("0")
        assert vwap.total_volume == Decimal("115")

    def test_calculate_vwap_with_limit(self, analyzer, sample_book):
        """Should respect size limit."""
        bids, asks = sample_book
        ask_levels = [OrderLevel(p, s, OrderBookSide.ASK) for p, s in asks]

        vwap = analyzer.calculate_vwap(ask_levels, up_to_size=Decimal("12"))

        assert vwap.total_volume == Decimal("12")

    def test_calculate_vwap_empty(self, analyzer):
        """Should handle empty levels."""
        vwap = analyzer.calculate_vwap([])

        assert vwap.vwap == Decimal("0")

    def test_find_support_resistance(self, analyzer):
        """Should find support/resistance levels."""
        bids = [
            (Decimal("50000"), Decimal("10")),
            (Decimal("49900"), Decimal("50")),  # Strong support
            (Decimal("49800"), Decimal("10")),
        ]
        asks = [
            (Decimal("50100"), Decimal("10")),
            (Decimal("50200"), Decimal("60")),  # Strong resistance
            (Decimal("50300"), Decimal("10")),
        ]
        analyzer.analyze(bids, asks)

        sr = analyzer.find_support_resistance(strength_threshold=Decimal("2"))

        assert len(sr.support_levels) >= 1
        assert len(sr.resistance_levels) >= 1

    def test_get_market_condition_balanced(self):
        """Should detect balanced market."""
        # Use medium liquidity thresholds for testing
        analyzer = OrderBookAnalyzer(
            liquidity_thresholds={
                "very_low": Decimal("100"),
                "low": Decimal("500"),
                "medium": Decimal("1000"),
                "high": Decimal("50000"),
                "very_high": Decimal("100000"),
            }
        )
        bids = [
            (Decimal("100"), Decimal("50")),
            (Decimal("99"), Decimal("50")),
        ]
        asks = [
            (Decimal("101"), Decimal("50")),
            (Decimal("102"), Decimal("50")),
        ]
        snapshot = analyzer.analyze(bids, asks)

        condition = analyzer.get_market_condition(snapshot)

        # Medium liquidity with balanced imbalance
        assert condition == MarketCondition.BALANCED

    def test_get_market_condition_bid_heavy(self):
        """Should detect bid-heavy market."""
        analyzer = OrderBookAnalyzer(
            liquidity_thresholds={
                "very_low": Decimal("100"),
                "low": Decimal("500"),
                "medium": Decimal("1000"),
                "high": Decimal("5000"),
                "very_high": Decimal("10000"),
            }
        )
        bids = [(Decimal("100"), Decimal("100"))]
        asks = [(Decimal("101"), Decimal("10"))]
        snapshot = analyzer.analyze(bids, asks)

        condition = analyzer.get_market_condition(snapshot)

        assert condition == MarketCondition.BID_HEAVY

    def test_get_market_condition_ask_heavy(self):
        """Should detect ask-heavy market."""
        analyzer = OrderBookAnalyzer(
            liquidity_thresholds={
                "very_low": Decimal("100"),
                "low": Decimal("500"),
                "medium": Decimal("1000"),
                "high": Decimal("5000"),
                "very_high": Decimal("10000"),
            }
        )
        bids = [(Decimal("100"), Decimal("10"))]
        asks = [(Decimal("101"), Decimal("100"))]
        snapshot = analyzer.analyze(bids, asks)

        condition = analyzer.get_market_condition(snapshot)

        assert condition == MarketCondition.ASK_HEAVY

    def test_add_callback(self, analyzer, sample_book):
        """Should call callbacks on analyze."""
        callback = MagicMock()
        analyzer.add_callback(callback)

        bids, asks = sample_book
        analyzer.analyze(bids, asks)

        callback.assert_called_once()

    def test_remove_callback(self, analyzer):
        """Should remove callback."""
        callback = MagicMock()
        analyzer.add_callback(callback)
        analyzer.remove_callback(callback)

        analyzer.analyze([(Decimal("100"), Decimal("1"))], [(Decimal("101"), Decimal("1"))])

        callback.assert_not_called()

    def test_get_latest_snapshot(self, analyzer, sample_book):
        """Should get latest snapshot."""
        bids, asks = sample_book
        analyzer.analyze(bids, asks, "BTC-USD-PERP")

        snapshot = analyzer.get_latest_snapshot()

        assert snapshot is not None
        assert snapshot.market == "BTC-USD-PERP"

    def test_get_snapshots(self, analyzer):
        """Should get recent snapshots."""
        for i in range(5):
            analyzer.analyze(
                [(Decimal(str(100 + i)), Decimal("1"))],
                [(Decimal(str(101 + i)), Decimal("1"))],
            )

        snapshots = analyzer.get_snapshots(limit=3)

        assert len(snapshots) == 3

    def test_get_spread_history(self, analyzer):
        """Should get spread history."""
        for i in range(5):
            analyzer.analyze(
                [(Decimal("100"), Decimal("1"))],
                [(Decimal(str(101 + i)), Decimal("1"))],
            )

        history = analyzer.get_spread_history(limit=5)

        assert len(history) == 5

    def test_get_average_spread(self, analyzer):
        """Should calculate average spread."""
        for _ in range(5):
            analyzer.analyze(
                [(Decimal("100"), Decimal("1"))],
                [(Decimal("102"), Decimal("1"))],  # 2% spread
            )

        avg = analyzer.get_average_spread()

        assert avg > 0

    def test_clear_history(self, analyzer, sample_book):
        """Should clear history."""
        bids, asks = sample_book
        analyzer.analyze(bids, asks)

        analyzer.clear_history()

        assert analyzer.get_latest_snapshot() is None


class TestRealTimeOrderBook:
    """Tests for RealTimeOrderBook."""

    @pytest.fixture
    def book(self):
        """Create real-time order book."""
        return RealTimeOrderBook("BTC-USD-PERP")

    def test_set_snapshot(self, book):
        """Should set full snapshot."""
        bids = [
            (Decimal("50000"), Decimal("10")),
            (Decimal("49990"), Decimal("15")),
        ]
        asks = [
            (Decimal("50010"), Decimal("12")),
            (Decimal("50020"), Decimal("18")),
        ]

        snapshot = book.set_snapshot(bids, asks, sequence=1)

        assert snapshot is not None
        assert len(snapshot.bids) == 2

    def test_apply_update(self, book):
        """Should apply incremental update."""
        # Initial snapshot
        book.set_snapshot(
            [(Decimal("100"), Decimal("10"))],
            [(Decimal("101"), Decimal("10"))],
            sequence=1,
        )

        # Update
        snapshot = book.apply_update(
            [(Decimal("100"), Decimal("20"))],  # Update existing
            [(Decimal("102"), Decimal("5"))],  # Add new
            sequence=2,
        )

        assert snapshot is not None

    def test_apply_update_delete(self, book):
        """Should handle deletions."""
        book.set_snapshot(
            [(Decimal("100"), Decimal("10"))],
            [(Decimal("101"), Decimal("10"))],
            sequence=1,
        )

        # Delete bid (size=0)
        book.apply_update(
            [(Decimal("100"), Decimal("0"))],
            [],
            sequence=2,
        )

        assert book.get_best_bid() is None

    def test_sequence_validation(self, book):
        """Should reject old sequences."""
        book.set_snapshot(
            [(Decimal("100"), Decimal("10"))],
            [(Decimal("101"), Decimal("10"))],
            sequence=5,
        )

        # Old sequence
        result = book.apply_update(
            [(Decimal("100"), Decimal("20"))],
            [],
            sequence=3,
        )

        assert result is None

    def test_get_best_bid(self, book):
        """Should get best bid."""
        book.set_snapshot(
            [(Decimal("100"), Decimal("10")), (Decimal("99"), Decimal("5"))],
            [(Decimal("101"), Decimal("10"))],
        )

        best = book.get_best_bid()

        assert best == (Decimal("100"), Decimal("10"))

    def test_get_best_ask(self, book):
        """Should get best ask."""
        book.set_snapshot(
            [(Decimal("100"), Decimal("10"))],
            [(Decimal("101"), Decimal("10")), (Decimal("102"), Decimal("5"))],
        )

        best = book.get_best_ask()

        assert best == (Decimal("101"), Decimal("10"))

    def test_get_mid_price(self, book):
        """Should get mid price."""
        book.set_snapshot(
            [(Decimal("100"), Decimal("10"))],
            [(Decimal("102"), Decimal("10"))],
        )

        mid = book.get_mid_price()

        assert mid == Decimal("101")

    def test_get_spread(self, book):
        """Should get spread."""
        book.set_snapshot(
            [(Decimal("100"), Decimal("10"))],
            [(Decimal("102"), Decimal("10"))],
        )

        spread = book.get_spread()

        assert spread == Decimal("2")

    def test_add_callback(self, book):
        """Should call callbacks."""
        callback = MagicMock()
        book.add_callback(callback)

        book.set_snapshot(
            [(Decimal("100"), Decimal("10"))],
            [(Decimal("101"), Decimal("10"))],
        )

        callback.assert_called_once()

    def test_get_depth(self, book):
        """Should get order book depth."""
        book.set_snapshot(
            [(Decimal("100"), Decimal("10")), (Decimal("99"), Decimal("5"))],
            [(Decimal("101"), Decimal("10")), (Decimal("102"), Decimal("5"))],
        )

        bid_depth = book.get_depth(OrderBookSide.BID, levels=2)
        ask_depth = book.get_depth(OrderBookSide.ASK, levels=2)

        assert len(bid_depth) == 2
        assert len(ask_depth) == 2
        assert bid_depth[0][0] == Decimal("100")  # Best bid first
        assert ask_depth[0][0] == Decimal("101")  # Best ask first


class TestMultiMarketOrderBook:
    """Tests for MultiMarketOrderBook."""

    @pytest.fixture
    def multi_book(self):
        """Create multi-market order book."""
        return MultiMarketOrderBook()

    def test_get_or_create(self, multi_book):
        """Should create book for new market."""
        book = multi_book.get_or_create("BTC-USD-PERP")

        assert book is not None
        assert book.market == "BTC-USD-PERP"

    def test_get_or_create_existing(self, multi_book):
        """Should return existing book."""
        book1 = multi_book.get_or_create("BTC-USD-PERP")
        book2 = multi_book.get_or_create("BTC-USD-PERP")

        assert book1 is book2

    def test_update_snapshot(self, multi_book):
        """Should update with snapshot."""
        snapshot = multi_book.update(
            "BTC-USD-PERP",
            [(Decimal("50000"), Decimal("10"))],
            [(Decimal("50010"), Decimal("10"))],
            is_snapshot=True,
        )

        assert snapshot is not None
        assert snapshot.market == "BTC-USD-PERP"

    def test_update_incremental(self, multi_book):
        """Should apply incremental update."""
        # Initial snapshot
        multi_book.update(
            "BTC-USD-PERP",
            [(Decimal("50000"), Decimal("10"))],
            [(Decimal("50010"), Decimal("10"))],
            is_snapshot=True,
            sequence=1,
        )

        # Incremental update
        snapshot = multi_book.update(
            "BTC-USD-PERP",
            [(Decimal("50000"), Decimal("20"))],
            [],
            is_snapshot=False,
            sequence=2,
        )

        assert snapshot is not None

    def test_get_all_snapshots(self, multi_book):
        """Should get all market snapshots."""
        multi_book.update("BTC-USD-PERP", [(Decimal("50000"), Decimal("10"))], [(Decimal("50010"), Decimal("10"))], is_snapshot=True)
        multi_book.update("ETH-USD-PERP", [(Decimal("3000"), Decimal("10"))], [(Decimal("3001"), Decimal("10"))], is_snapshot=True)

        snapshots = multi_book.get_all_snapshots()

        assert "BTC-USD-PERP" in snapshots
        assert "ETH-USD-PERP" in snapshots

    def test_add_callback(self, multi_book):
        """Should add callback to all markets."""
        callback = MagicMock()
        multi_book.add_callback(callback)

        multi_book.update("BTC-USD-PERP", [(Decimal("50000"), Decimal("10"))], [(Decimal("50010"), Decimal("10"))], is_snapshot=True)

        callback.assert_called()

    def test_get_market_count(self, multi_book):
        """Should count tracked markets."""
        multi_book.get_or_create("BTC-USD-PERP")
        multi_book.get_or_create("ETH-USD-PERP")

        assert multi_book.get_market_count() == 2

    def test_get_markets(self, multi_book):
        """Should list tracked markets."""
        multi_book.get_or_create("BTC-USD-PERP")
        multi_book.get_or_create("ETH-USD-PERP")

        markets = multi_book.get_markets()

        assert "BTC-USD-PERP" in markets
        assert "ETH-USD-PERP" in markets

    def test_remove_market(self, multi_book):
        """Should remove market."""
        multi_book.get_or_create("BTC-USD-PERP")

        result = multi_book.remove_market("BTC-USD-PERP")

        assert result is True
        assert multi_book.get_market_count() == 0

    def test_clear(self, multi_book):
        """Should clear all markets."""
        multi_book.get_or_create("BTC-USD-PERP")
        multi_book.get_or_create("ETH-USD-PERP")

        multi_book.clear()

        assert multi_book.get_market_count() == 0


class TestGlobalOrderBookAnalyzer:
    """Tests for global analyzer functions."""

    def test_get_order_book_analyzer(self):
        """Should get or create analyzer."""
        reset_order_book_analyzer()

        a1 = get_order_book_analyzer()
        a2 = get_order_book_analyzer()

        assert a1 is a2

    def test_reset_order_book_analyzer(self):
        """Should reset analyzer."""
        a1 = get_order_book_analyzer()
        reset_order_book_analyzer()
        a2 = get_order_book_analyzer()

        assert a1 is not a2


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_order_book(self):
        """Should handle empty order book."""
        analyzer = OrderBookAnalyzer()

        snapshot = analyzer.analyze([], [])

        assert len(snapshot.bids) == 0
        assert len(snapshot.asks) == 0

    def test_single_level(self):
        """Should handle single level."""
        analyzer = OrderBookAnalyzer()

        snapshot = analyzer.analyze(
            [(Decimal("100"), Decimal("10"))],
            [(Decimal("101"), Decimal("10"))],
        )

        assert snapshot.spread.spread == Decimal("1")

    def test_zero_size_levels(self):
        """Should filter zero size levels."""
        book = RealTimeOrderBook("TEST")

        book.set_snapshot(
            [(Decimal("100"), Decimal("0")), (Decimal("99"), Decimal("10"))],
            [(Decimal("101"), Decimal("10"))],
        )

        best_bid = book.get_best_bid()

        assert best_bid == (Decimal("99"), Decimal("10"))

    def test_very_wide_spread(self):
        """Should handle wide spread."""
        analyzer = OrderBookAnalyzer()

        snapshot = analyzer.analyze(
            [(Decimal("100"), Decimal("10"))],
            [(Decimal("200"), Decimal("10"))],
        )

        assert snapshot.spread.spread_pct > 50

    def test_very_deep_book(self):
        """Should handle deep order book."""
        analyzer = OrderBookAnalyzer(depth_levels=100)

        bids = [(Decimal(str(100 - i * 0.1)), Decimal("10")) for i in range(50)]
        asks = [(Decimal(str(101 + i * 0.1)), Decimal("10")) for i in range(50)]

        snapshot = analyzer.analyze(bids, asks)

        assert len(snapshot.bids) == 50

    def test_price_impact_large_order(self):
        """Should handle large order impact."""
        analyzer = OrderBookAnalyzer()

        bids = [(Decimal(str(100 - i)), Decimal("10")) for i in range(10)]
        asks = [(Decimal(str(101 + i)), Decimal("10")) for i in range(10)]

        analyzer.analyze(bids, asks)

        # Order larger than total liquidity
        impact = analyzer.estimate_price_impact(
            size=Decimal("200"),
            side=OrderBookSide.BID,
        )

        assert impact.fully_filled is False
        assert impact.levels_consumed == 10

    def test_microprice_calculation(self):
        """Should calculate microprice."""
        analyzer = OrderBookAnalyzer()

        # Asymmetric sizes
        snapshot = analyzer.analyze(
            [(Decimal("100"), Decimal("10"))],  # Small bid
            [(Decimal("101"), Decimal("100"))],  # Large ask
        )

        # Microprice should be closer to bid due to larger ask size
        assert snapshot.spread.microprice < snapshot.spread.mid_price

    def test_support_resistance_empty(self):
        """Should handle empty book for S/R."""
        analyzer = OrderBookAnalyzer()

        sr = analyzer.find_support_resistance()

        assert len(sr.support_levels) == 0
        assert len(sr.resistance_levels) == 0
